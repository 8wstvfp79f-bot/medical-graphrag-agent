# FastAPI `/chat`、异步检索与 SSE

## 这一阶段做了什么

第 8 阶段把前七个阶段的“证据流水线”变成一个真正可调用的问答接口：

```text
POST /chat
   ↓
请求结构校验
   ↓
疾病实体 + 意图 + 严格工具调用
   ↓
Milvus 文本检索 ─┐
                  ├─ asyncio 并发 → RRF → BGE 重排 → 上下文预算
Neo4j 图谱检索 ──┘
   ↓
抽取式带引用回答
   ↓
完整 JSON，或 SSE 分阶段事件
```

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `src/api/schemas.py` | 定义 `/chat` 请求和响应结构；限制字符串长度、Top-K 1–50、元数据字段与未知字段 |
| `src/api/service.py` | 串联规划、异步智能体、答案生成与响应组装；为每次请求生成 `request_id` |
| `src/api/runtime.py` | 服务启动时一次性加载本地 BGE、Reranker、Milvus、Neo4j 和疾病词典；关闭时释放连接 |
| `src/api/app.py` | 提供 `/health`、`/chat`、OpenAPI 文档和 SSE 编码 |
| `src/generation/answer_generator.py` | 定义可替换的答案生成器；既支持本地抽取式回答，也支持带引用校验的兼容对话模型 |
| `src/llm/client.py` | 调用本地或托管的 OpenAI 兼容对话接口；密钥只进入请求头，不写入结果 |
| `src/retrieval/hybrid_retriever.py` | `hybrid_search_async` 同时启动向量与图谱分支，完成后统一融合 |

## 启动和网页调试

先确认 Milvus 的 `19530` 端口、Neo4j 的 `7687` 端口可用，而且 `.env` 中模型目录、Collection 和数据库凭据正确。

```powershell
python -m src.api.app
```

启动完成后：

- `http://127.0.0.1:8000/health`：只检查服务是否就绪；
- `http://127.0.0.1:8000/docs`：FastAPI 自动生成的交互式网页，可展开 `POST /chat`，点击 **Try it out** 后粘贴 JSON；
- `http://127.0.0.1:8000/openapi.json`：机器可读接口定义。

## 请求示例

```json
{
  "query": "百日咳有哪些典型症状？",
  "top_k": 10,
  "vector_top_k": 6,
  "graph_top_k": 6,
  "metadata_filter": {
    "department": ["儿科", "小儿内科"]
  },
  "allow_partial": false,
  "stream": false
}
```

`disease_name` 可由问题自动识别，也可在请求顶层显式传入。混合检索计划会把最终疾病名同步到向量元数据过滤条件与图谱参数中。若没有疾病名或同时识别出多个独立疾病，接口返回 `needs_clarification`，不会访问数据库。

## 非流式 JSON 怎么看

| 字段 | 零基础解释 |
| --- | --- |
| `status` | `completed` 表示整条问答链完成；`needs_clarification` 表示需要补充疾病名 |
| `answer` | 给用户看的答案；其中 `[E1]` 指向证据编号 |
| `answer_generation` | 当前答案方式和实际使用的证据 ID；默认是抽取式后端，配置真实模型后可为 `llm_grounded` |
| `sources` / `citations` | 原始来源列表，以及 E1-En 到证据的对应关系 |
| `evidence` | 上下文预算最终允许进入回答层的证据 |
| `retrieved_chunks` | 来自 Milvus 文本召回的证据 |
| `graph_evidence` | 来自 Neo4j 关系扩展的证据 |
| `query_plan` | 系统识别的疾病、意图、所选工具和参数 |
| `tool_calls` | 实际执行了哪个工具、耗时多少、召回多少条 |
| `retrieval_stats` | 两路候选数、去重数与最终融合数 |
| `context_budget` | 上下文使用了多少估算 token、选择了几条证据 |

## SSE 怎么看

将 `stream` 改为 `true` 后，HTTP 连接不会一次返回一个大 JSON，而会按阶段推送命名事件：

1. `plan`：问题被理解成什么疾病、什么意图；
2. `tool_start`：准备调用哪个检索工具；
3. `tool_result`：检索结束和耗时统计；
4. `citation`：逐条发送最终引用；
5. `token`：分段发送答案文本；当前是文本分块，不冒充模型逐 token 解码；
6. `done`：发送完整的最终响应；
7. `error`：任一阶段异常时给出错误类型和消息。

## 并发与安全边界

- `asyncio` 只并行 Milvus 和 Neo4j 这两个互不依赖的分支；融合、重排、预算与答案仍等待两路结果完成。
- 同步数据库/模型客户端通过 `asyncio.to_thread` 放进工作线程，不阻塞 FastAPI 的事件循环。
- `API_MAX_CONCURRENCY` 默认是 1，用信号量保护共享 GPU 模型；确认显存和线程安全后再提高。
- `allow_partial=false` 时任一检索库失败就结束请求；设为 `true` 时可保留另一分支，并在 `errors` 中明确记录降级原因。
- 当前服务是知识库问答系统，不替代医疗诊断。

## 真实验收结果

在本地BGE、Milvus 3.0和Neo4j上通过HTTP请求验证“百日咳有哪些典型症状？”：

| 项目 | 结果 |
| --- | --- |
| HTTP / 最终状态 | `200` / `completed` |
| 自动工具 | `hybrid_search` |
| 向量 / 图谱候选 | 5 / 6 |
| 融合 / 上下文证据 | 10 / 4 |
| 最终引用 | E1、E2、E4，包含文本和图谱证据 |
| 回答模式 | `extractive-citation-v1` |
| 本机20次暖机后SSE基准 | TTFB P50/P95 1.3/11.2ms；首回答Token P50/P95 55.4/64.8ms |
| SSE | 11 个事件，最终以 `done` 结束 |

自动化测试同时覆盖请求 schema、澄清分支、并发启动、单路降级、引用答案、JSON 与 SSE 契约。运行：

```powershell
python -m unittest discover -s tests -v
```

## 生成模型切换

默认使用不需要密钥的抽取式后端。配置兼容对话服务后可切换到带引用约束的大模型生成：

```dotenv
ANSWER_GENERATOR_BACKEND=openai-compatible
LLM_BASE_URL=http://127.0.0.1:1234
LLM_MODEL=your-chat-model
LLM_API_KEY=
GENERATION_LLM_FALLBACK=true
```

模型只能看到上下文预算选出的 E1-En 证据。返回后会再次校验引用编号；无引用、空回答或引用越界时，默认退回抽取式答案，响应中的 `answer_generation.mode` 会说明实际使用了哪条路径。

性能报告由下列命令重新生成。切换到LLM后必须重跑，不能沿用抽取式后端的55ms首回答Token结果：

```powershell
python -m src.evaluation.benchmark_sse --requests 20 --warmups 3
```

## 当前边界

真实大模型生成适配器已经实现，但只有在配置模型地址并且响应中的 `mode=llm_grounded` 时，该次请求才确实由大模型生成。当前 500 条批量回答和 500 条裁判评分已使用本地模型完成；本页所列 20 次 SSE 性能数据仍来自抽取式后端，切换模型后必须重新测试，不能混用两种延迟口径。
