# Function Calling Agent

## 这一阶段解决什么问题

Phase 6 以前，调用方必须自己填写 `disease_name` 和 `relations`。Phase 7 新增一个“检索调度员”：用户只输入自然语言问题，调度员先判断在问哪个疾病、想了解什么，再生成一个受校验的工具调用，最后复用已有检索、重排和上下文构建代码。

```text
“百日咳有哪些典型症状？”
        ↓
疾病实体：百日咳
查询意图：symptom
图谱关系：has_symptom
工具选择：hybrid_search
        ↓
参数 schema + Python 二次校验
        ↓
Milvus + Neo4j → Weighted RRF → BGE-Reranker → Context Budget
        ↓
evidence_ready + E1/E2/E3
        ↓
AnswerGenerator → FastAPI JSON / SSE
```

## 文件职责

| 文件 | 作用 |
| --- | --- |
| `src/agent/tool_schemas.py` | 定义三个 strict JSON Function schemas；校验工具名、字段、类型、Top-K、metadata、关系白名单和疾病一致性 |
| `src/agent/query_planner.py` | 从 `documents.jsonl` 构建疾病目录；执行最长精确实体匹配、意图识别和 Vector/Hybrid 路由 |
| `src/agent/orchestrator.py` | 执行已校验工具，接入 Hybrid、BGE-Reranker 和 Context Budget，返回 query plan、tool calls、统计和引用证据 |

## 当前意图路由

| 用户问题里的意图 | 关系参数 | 工具 | 原因 |
| --- | --- | --- | --- |
| 症状/表现/征兆/体征 | `has_symptom` | Hybrid | 文本描述和结构化症状都可提供证据 |
| 检查/诊断/确诊/化验/检测 | `diagnosed_by` | Hybrid | 同时检索检查说明和图谱检查实体 |
| 治疗/用药/药物 | `treated_by` | Hybrid | 同时检索治疗文本和药品关系 |
| 科室/挂什么科 | `belongs_to` | Hybrid | 同时检索就诊说明和科室关系 |
| 并发症/后遗症 | `has_complication` | Hybrid | 同时检索风险描述和并发症关系 |
| 病因/预防/传播/介绍 | 无 | Vector | 当前 Neo4j 主链路没有这些关系，避免伪造图谱能力 |

一个问题可以命中多个结构化意图，例如“百日咳有什么症状，需要做什么检查？”会生成 `has_symptom + diagnosed_by`。若一句话出现两个独立疾病，系统返回澄清请求；“副百日咳”与“百日咳”这种包含关系则采用更长的疾病全名，避免子串误判。

## 为什么还需要参数校验

Function Calling 只是让模型或规划器输出结构化 JSON，不代表 JSON 一定正确。`validate_tool_call` 是数据库前的信任边界：

- 只能调用 `vector_search`、`graph_search`、`hybrid_search`；
- 不接受 schema 以外的字段；
- Top-K/limit 必须为 1–50；
- metadata 只能使用 `disease_name/category/department/source`；
- Neo4j 只能使用五类核心医学关系；
- Hybrid 中图分支的 `disease_name` 必须与向量 metadata 的疾病一致；
- Vector 主路由必须带一个精确疾病过滤条件。

因此，未来即使接入外部 LLM 产生工具参数，也必须经过同一个校验器，不能把 LLM 输出直接拼成数据库查询。

## 怎么运行

```powershell
# 只查看计划，不加载 BGE/Reranker，也不访问数据库
python -m src.agent.orchestrator "百日咳有哪些典型症状？" --plan-only

# 执行真实主链路
python -m src.agent.orchestrator "百日咳有哪些典型症状？" `
  --context-max-tokens 1200 `
  --context-max-items 6

# 查看三个 Function Calling schema
python -m src.agent.orchestrator --show-tool-schemas
```

运行配置默认从项目根目录的 `.env` 读取。正式查询应使用与 Milvus Collection 入库时相同的 `bge-base-zh-v1.5` 及 768 维配置；Hash Collection 不进入该主链路。

## 真实验收

2026-09-02 使用“百日咳有哪些典型症状？”完成本地真实验收：

| 项目 | 结果 |
| --- | --- |
| 自动疾病实体 | 百日咳 |
| 自动意图 | `symptom` |
| 自动工具/关系 | `hybrid_search` / `has_symptom` |
| Milvus / Neo4j 候选 | 5 / 6 |
| 融合 / 重排证据 | 10 / 8 |
| Context | 3 条，E1-E3，1177/1200 估算 tokens |
| 分支错误 | 无 |

另用“百日咳的病因是什么？”验证为 `vector_search`，Graph 候选为 0；用“有哪些典型症状？”验证为 `needs_clarification`，未调用任何数据库。

## 当前边界与后续扩展

当前规划器是本地、确定性、无需 API Key 的实现。这意味着路由结果稳定、易测试，但没有声称已经由外部大模型理解任意复杂表达。三个工具的 strict schemas 已可交给后续 LLM adapter；无论由规则还是 LLM 产生参数，都复用同一校验与执行层。

Agent 执行器本身仍以 `evidence_ready` 为职责边界；Phase 8 的 `MedicalChatService` 在它外面调用 `AnswerGenerator`，再由 FastAPI `/chat` 返回 JSON 或 SSE。当前 `extractive-citation-v1` 只抽取已有证据并加入医疗安全声明和事实引用，不需要 API Key。未来可新增 LLM adapter，但必须继续只读取受预算控制的 E1-En 上下文，并沿用参数校验、引用和响应字段。
