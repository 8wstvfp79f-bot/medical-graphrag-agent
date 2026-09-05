# BGE 重排与上下文预算

## 本阶段新增的能力

第 5 阶段已经生成一份可审计的向量与图谱候选证据。第 6 阶段把候选列表转换成长度受控、可供答案生成器使用的输入：

```text
混合候选证据
  → BGE Cross-Encoder 相关性评分
  → Top-K / 可选最低分数
  → 向量与图谱来源配额
  → token 和证据条数预算
  → 上下文块 + E1-En 引用
```

这一阶段仍然输出证据，不直接输出医疗回答。后续的 Function Calling、证据约束生成、FastAPI/SSE 和固定评测流程会继续使用这些结果。

## 向量模型与重排模型的职责区别

| 组件 | 单次读取内容 | 职责 | 输出 |
|---|---|---|---|
| `bge-base-zh-v1.5` 向量模型 | 单独读取问题或分块 | 在 Milvus 的 20,801 条干净向量中快速完成第一阶段召回 | 一个 768 维向量 |
| `bge-reranker-base` Cross-Encoder | 同时读取问题和一条候选证据 | 对小规模候选做更精细的第二阶段相关性排序 | 一个相关性 logit，本项目可用 sigmoid 转为 0–1 分数 |

重排器不会写入 Milvus，也不会替代数据库中保存的向量。它只在混合检索返回有限候选后运行。

## 可审计重排

`src/retrieval/reranker.py` 定义重排后端协议和 `LocalCrossEncoderReranker`。本地适配器具有以下特征：

- 只接受已经存在的本地模型目录，不会静默下载模型；
- 分批处理“问题 + 候选证据”组合，并支持 CUDA FP16；
- 默认最大组合长度为 512；
- 可以把原始 logit 转换为 0–1 sigmoid 分数。

`rerank_evidence` 不会修改输入对象，并同时保留两阶段排名：

- `rank`：原始混合检索名次；
- `retrieval_rank`：对原始名次的明确复制；
- `rerank_rank`：Cross-Encoder 新名次；
- `rerank_score`：模型相关性分数；
- `reranker_model`：重排模型目录名称；
- `excluded_evidence`：被 Top-K 或最低分数排除的证据及原因。

这些分数适合排序和调试，但不是经过校准的医学可信度概率。

## 上下文预算

`src/generation/context_builder.py` 在 `max_tokens` 和 `max_items` 两个硬限制内选择完整证据块。默认先尝试满足 `vector=1`、`graph=1` 的最低来源配额，再按照重排名次填充剩余空间。同时具有向量和图谱来源的混合证据可以满足两个配额。

当前 `estimate_tokens` 使用保守且不绑定 tokenizer 的估算：中文字符和标点逐个计数，ASCII 连续片段按近似值计算。未来可以注入答案模型对应的 tokenizer。无法完整放入预算的证据会被排除，不会在中间静默截断。

返回结果包含：

- 带 `[E1 | type=... | disease=... | source=...]` 头部的上下文；
- 带 `context_rank`、`citation_id` 和 `context_token_count` 的已选证据；
- E1-En 到证据 ID 和来源的紧凑引用映射；
- 被排除的证据 ID；
- 已用、剩余和最大 token 数；
- 已选来源类型数量和未满足配额。

## 运行完整检索链路

先在 `.env` 中配置本地模型，或者通过命令行显式传入：

```dotenv
BGE_MODEL_PATH=C:/path/to/bge-base-zh-v1.5
BGE_RERANKER_MODEL_PATH=C:/path/to/bge-reranker-base
BGE_DEVICE=cuda
BGE_RERANKER_DEVICE=cuda
MILVUS_LOAD_TIMEOUT_SECONDS=60
```

```powershell
python -m src.retrieval.hybrid_retriever "百日咳有哪些典型症状？" `
  --disease-name 百日咳 `
  --relation has_symptom `
  --top-k 10 `
  --vector-top-k 6 `
  --graph-top-k 6 `
  --rerank-top-k 8 `
  --context-max-tokens 1200 `
  --context-max-items 6
```

## 本地真实验收结果

验收使用 Milvus 集合 `medical_chunks_bge_base_zh_v15_phase5`、Neo4j 核心关系 `has_symptom`、本地 `BAAI/bge-base-zh-v1.5` 和本地 `BAAI/bge-reranker-base`，计算设备为 CUDA。

| 阶段 | 结果 |
|---|---|
| 向量候选 | 5 |
| 图谱候选 | 6 |
| 融合后返回证据 | 10 |
| 重排输入 / 返回 | 10 / 8 |
| 被重排 Top-K 排除 | 2 |
| 上下文最终选择 | 3 条：2 条向量 + 1 条图谱 |
| 上下文预算 | 已用 1177 / 最大 1200 / 剩余 23 |
| 引用 | E1-E3 |

直接描述症状的综述从检索第 5 名提升到重排第 1 名，分数为 `0.99681491`。图谱事实 `百日咳 -[has_symptom]-> 吸气时有蝉鸣音` 被保留为 E3，因此最终上下文同时包含描述性文本和结构化症状关系。

## 测试覆盖

单元测试覆盖分数校验、同分稳定顺序、Top-K/最低分排除原因、输入不变性、token 预算、来源配额、混合来源、非法参数，以及从重排到上下文构建的集成链路。本地真实验收另外证明两个本地模型、Milvus、Neo4j 和 CUDA 可以协同运行。
