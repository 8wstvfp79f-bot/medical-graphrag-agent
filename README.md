# Medical GraphRAG Agent

## Project Overview

这是一个面向求职展示的医疗知识增强型 GraphRAG Agent 项目。系统以公开疾病百科数据为知识源，目标是构建“向量检索 + 图谱关系扩展 + 重排 + 受控生成 + 自动评测”的可解释医疗问答后端。


## Why GraphRAG

普通 RAG 主要依赖向量相似度检索，适合找语义相近内容；
GraphRAG 在此基础上引入实体和关系，可以更好处理复杂问题、跨文档关系和多跳推理。

## Architecture

```text
User Query
↓
Disease Entity + Intent Recognition
↓
Validated Function Call
├── vector_search → Query Embedding → Milvus + Metadata Filtering
├── graph_search  → Neo4j Graph Expansion (Core Relations)
└── hybrid_search → Milvus + Neo4j
          ↓
Hybrid Evidence Merge
          ↓
BGE-Reranker + Top-K + Context Budget
          ↓
Evidence-ready Context with E1-En Citations
          ↓
Evidence-bound Answer
(default: local extractive backend; optional real compatible Chat model)
          ↓
FastAPI /chat + asyncio + JSON/SSE
          ↓
Fixed Evaluation Set + Top-3 Hit Rate + Citation/Faithfulness Checks
(250 cases; 500 grounded answers and 500 local LLM Judge scores completed)
```

## Metadata Filtering

`documents.jsonl` 生成的 chunk metadata 固定包含 `disease_name`、`category`、`department`、`source`。`vector_search` 支持按这些字段过滤：字段之间是 AND，同一字段的多个值是 OR。过滤条件会下推为 Milvus JSON expression，并在应用层再次校验，避免相近疾病名、相近类别或错误科室召回进入生成上下文。

```python
from src.retrieval.vector_retriever import vector_search

hits = vector_search(
    "百日咳有哪些症状？",
    backend=milvus_backend,
    top_k=5,
    metadata_filter={
        "disease_name": "百日咳",
        "department": ["传染科", "呼吸内科"],
    },
)
```

每条结果返回 `chunk_id`、`doc_id`、`text`、`score`、`metadata` 和 `matched_metadata`，便于记录和复盘 Bad Case。

生成 metadata-rich chunks：

```bash
python -m src.data.build_chunks --max-chars 800 --overlap-chars 100
```

## Embedding

Phase 2 提供统一的 `EmbeddingModel` 接口：

- `HashEmbeddingModel`：默认离线后端，输出确定、L2 归一化的固定维向量，用于开发、测试和 Milvus 链路联调；它不是语义模型，不能用于最终检索效果或简历指标。
- `LocalSentenceTransformerEmbeddingModel`：仅接受已经存在的本地模型目录，可接入本地 BGE/Sentence-Transformers 模型，不会通过模型名称触发在线下载；BGE 查询会自动添加中文检索指令，文档 chunk 不添加。
- `embed_chunk_records`：按批次处理 chunks，保留原始 metadata，并增加 `embedding` 与 `embedding_metadata`，供 Phase 3 批量写入 Milvus。

离线冒烟验证：

```bash
python -m src.retrieval.embedding --backend hash --dimension 384 --sample-size 5
```

使用本地 BGE 模型（当前正式模型为 `BAAI/bge-base-zh-v1.5`，768维）：

```bash
python -m src.retrieval.embedding --backend bge --model-path /path/to/bge-base-zh-v1.5 --device cuda
```

本地 BGE 模式需要环境中已有 `sentence-transformers`；项目不会自动下载模型。

## Milvus Retrieval

Phase 3 的代码入口是 `src/retrieval/milvus_client.py`。它负责 collection schema、`AUTOINDEX + COSINE`、分批 embedding/upsert、实体计数和 metadata-aware Top-K search。`count` 会区分当前唯一 `chunk_id` 的 `logical_entity_count` 与可能包含 upsert 旧版本的 `storage_row_count`；幂等验收以逻辑数量为准。

Collection 使用 `chunk_id` 作为主键，因此重复 ingestion 会更新同一批记录，而不是不断复制数据。已有 collection 会先检查向量维度；只有显式传入 `--recreate` 才会删除重建。

```bash
# 无需数据库，先验证全部 chunk 是否符合 Milvus schema
python -m src.retrieval.milvus_client validate --backend hash --batch-size 64

# 创建并加载 collection
python -m src.retrieval.milvus_client --uri http://localhost:19530 init

# 先写入 100 条做连接验证
python -m src.retrieval.milvus_client --uri http://localhost:19530 ingest --backend hash --limit 100

# 全量写入清洗后的 20,801 个正式 BGE chunks（独立 Collection）
python -m src.retrieval.milvus_client --uri http://localhost:19530 --collection medical_chunks_bge_base_zh_v15 --dimension 768 ingest --backend bge --model-path /path/to/bge-base-zh-v1.5 --device cuda --batch-size 16

# 正式 BGE + Metadata Filtering 搜索
python -m src.retrieval.milvus_client --uri http://localhost:19530 --collection medical_chunks_bge_base_zh_v15 --dimension 768 search "百日咳有哪些症状" --backend bge --model-path /path/to/bge-base-zh-v1.5 --device cuda --top-k 5 --disease-name 百日咳 --department 儿科
```

Hash Embedding 只能验证数据库链路。正式语义召回必须使用同一份本地 BGE 模型完成 ingestion 和 query，并让 `--dimension` 与模型维度一致。详细配置、字段 schema、安全边界和验收标准见 `docs/milvus_setup.md`。

## Neo4j Graph Retrieval

Phase 4 已把 `triples.csv` 中的五类核心医学关系真实导入 Neo4j。`src/retrieval/neo4j_client.py` 负责连接、唯一性约束、`UNWIND + MERGE` 批量幂等导入与数量统计；`src/retrieval/graph_retriever.py` 负责按精确疾病实体和关系白名单做一跳扩展，并返回 `evidence_type`、`relation`、`entity_type`、`source` 和 `path_text`。

```powershell
$env:NEO4J_PASSWORD = "your-local-password"

python -m src.retrieval.neo4j_client validate
python -m src.retrieval.neo4j_client init
python -m src.retrieval.neo4j_client prune-rejected
python -m src.retrieval.neo4j_client ingest --batch-size 2000
python -m src.retrieval.neo4j_client count
python -m src.retrieval.neo4j_client search "百日咳" `
  --relation has_symptom `
  --relation diagnosed_by `
  --relation belongs_to `
  --limit 12
```

数据质量同步后的真实 Neo4j 验收结果为 24,237 个 typed nodes、182,565 条核心关系，与清洗后的 CSV 完全一致。Neo4j Browser 位于 `http://127.0.0.1:7474`，详细 schema、Docker 启动方式、可视化查询和数量口径见 `docs/neo4j_setup.md`。

## Hybrid Retrieval

Phase 5 已将 Milvus 文本证据与 Neo4j 关系证据接到同一个检索入口 `src/retrieval/hybrid_retriever.py`。这个底层入口仍显式接收精确 `disease_name`；Phase 7 的 Agent 已负责从自然语言中生成和校验这些结构化参数。

检索流程如下：

1. 将问题用同一份本地 BGE 模型编码，并把 `disease_name` 自动加入 Milvus Metadata Filtering；
2. 用精确疾病实体和核心关系白名单查询 Neo4j；
3. 把两路结果归一为统一 evidence schema；
4. 按疾病名 + chunk/fact/content 去重，同时保留全部 `provenance`；
5. 用 Weighted Reciprocal Rank Fusion 合并排名，默认分数为 `weight / (60 + source_rank)`；
6. 返回融合证据、来源、原始向量分数、关系字段、去重统计和分支错误。

```powershell
python -m src.retrieval.hybrid_retriever "百日咳有哪些典型症状？" `
  --disease-name 百日咳 `
  --relation has_symptom `
  --top-k 6 `
  --vector-top-k 4 `
  --graph-top-k 4
```

真实验收使用 `medical_chunks_bge_base_zh_v15_phase5`：Milvus 返回 4 条 BGE 文本证据，Neo4j 返回 4 条症状关系，8 条候选无错误地融合为 6 条最终 evidence。前六名按 Vector / Graph 交错，且所有结果的疾病均为“百日咳”。详细算法、字段和验收输出见 `docs/hybrid_retrieval.md`。

## BGE Reranker and Context Budget

Phase 6 已把融合 evidence 接入本地 `BAAI/bge-reranker-base` Cross-Encoder。Embedding 负责从清洗后的 20,801 个 chunks 中快速召回候选；Reranker 再把“问题 + 每条候选证据”成对阅读，给候选重新排序。原 `rank` 不会被覆盖，输出同时保留 `retrieval_rank`、`rerank_rank`、`rerank_score` 和 `reranker_model`，因此可以复盘某条证据为什么升降名次。

重排后由 `src/generation/context_builder.py` 执行 Context Budget：默认至少保留 1 条 Vector 和 1 条 Graph 证据，再按重排顺序补充；任何证据只有能完整放入预算时才会进入上下文，不做无提示截断。输出包含 `[E1]` 等引用标记、选中/排除证据、token 估算、来源类型数量和未满足配额。当前估算器不绑定某个 LLM；如需与特定模型完全一致的 token 计数，可注入该模型的 tokenizer。

```powershell
python -m src.retrieval.hybrid_retriever "百日咳有哪些典型症状？" `
  --disease-name 百日咳 `
  --relation has_symptom `
  --top-k 10 `
  --vector-top-k 6 `
  --graph-top-k 6 `
  --reranker-model-path "C:\path\to\bge-reranker-base" `
  --reranker-device cuda `
  --rerank-top-k 8 `
  --context-max-tokens 1200 `
  --context-max-items 6
```

真实双库验收得到 5 条 Vector 和 6 条 Graph 候选，融合返回 10 条 evidence；BGE 重排保留 8 条，Context Budget 最终选择 3 条（2 Vector + 1 Graph），使用 1177/1200 个估算 token，并生成 E1-E3 引用。原检索第 5 名的直接症状综述被提升到重排第 1 名（0.99681491），图谱症状“吸气时有蝉鸣音”作为 E3 保留。详细设计和字段见 `docs/reranker_context.md`。

## Function Calling Agent

Phase 7 已新增 `src/agent`。用户现在只需输入自然语言问题；本地确定性规划器会从 `documents.jsonl` 构建疾病名词典，提取精确疾病实体，识别查询意图，再生成一个经过严格校验的工具调用。当前支持：

- 症状、检查、治疗、科室、并发症 → `hybrid_search`，同时使用 Milvus 文本证据和对应的 Neo4j 核心关系；
- 病因、预防、传播、疾病介绍 → `vector_search`，因为这些内容目前没有进入五类核心图谱关系；
- 未识别到疾病或同时出现多个独立疾病 → `needs_clarification`，不会盲猜或访问数据库；
- 工具执行前校验工具名、全部参数、Top-K 1–50、metadata 字段、关系白名单，以及 Hybrid 两路的疾病名一致性。

```powershell
# 只看系统如何理解问题，不加载模型、不访问数据库
python -m src.agent.orchestrator "百日咳有哪些典型症状？" --plan-only

# 自动规划并执行完整 Phase 7 主链路
python -m src.agent.orchestrator "百日咳有哪些典型症状？" `
  --context-max-tokens 1200 `
  --context-max-items 6

# 查看可交给外部 LLM 的严格 Function Calling JSON Schema
python -m src.agent.orchestrator --show-tool-schemas
```

真实验收中，症状问题自动提取“百日咳”、识别 `symptom`、选择 `hybrid_search` 和 `has_symptom`；随后得到 5 条 Vector + 6 条 Graph 候选、10 条融合证据、8 条重排证据和 3 条 E1-E3 上下文证据。病因问题自动选择 `vector_search`，Graph 候选为 0。当前规划器是无需 API Key 的本地规则实现，而不是声称已经由外部 LLM 决策；下一阶段可把同一套 strict schemas 交给 LLM，同时复用当前校验器与执行器。详细说明见 `docs/function_calling.md`。

## FastAPI, Async Retrieval and SSE

Phase 8 已把 `evidence_ready` 接成可调用的 `/chat` 服务。启动时只加载一次本地 BGE、BGE-Reranker、Milvus 与 Neo4j 连接；每次请求先生成并校验查询计划。Hybrid 路由使用 `asyncio` 把相互独立的 Milvus 和 Neo4j 阻塞调用放到工作线程并发执行，完成后再做确定性的融合、重排和 Context Budget。

默认答案后端是`extractive-citation-v1`：它从已选证据中抽取内容并强制携带`[E1]`等引用，不需要API Key。项目同时实现了`GroundedLLMAnswerGenerator`和通用兼容Chat客户端；配置`ANSWER_GENERATOR_BACKEND=openai-compatible`后，真实模型只能读取Context Budget证据，返回的引用编号还会在应用层校验。模型不可用或引用无效时可显式回退到抽取式后端，实际路径通过`answer_generation.mode`暴露。

```powershell
# 启动服务；配置从项目根目录 .env 读取
python -m src.api.app

# 面向演示的简单网页
# http://127.0.0.1:8000/

# 面向开发和调试的 Swagger 接口页
# http://127.0.0.1:8000/docs
```

首页前端与 API 由同一个 FastAPI 进程提供，不需要单独安装 Node.js。页面可输入自然语言问题、指定疾病名，并选择科室或类别做 Metadata Filtering；结果区会同时展示回答、路由方式、引用和证据清单。

Windows 也可以直接在项目根目录运行 `.\start_frontend.ps1`，脚本会使用现有的 `medical-graphrag` Conda 环境启动同一个服务。

非流式请求设置 `"stream": false`，一次返回完整 JSON；流式请求设置 `"stream": true`，按顺序返回 `plan`、`tool_start`、`tool_result`、`citation`、`token`、`done`，异常时返回 `error`。详细请求、字段和测试方法见 `docs/api_sse.md`。

暖机后20次本地SSE实测全部成功：TTFB P50/P95为1.3/11.2ms，首回答Token P50/P95为55.4/64.8ms，完整响应P50/P95为55.9/65.3ms。该结果使用当前抽取式答案后端；切换外部生成模型后必须重新运行`python -m src.evaluation.benchmark_sse`。

## Evaluation

Phase 9固定评估集已经扩展到250题，覆盖50种疾病×症状、检查、治疗、科室、并发症5类核心意图，共587个参考实体。数据集由清洗后的核心关系确定性生成并经过schema回读校验；它可复现，但尚未经过独立临床专家标注。每道题使用同一个问题和同一个BGE-Reranker，分别执行`vector_search`与`hybrid_search`。

250题真实Milvus+Neo4j对照全部完成：Vector-only原始/重排后Top-3命中率为96.8%/97.6%，Hybrid为100.0%/99.6%。两种模式的metadata正确率、引用有效率和陈述引用覆盖率均为100%，脏词输出题目占比均为0。Hybrid重排后图关系命中率为95.2%，回答参考实体覆盖率为82.5%，高于Vector-only的75.6%；固定参考子集Recall@3为86.9%，低于Vector-only的97.5%，说明图谱会补入正确但不在参考子集中的关系实体。

Top-3命中的判定是：前三条证据至少包含一个固定参考实体，或包含“正确疾病+正确核心关系”的图谱事实。`unsupported_claim_rate`只做严格字符串回查，对LLM改写较敏感，不冒充语义幻觉率。500条回答已由本地`qwen2.5-7b-instruct`真实Judge完成（0失败）：Vector-only/Hybrid平均Faithfulness为66.8%/88.4%，语义幻觉题目占比为49.6%/29.6%，即绝对下降20.0个百分点、相对下降40.3%。口径是“存在Unsupported Claim或Faithfulness低于0.8”；数据集未由临床专家独立标注，且当前生成与Judge使用同一模型，结果是可复现的模型评审而非临床安全认证。统一结果与性能口径见`docs/evaluation_report.md`。

```powershell
# 需要 Milvus、Neo4j 以及 .env 中的本地 BGE / Reranker 配置
python -m src.evaluation.runner

# 本机LM Studio建议并发1；失败时返回非零状态，意外中断可加--resume续跑
python -m src.evaluation.judge_runner --concurrency 1 --fail-on-error
```


## Folder Structure

- `src/retrieval`: 已包含 BGE/Milvus、Metadata Filtering、Neo4j 图谱检索、Hybrid Retrieval、BGE-Reranker 和检索后处理管线。
- `src/agent`: 已包含严格工具 schema、疾病实体/意图识别、查询计划和可审计 orchestrator。
- `src/generation`: 已包含Context Budget、E1-En引用、本地抽取式后端和带引用校验的真实LLM生成适配器。
- `src/llm`: 不绑定厂商的兼容Chat客户端，供答案生成与Judge复用。
- `src/evaluation`: 已包含250题构建与校验、Vector-only/Hybrid对照、SSE性能基准、真实Judge运行器及幻觉率下降计算。
- `src/api`: 已包含请求/响应 schema、运行时资源生命周期、业务服务、FastAPI `/chat` 与 SSE 事件流。
- `frontend`: 无额外框架依赖的演示网页，通过 `/chat` 展示问题规划、检索模式、回答、引用与证据。
- `config`: 版本化医疗字段质量规则；ETL与回答层共用，避免文本库和图谱口径漂移。
- `data/sample`: 学习用的小型样例医学知识数据。
- `docs`: 按入门、部署、检索、评测和简历映射整理的项目文档；入口见 `docs/README.md`。
- `results`: 可复现实验输出和正式评测结果；一次性 smoke/check 中间文件不纳入项目交付。
- `tests`: 测试用例目录。

## Data and Relation Scale

当前清洗后的本地处理数据包含8,807篇documents、20,801个chunks、182,565条核心triples和21,604个核心文本实体。独立候选关系流程另外生成548,982条候选关系；离线核心+候选共731,547条关系记录、259,912个去重文本实体。Neo4j正式主链路仍只使用五类核心关系，候选池不会为了扩大数字而进入生产查询。

数据质量层会在文档与三元组写出前统一过滤占位符、无有效字符、已确认污染项和归一化重复值，并输出 `rejected_medical_items.jsonl` 审计记录。一次真实问答发现的人名症状污染已在 15 条原始记录中定位；清洗后的 documents、chunks、triples 中该值均为 0。答案生成层还会逐项校验结构化字段与 graph entity，并直接使用 Planner 的 intents，作为旧数据库完成同步前的最后防线。详见 `docs/data_quality.md`。

```bash
python -m src.data.stat_relation_scale --format markdown
python -m src.data.build_candidate_triples
python -m src.data.stat_relation_scale --candidate-triples data/processed/candidate_triples.csv --format json
python -m unittest discover -s tests -v
```

详细口径见 `docs/relation_stats.md`。本仓库只包含医疗 GraphRAG 项目实现；Agent Evaluation / Reward Modeling 保持为独立仓库。


## Current Progress

| Phase | Status | Deliverable |
| --- | --- | --- |
| Phase 1: Data ETL | Complete | `medical.json` → 8,807 documents / 182,565 clean core triples / 20-row sample |
| Phase 2: Chunking & Embedding | Complete | 20,801 clean metadata-rich chunks；Hash 测试后端；`bge-base-zh-v1.5` 768维 GPU 编码；BGE 查询指令 |
| Phase 3: Milvus Retrieval | Complete | Milvus 3.0 + PyMilvus 3.0.1；20,801 条干净 BGE 向量真实写入；逻辑/存储计数；真实 Top-K 与 Metadata Filtering 验收 |
| Phase 4: Neo4j Graph Retrieval | Complete | Neo4j 2026.07.1；24,237 typed nodes；182,565 条干净核心关系；幂等导入；受控一跳 graph evidence |
| Phase 5: Hybrid Retrieval | Complete | Milvus + Neo4j 双路召回；统一 evidence；Weighted RRF；跨路去重与 provenance；真实双库查询验收 |
| Phase 6: Reranker & Context Budget | Complete | 本地 `bge-reranker-base` Cross-Encoder；可审计重排；Top-K/阈值；Vector/Graph 配额；E1-En 引用与 token budget；真实双库验收 |
| Phase 7: Agent / Function Calling | Complete | 疾病目录实体识别；多意图映射；strict schemas；参数安全边界；自动 Vector/Hybrid 路由；完整证据链真实验收 |
| Phase 8: Answer API | Complete | 可替换 AnswerGenerator；抽取式带引用答案；FastAPI `/health`/`/chat`；JSON/SSE；async 双路检索；真实双库 HTTP 验收 |
| Phase 8.1: Data Quality Hardening | Complete | 统一字段策略、版本化拒绝配置、243条审计、字段级去重、Planner intents直传、双数据库同步、真实 `/chat` 污染词为0 |
| Phase 9: Evaluation | Complete | 250题/587参考实体；500次双路实库评测；Top-3/MRR/Recall；500/500条真实Judge评分 |
| Phase 10: Metrics | Complete | 548,982条隔离候选关系；SSE TTFB实测；真实LLM生成/Judge；语义幻觉率相对下降40.3% |

## Development Roadmap

- Day8 Project Init
- Day9 Document Loader + Embedding Concept
- Day10 Milvus Vector Retrieval
- Day11 Neo4j Graph Retrieval
- Day12 Hybrid Retrieval + Reranker
- Day13 Function Calling
- Day14 FastAPI + SSE
- Day15 Evaluation + LLM-as-a-Judge
- Day16 Context Budget
- Day17 Test Set Design
