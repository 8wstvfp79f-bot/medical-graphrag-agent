# 文档导航

这组文档按“先理解、再运行、再验证”的顺序组织。第一次阅读不必从头看完所有文件。

## 1. 先理解项目

- `../README.md`：项目总览、完成状态、运行入口和简历能力映射。
- `architecture.md`：从原始医疗数据到回答结果的整体信息流。
- `relation_stats.md`：数据产物、核心关系、候选关系、质量审计和 Neo4j 数量口径。

## 2. 启动本地基础服务

- `milvus_setup.md`：启动 Milvus、写入 BGE 向量、检索和 Attu 可视化。
- `neo4j_setup.md`：导入核心医学关系、验证数量和 Neo4j Browser 可视化。
- `llm_integration.md`：接入 LM Studio/OpenAI-compatible 本地生成模型。

## 3. 理解问答主链路

- `hybrid_retrieval.md`：Milvus 文本证据与 Neo4j 图谱证据如何融合。
- `reranker_context.md`：BGE-Reranker 如何重排，以及 Context Budget 如何选证据。
- `function_calling.md`：问题意图、参数 schema、工具选择和参数校验。
- `api_sse.md`：FastAPI `/chat`、JSON 返回和 SSE 流式事件。
- `data_quality.md`：字段清洗、污染过滤、去重与回答前安全检查。

## 4. 查看实验与指标

- `evaluation_report.md`：250 题 Vector-only/Hybrid 对照、500 条 LLM Judge 与 SSE 性能结果。

## 推荐阅读路线

建议依次阅读：`../README.md` → `architecture.md` → `relation_stats.md` → `milvus_setup.md` → `neo4j_setup.md` → `hybrid_retrieval.md` → `function_calling.md` → `api_sse.md` → `evaluation_report.md`。
