# 医疗数据质量

## 为什么需要这一层

一次真实 `/chat` 验收发现原始 `symptom` 字段把人名“闫鹏辉”混入了百日咳症状。继续追查后确认，该值在 15 种互不相关的疾病记录中重复出现，属于原始公开数据的系统性污染，而不是 BGE、Reranker 或答案模型生成的幻觉。

## 统一校验边界

`src/data/medical_quality.py` 是统一字段质量边界。`build_documents.py` 与 `build_triples.py` 在写出任何 RAG 文档或图谱关系前都会调用同一份版本化配置 `config/medical_data_quality.json`，执行：

- 空白、占位符和无有效字符值过滤；
- 经人工确认的非医学实体精确拒绝；
- 可配置别名归一化；
- 归一化后的字段内去重；
- 被拒绝条目写入 `data/processed/rejected_medical_items.jsonl`，保留疾病、字段、原值、原因和来源。

答案生成层还会对结构化字段和 graph entity 再应用相同规则，用于保护尚未完成数据同步的旧数据库或外部证据。这个检查是最后防线，不能替代 ETL 入口清洗。

## 当前干净数据快照

| 指标 | 数量 |
| --- | ---: |
| 文档 | 8,807 |
| 分块 | 20,801 |
| 核心三元组 | 182,565 |
| 核心实体 | 21,604 |
| 已审计拒绝项 | 243 |
| 已确认的非医学症状记录 | 15 |
| “闫鹏辉”在文档、分块和三元组中的出现次数 | 0 |

243 条审计记录由 15 条已确认非医学实体、146 条归一化重复项和 82 条无有效字符项构成。后两类在旧图谱生成逻辑中通常也会被去重或跳过，因此本次核心 triples 净减少 15 条 `has_symptom` 关系。

## 数据库同步状态

干净的 `documents.jsonl`、`chunks.jsonl` 与 `triples.csv` 已重建并完成离线全量校验。真实 Milvus 正式 BGE Collection 已重建为 20,801 条向量；Neo4j 已精准删除 15 条污染关系与 1 个孤立症状节点，并重新导入为 24,237 个 typed nodes、182,565 条关系。

同步过程只重建了正式 BGE Collection，Hash 测试 Collection 保留。真实 `/chat` 回归返回 3 条引用、4 条上下文证据，完整响应中“闫鹏辉”出现次数为 0。

后续再次更新拒绝配置时，按以下顺序同步 Neo4j：

```powershell
python -m src.retrieval.neo4j_client prune-rejected
python -m src.retrieval.neo4j_client ingest --batch-size 2000
python -m src.retrieval.neo4j_client count
```

`prune-rejected` 默认只处理原因以 `confirmed_` 开头的审计项，不会因为“字段内重复”而删除仍然有效的医学关系。Milvus 应对 `.env` 中指定的正式 BGE Collection 执行显式 `--recreate` 后全量写入；不要对 Hash 测试 Collection 使用该参数。
