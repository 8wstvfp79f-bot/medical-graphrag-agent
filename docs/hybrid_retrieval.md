# 混合检索与第 5 阶段运行手册

## 这一步解决什么问题

第 3 阶段的 Milvus 能找到与问题语义相近的文本，第 4 阶段的 Neo4j 能返回疾病与症状、检查、治疗、科室、并发症之间的明确关系。第 5 阶段的职责是把两路候选变成统一证据，并为后续重排器和大模型提供可审计的候选列表。

本模块自身只负责检索，不直接生成医疗回答，也不识别疾病名。完整问答链路由上游 Function Calling 自动抽取 `query`、`disease_name` 和关系参数，再调用本模块。

## 数据流

```text
问题 + 疾病名称
├── BGE 问题向量 → Milvus + 疾病元数据过滤 → 向量候选
└── 精确疾病节点 → Neo4j + 核心关系白名单 → 图谱候选
                         ↓
                      统一证据结构
                         ↓
                    去重 + 来源合并
                         ↓
                    加权 RRF 排名
                         ↓
              证据列表 + 统计 + 策略 + 错误
```

疾病名会自动写入 Milvus metadata filter。如果调用方另传了不同的 `metadata_filter.disease_name`，检索会在访问数据库前拒绝执行，避免向量证据与图谱证据属于不同疾病。

## 加权 RRF

向量分支有 COSINE score，图谱分支只有结构化顺序，两者的原始分数不能直接相加。因此项目使用 Weighted Reciprocal Rank Fusion：

```text
融合贡献 = 来源权重 /（rrf_k + 来源名次）
```

默认 `rrf_k=60`，Vector 与 Graph 权重均为 `1.0`。这会让两路第一名获得相同贡献，并在没有重复项时形成 Vector / Graph 交错候选。`fusion_score` 只用于候选融合，不代表概率或医学可信度。

## 统一证据结构

每条 evidence 至少包含：

| 字段 | 含义 |
| --- | --- |
| `evidence_id` | 根据规范化内容或图谱事实生成的稳定 ID |
| `evidence_type` | `vector`（向量）、`graph`（图谱）或跨路去重后的 `hybrid`（混合） |
| `content` | 可交给后续重排器的证据文本 |
| `disease_name` | 精确疾病名 |
| `source` / `sources` | 主来源与全部合并来源 |
| `rank` | 融合后的最终顺序 |
| `best_source_rank` | 该证据在原分支中的最好名次 |
| `retrieval_score` | Milvus 原始相似度；纯图谱证据为 `null` |
| `fusion_score` | Weighted RRF 融合分数 |
| `metadata` | 疾病、类别、科室、来源或关系实体字段 |
| `relation` / `entity_name` | 图谱关系信息；纯向量证据为 `null` |
| `provenance` | 每个原始候选的分支、记录 ID、名次、分数和融合贡献 |

## 去重规则

- 向量分支：同一个 `chunk_id` 只保留一个候选；
- 图谱分支：同一个 `disease_name + relation + entity_name` 只保留一个事实；
- 跨来源：同一疾病下规范化后完全相同的文本会合并；
- 合并后累加各分支的 RRF 融合贡献，并保留全部 `sources` 和 `provenance`；
- 不做模糊文本去重，避免把含义相近但医学细节不同的证据误合并。

## 运行命令

`.env` 需要配置 Milvus、BGE 本地模型和 Neo4j 连接。随后运行：

```powershell
python -m src.retrieval.hybrid_retriever "百日咳有哪些典型症状？" `
  --disease-name 百日咳 `
  --relation has_symptom `
  --top-k 6 `
  --vector-top-k 4 `
  --graph-top-k 4
```

默认任一数据库失败都会让请求失败，防止系统静默返回不完整证据。只有显式传入 `--allow-partial` 时才允许单路降级，并把错误写入返回值的 `errors.vector` 或 `errors.graph`。

## 本地真实验收结果

本机真实验收使用：

- Milvus 3.0.0 / PyMilvus 3.0.1；
- `medical_chunks_bge_base_zh_v15_phase5`；
- 20,801 个清洗后的 `bge-base-zh-v1.5` 768 维文本块；
- Neo4j 2026.07.1；
- 24,237 个 typed nodes 与 182,565 条核心关系。

问题“百日咳有哪些典型症状？”并限制 `has_symptom` 后：

| 指标 | 结果 |
| --- | ---: |
| 向量候选 | 4 |
| 图谱候选 | 4 |
| 去重前候选 | 8 |
| 合并重复项 | 0 |
| 去重后候选 | 8 |
| 最终证据 | 6 |
| 分支错误 | 0 |

结果中同时出现 BGE 文本证据，以及“百日咳 -[has_symptom]-> 低热 / 吸气时有蝉鸣音 / 惊厥”等图谱证据；所有向量元数据的 `disease_name` 均精确匹配“百日咳”。

## Milvus 3.0 本地存储注意事项

本机的两个早期集合在 Docker 重启后触发了 Milvus 本地 Storage V2 的路径缺陷：部分 packed stats 被写入双重根目录，而加载器从 `files/json_stats/...` 查找。原数据文件仍存在，但集合无法重新 load。

项目没有删除或原地改写旧集合，而是将它们 release，另建 `medical_chunks_bge_base_zh_v15_phase5` 完成当前验收。这个问题属于 Milvus 运行时，不属于本项目的向量、metadata 或 fusion 逻辑。生产部署前应改用包含上游修复的 Milvus 版本或受支持的对象存储方案，并执行一次容器重启后的 load/search 验收。

相关上游记录：

- <https://github.com/milvus-io/milvus/issues/45959>
- <https://github.com/milvus-io/milvus/issues/49419>

## 验收标准

- 向量分支与图谱分支都通过真实数据库返回候选；
- 两路疾病上下文一致，Metadata Filtering 仍然生效；
- 统一证据结构能同时表达文本分块与图谱事实；
- 重复候选不占用多个 Top-K 位置，并保留 provenance；
- RRF 参数、候选数量、去重数量和错误均可审计；
- 单路降级必须显式开启；
- 单元测试覆盖融合顺序、疾病冲突、去重、降级和非法参数。
