# 关系规模统计

## 数据快照

以下数字由当前本地核心CSV和独立候选CSV计算得到。统计脚本只读取处理结果，不会修改原始数据或Neo4j。

| 指标 | 数量 |
| --- | ---: |
| 文档 | 8,807 |
| 核心三元组 | 182,565 |
| 候选三元组（独立候选池） | 548,982 |
| 核心与候选三元组总数 | 731,547 |
| 核心实体 | 21,604 |
| 核心疾病实体 | 8,806 |
| 全部去重文本实体 | 259,912 |

`documents` 比核心疾病实体多 1，是因为 `doc_id` 按记录生成，而核心疾病实体按去重后的疾病名称计算。

## 数据产物

原始输入为 `data/raw/xywy/medical.json`，共 8,808 条记录；清洗后保留 8,807 条有效文档。原始数据只用于本地 ETL，不进入 Git。

| 产物 | 规模 | 主要字段 |
| --- | ---: | --- |
| `data/processed/documents.jsonl` | 8,807 | `doc_id`, `disease_name`, `source`, `text`, `metadata` |
| `data/processed/chunks.jsonl` | 20,801 | 文本块内容、BGE 检索字段和元数据 |
| `data/processed/triples.csv` | 182,565 | `head`, `relation`, `tail`, `source` |
| `data/processed/candidate_triples.csv` | 548,982 | 与核心图谱隔离的候选关系池 |

文档与图谱构建共用 `src/data/medical_quality.py` 和版本化配置 `config/medical_data_quality.json`。清洗审计共拒绝 243 个核心字段值；已确认的污染症状实体在最终文档、文本块和三元组中均为 0。详细异常数据案例与数据库同步过程见 `data_quality.md`。

## 核心关系分布

| 关系 | 数量 | Neo4j 中的用途 |
| --- | ---: | --- |
| `has_symptom` | 54,693 | 疾病到症状 |
| `diagnosed_by` | 39,413 | 疾病到检查项 |
| `treated_by` | 59,654 | 疾病到常用/推荐药品 |
| `belongs_to` | 16,781 | 疾病到科室 |
| `has_complication` | 12,024 | 疾病到并发症 |
| **总计** | **182,565** | 核心医学关系主链路 |

## 核心关系与候选关系

核心关系是当前 Neo4j 主链路直接使用的五类医学关系：症状、检查、药品、科室和并发症。它们结构稳定、可解释性强，适合从疾病实体出发做一跳或受控多跳扩展，并可在回答中作为证据类型展示。

候选关系是可从原始字段继续派生、但默认不进入主图的关系，例如：

- `caused_by`：病因文本通常较长，实体边界不稳定；
- `prevented_by`：预防建议常是句子级文本；
- `treated_with_method`：治疗方式粒度不统一；
- `recommended_food` / `avoid_food`：重复项多，医疗问答收益有限；
- `has_duration` / `has_cost_range` / `transmitted_by`：值域和表达格式需要额外标准化。

候选关系已经由独立脚本生成并完成基础清洗，共548,982条。它们仍需经过频次阈值、抽样审核和离线检索消融，确认能提高Recall@K或回答完整性后才允许进入实验图谱。`triples.csv` 继续只包含Core relations。

### 候选关系分布

| 关系 | 数量 | 来源字段 |
| --- | ---: | --- |
| `caused_by` | 177,551 | 病因句段 |
| `has_brand_drug` | 156,959 | 品牌/药品明细 |
| `recommended_food` | 62,308 | 宜吃/推荐饮食 |
| `prevented_by` | 56,409 | 预防句段 |
| `avoid_food` | 22,192 | 忌吃饮食 |
| `treated_with_method` | 21,047 | 治疗方式 |
| `affects_population` | 8,802 | 易感人群 |
| `transmitted_by` | 8,803 | 传播方式 |
| `has_cost_range` | 8,797 | 费用区间 |
| `has_duration` | 8,796 | 治疗周期 |
| `has_cure_probability` | 8,701 | 治愈概率 |
| `has_prevalence` | 8,617 | 患病概率 |
| **总计** | **548,982** | 独立候选池 |

### 候选关系质量审计

| 拒绝原因 | 数量 |
| --- | ---: |
| 无法构成有效关系尾实体 | 12,455 |
| 字段内归一化重复 | 8,112 |
| 无有效字符 | 119 |
| 纯标点 | 28 |
| 占位值 | 11 |
| **总计** | **20,725** |

另外发现并移除了 22 条跨字段重复边。病因和预防文本会按句号、分号、换行与编号切成不超过 220 字符的候选片段；长文本片段、品牌药名和统计属性在证明能够提升检索效果前不会进入正式图谱。

## Neo4j 导入策略

实际 Neo4j 主链路优先导入核心关系。这样可以控制节点和边的规模、减少长文本伪实体与低价值关系造成的图扩展噪声，并让 `graph_search` 的证据类型保持清晰。候选关系应进入独立实验标签或灰度集合，不应为了扩大关系数量直接混入正式查询路径。

因此当前有两个都可复现、但不能混用的口径：Neo4j正式主链路为 **21,604个核心文本实体、182,565条核心关系边**；离线核心+候选池为 **259,912个去重文本实体、731,547条关系记录**。候选池规模不能直接等同于Neo4j正式检索规模。

## Neo4j 本地真实验收

数据质量加固后已在本地 Neo4j 2026.07.1 完成真实同步与全量导入：

| 节点标签 | 数量 |
| --- | ---: |
| `Disease` | 8,806 |
| `Symptom` | 5,996 |
| `Check` | 3,351 |
| `Drug` | 3,825 |
| `Department` | 54 |
| `Complication` | 2,205 |
| **类型化节点总数** | **24,237** |

关系分布与干净 CSV 完全一致，总数为 **182,565**。同步前通过审计删除了 15 条 `has_symptom` 污染关系和 1 个孤立 `Symptom` 节点；随后全量 `MERGE` 导入数量保持不变。

离线的 21,604 是跨类型去重后的名称文本实体数；Neo4j 的 24,237 是按标签区分后的 typed node 数。同一名称如果既是症状又是并发症，会保留为两个语义类型节点，因此两个口径不同。

## 复现统计

从项目根目录运行：

```bash
python -m src.data.stat_relation_scale --format markdown
python -m src.data.build_candidate_triples
python -m src.data.stat_relation_scale --candidate-triples data/processed/candidate_triples.csv --format markdown
```

也可以输出机器可读结果：

```bash
python -m src.data.stat_relation_scale --format json
```

机器可读候选审计写入 `results/data/candidate_relation_stats.json`。`results/` 和大体积处理文件默认不提交，可由以上命令重新生成。
