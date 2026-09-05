# 第 9 阶段：检索与回答评测报告

- 生成时间（UTC）：2026-09-04T14:08:01.468025+00:00
- 固定测试集：`data/evaluation/medical_qa_eval.jsonl`（250 题）
- Milvus 集合：`medical_chunks_bge_base_zh_v15_phase5`
- 测试集来源：从本项目清洗后的核心关系三元组抽取，尚未经过独立临床专家标注。
- 大模型裁判：500/500 条真实本地模型评分已完成；纯向量/混合检索语义幻觉题目占比为 49.6%/29.6%，相对下降 40.3%。

## 核心结果

| 指标 | 纯向量检索 | 混合检索（Milvus + Neo4j） |
|---|---:|---:|
| 原始 Top-3 命中率 | 96.8% | 100.0% |
| 重排后 Top-3 命中率 | 97.6% | 99.6% |
| 重排后固定参考实体召回率@3 | 97.5% | 86.9% |
| 元数据正确率@3 | 100.0% | 100.0% |
| 图关系命中率@3 | 0.0% | 95.2% |
| 引用有效率 | 100.0% | 100.0% |
| 回答参考实体覆盖率 | 75.6% | 82.5% |
| 无证据陈述率（规则代理） | 89.6% | 11.6% |
| 脏词输出题目占比 | 0.0% | 0.0% |

## 平均链路耗时

| 指标 | 纯向量检索 | 混合检索（Milvus + Neo4j） |
|---|---:|---:|
| 工具与检索后处理 | 15.1 ms | 16.5 ms |
| 检索到回答生成完成 | 2143.4 ms | 2661.2 ms |

Top-3 命中定义：前三条证据至少包含一个固定参考实体，或包含“正确疾病 + 正确核心关系”的 Neo4j 事实。固定参考实体召回率@3 只计算参考实体子集，因此图谱返回同关系下其他有效实体时可能命中，但召回率不增加。

结果解读：扩展测试集上，混合检索提高了 Top-3 命中、图关系覆盖和回答参考实体覆盖；固定参考实体召回率@3 低于纯向量检索，说明图谱也会补入同一关系下正确但不在固定参考子集中的实体。后续仍需独立专家标注和融合权重实验。

`无证据陈述率`只检查编号陈述是否带有效引用，以及陈述能否直接回查到引用证据；它是可复现的规则代理，不等于真正的语义幻觉率。

## 大模型裁判

- 裁判模型：`qwen2.5-7b-instruct`
- 提示词版本：`medical-evidence-judge-v1`
- 完成/失败：500/0
- 忠实度阈值：0.8
- 判定口径：存在无证据支持的陈述，或忠实度低于阈值，即记为语义幻觉题目。

| 指标 | 纯向量检索 | 混合检索 |
| --- | ---: | ---: |
| 平均忠实度 | 66.8% | 88.4% |
| 平均相关性 | 82.4% | 97.2% |
| 平均完整性 | 65.4% | 85.2% |
| 语义幻觉题目占比 | 49.6% | 29.6% |

混合检索的语义幻觉题目占比绝对下降 20.0 个百分点、相对下降 40.3%。该结果是模型评审，不是临床安全认证；固定集尚未经过独立临床专家标注，且生成与裁判使用同一模型时可能存在自评偏差。

## SSE 性能基准

- 测试接口：`http://127.0.0.1:8000/chat`
- 正式请求/预热：20/3
- 成功/失败：20/0

| 指标 | P50 | P95 | 定义 |
| --- | ---: | ---: | --- |
| TTFB | 1.3 ms | 11.2 ms | 收到首个非空 SSE 响应行 |
| 首 SSE 事件 | 1.3 ms | 11.2 ms | 通常是 plan，不代表答案已经生成 |
| 首回答 Token | 55.4 ms | 64.8 ms | 检索、重排和生成后首段答案 |
| 完整响应 | 55.9 ms | 65.3 ms | 收到完整 SSE 响应 |

TTFB、首 SSE 事件和首回答 Token 是三个不同时间点，不能混为同一个指标。以上结果使用抽取式回答后端；切换真实生成模型后应重新运行性能基准。

## 每题 Top-3 结果

| 用例 | 意图代码 | 纯向量 原始/重排 | 混合 原始/重排 | 混合回答覆盖率 |
|---|---|---:|---:|---:|
| case_001_symptom | symptom | 1/1 | 1/1 | 0.0% |
| case_001_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_001_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_001_department | department | 1/1 | 1/1 | 100.0% |
| case_001_complication | complication | 1/1 | 1/1 | 100.0% |
| case_002_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_002_diagnosis | diagnosis | 0/1 | 1/1 | 66.7% |
| case_002_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_002_department | department | 1/1 | 1/1 | 100.0% |
| case_002_complication | complication | 0/0 | 1/1 | 33.3% |
| case_003_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_003_diagnosis | diagnosis | 0/1 | 1/1 | 0.0% |
| case_003_treatment | treatment | 0/1 | 1/1 | 33.3% |
| case_003_department | department | 1/1 | 1/1 | 100.0% |
| case_003_complication | complication | 1/1 | 1/1 | 100.0% |
| case_004_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_004_diagnosis | diagnosis | 1/1 | 1/1 | 0.0% |
| case_004_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_004_department | department | 1/1 | 1/1 | 100.0% |
| case_004_complication | complication | 1/1 | 1/1 | 33.3% |
| case_005_symptom | symptom | 1/0 | 1/1 | 33.3% |
| case_005_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_005_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_005_department | department | 1/1 | 1/1 | 100.0% |
| case_005_complication | complication | 0/0 | 1/0 | 100.0% |
| case_006_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_006_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_006_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_006_department | department | 1/1 | 1/1 | 100.0% |
| case_006_complication | complication | 1/1 | 1/1 | 100.0% |
| case_007_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_007_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_007_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_007_department | department | 1/1 | 1/1 | 100.0% |
| case_007_complication | complication | 1/1 | 1/1 | 100.0% |
| case_008_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_008_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_008_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_008_department | department | 1/1 | 1/1 | 100.0% |
| case_008_complication | complication | 1/1 | 1/1 | 100.0% |
| case_009_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_009_diagnosis | diagnosis | 0/1 | 1/1 | 66.7% |
| case_009_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_009_department | department | 1/1 | 1/1 | 100.0% |
| case_009_complication | complication | 0/1 | 1/1 | 100.0% |
| case_010_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_010_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_010_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_010_department | department | 1/1 | 1/1 | 100.0% |
| case_010_complication | complication | 1/1 | 1/1 | 100.0% |
| case_011_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_011_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_011_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_011_department | department | 1/1 | 1/1 | 100.0% |
| case_011_complication | complication | 1/1 | 1/1 | 100.0% |
| case_012_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_012_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_012_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_012_department | department | 1/1 | 1/1 | 100.0% |
| case_012_complication | complication | 1/1 | 1/1 | 33.3% |
| case_013_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_013_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_013_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_013_department | department | 1/1 | 1/1 | 100.0% |
| case_013_complication | complication | 1/1 | 1/1 | 100.0% |
| case_014_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_014_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_014_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_014_department | department | 1/1 | 1/1 | 100.0% |
| case_014_complication | complication | 1/1 | 1/1 | 100.0% |
| case_015_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_015_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_015_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_015_department | department | 1/1 | 1/1 | 100.0% |
| case_015_complication | complication | 1/1 | 1/1 | 100.0% |
| case_016_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_016_diagnosis | diagnosis | 1/0 | 1/1 | 66.7% |
| case_016_treatment | treatment | 1/0 | 1/1 | 66.7% |
| case_016_department | department | 1/1 | 1/1 | 100.0% |
| case_016_complication | complication | 0/0 | 1/1 | 100.0% |
| case_017_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_017_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_017_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_017_department | department | 1/1 | 1/1 | 100.0% |
| case_017_complication | complication | 1/1 | 1/1 | 100.0% |
| case_018_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_018_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_018_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_018_department | department | 1/1 | 1/1 | 100.0% |
| case_018_complication | complication | 1/1 | 1/1 | 100.0% |
| case_019_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_019_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_019_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_019_department | department | 1/1 | 1/1 | 100.0% |
| case_019_complication | complication | 1/1 | 1/1 | 100.0% |
| case_020_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_020_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_020_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_020_department | department | 1/1 | 1/1 | 100.0% |
| case_020_complication | complication | 1/1 | 1/1 | 100.0% |
| case_021_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_021_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_021_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_021_department | department | 1/1 | 1/1 | 100.0% |
| case_021_complication | complication | 1/1 | 1/1 | 100.0% |
| case_022_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_022_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_022_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_022_department | department | 1/1 | 1/1 | 100.0% |
| case_022_complication | complication | 1/1 | 1/1 | 100.0% |
| case_023_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_023_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_023_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_023_department | department | 1/1 | 1/1 | 100.0% |
| case_023_complication | complication | 1/1 | 1/1 | 100.0% |
| case_024_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_024_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_024_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_024_department | department | 1/1 | 1/1 | 100.0% |
| case_024_complication | complication | 1/1 | 1/1 | 100.0% |
| case_025_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_025_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_025_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_025_department | department | 1/1 | 1/1 | 100.0% |
| case_025_complication | complication | 1/1 | 1/1 | 100.0% |
| case_026_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_026_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_026_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_026_department | department | 1/1 | 1/1 | 100.0% |
| case_026_complication | complication | 1/1 | 1/1 | 100.0% |
| case_027_symptom | symptom | 1/1 | 1/1 | 0.0% |
| case_027_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_027_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_027_department | department | 1/1 | 1/1 | 100.0% |
| case_027_complication | complication | 1/1 | 1/1 | 100.0% |
| case_028_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_028_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_028_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_028_department | department | 1/1 | 1/1 | 100.0% |
| case_028_complication | complication | 1/1 | 1/1 | 100.0% |
| case_029_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_029_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_029_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_029_department | department | 1/1 | 1/1 | 100.0% |
| case_029_complication | complication | 1/1 | 1/1 | 100.0% |
| case_030_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_030_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_030_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_030_department | department | 1/1 | 1/1 | 100.0% |
| case_030_complication | complication | 1/1 | 1/1 | 100.0% |
| case_031_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_031_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_031_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_031_department | department | 1/1 | 1/1 | 100.0% |
| case_031_complication | complication | 1/1 | 1/1 | 100.0% |
| case_032_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_032_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_032_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_032_department | department | 1/1 | 1/1 | 100.0% |
| case_032_complication | complication | 1/1 | 1/1 | 100.0% |
| case_033_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_033_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_033_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_033_department | department | 1/1 | 1/1 | 100.0% |
| case_033_complication | complication | 1/1 | 1/1 | 100.0% |
| case_034_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_034_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_034_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_034_department | department | 1/1 | 1/1 | 100.0% |
| case_034_complication | complication | 1/1 | 1/1 | 33.3% |
| case_035_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_035_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_035_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_035_department | department | 1/1 | 1/1 | 100.0% |
| case_035_complication | complication | 1/1 | 1/1 | 100.0% |
| case_036_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_036_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_036_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_036_department | department | 1/1 | 1/1 | 100.0% |
| case_036_complication | complication | 1/1 | 1/1 | 100.0% |
| case_037_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_037_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_037_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_037_department | department | 1/1 | 1/1 | 100.0% |
| case_037_complication | complication | 1/1 | 1/1 | 100.0% |
| case_038_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_038_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_038_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_038_department | department | 1/1 | 1/1 | 100.0% |
| case_038_complication | complication | 1/1 | 1/1 | 100.0% |
| case_039_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_039_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_039_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_039_department | department | 1/1 | 1/1 | 100.0% |
| case_039_complication | complication | 1/1 | 1/1 | 100.0% |
| case_040_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_040_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_040_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_040_department | department | 1/1 | 1/1 | 100.0% |
| case_040_complication | complication | 1/1 | 1/1 | 100.0% |
| case_041_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_041_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_041_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_041_department | department | 1/1 | 1/1 | 100.0% |
| case_041_complication | complication | 1/1 | 1/1 | 100.0% |
| case_042_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_042_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_042_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_042_department | department | 1/1 | 1/1 | 100.0% |
| case_042_complication | complication | 1/1 | 1/1 | 100.0% |
| case_043_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_043_diagnosis | diagnosis | 1/1 | 1/1 | 0.0% |
| case_043_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_043_department | department | 1/1 | 1/1 | 100.0% |
| case_043_complication | complication | 1/1 | 1/1 | 100.0% |
| case_044_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_044_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_044_treatment | treatment | 1/1 | 1/1 | 0.0% |
| case_044_department | department | 1/1 | 1/1 | 100.0% |
| case_044_complication | complication | 1/1 | 1/1 | 100.0% |
| case_045_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_045_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_045_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_045_department | department | 1/1 | 1/1 | 50.0% |
| case_045_complication | complication | 1/1 | 1/1 | 100.0% |
| case_046_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_046_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_046_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_046_department | department | 1/1 | 1/1 | 100.0% |
| case_046_complication | complication | 1/1 | 1/1 | 100.0% |
| case_047_symptom | symptom | 1/1 | 1/1 | 33.3% |
| case_047_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_047_treatment | treatment | 1/1 | 1/1 | 33.3% |
| case_047_department | department | 1/1 | 1/1 | 100.0% |
| case_047_complication | complication | 1/1 | 1/1 | 100.0% |
| case_048_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_048_diagnosis | diagnosis | 1/1 | 1/1 | 33.3% |
| case_048_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_048_department | department | 1/1 | 1/1 | 100.0% |
| case_048_complication | complication | 1/1 | 1/1 | 100.0% |
| case_049_symptom | symptom | 1/1 | 1/1 | 100.0% |
| case_049_diagnosis | diagnosis | 1/1 | 1/1 | 100.0% |
| case_049_treatment | treatment | 1/1 | 1/1 | 66.7% |
| case_049_department | department | 1/1 | 1/1 | 50.0% |
| case_049_complication | complication | 1/1 | 1/1 | 100.0% |
| case_050_symptom | symptom | 1/1 | 1/1 | 66.7% |
| case_050_diagnosis | diagnosis | 1/1 | 1/1 | 66.7% |
| case_050_treatment | treatment | 1/1 | 1/1 | 100.0% |
| case_050_department | department | 1/1 | 1/1 | 100.0% |
| case_050_complication | complication | 1/1 | 1/1 | 100.0% |

## 如何复现

确保 Milvus、Neo4j 已启动，并在项目环境中执行：

```powershell
python -m src.evaluation.runner
```

完整机器可读结果位于 `results/evaluation/phase9_metrics.json`；裁判模型输入、500 条评分结果和汇总分别位于同目录的 `phase9_metrics.judge_requests.jsonl`、`phase9_metrics.judge_results.jsonl` 与 `phase9_judge_summary.json`。SSE 原始性能结果位于 `results/performance/sse_ttfb.json`。
