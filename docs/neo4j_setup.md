# Neo4j 图谱检索

> 数据质量刷新已完成：当前本地数据库包含 24,237 个类型化节点和 182,565 条关系。同步过程先删除 15 条已确认污染的 `HAS_SYMPTOM` 关系和 1 个孤立 `Symptom` 节点，再幂等导入干净 CSV。

## Neo4j 在项目中的作用

Milvus 保存“文本 chunk + BGE 向量”，适合回答“哪些文本和问题语义相近”。Neo4j 保存“疾病实体 + 明确医学关系”，适合回答“这个疾病有哪些症状、检查、药品、科室和并发症”。下游 Hybrid Retrieval 已将这两类证据统一融合。

```text
triples.csv
    ↓ validate（字段、关系白名单）
Neo4j batch MERGE
    ↓
(Disease)-[CORE_RELATION]->(Typed Entity)
    ↓ graph_search(disease_name, relations, limit)
结构化 graph evidence + source + path_text
```

## 图谱结构

所有节点都用 `name` 作为该标签内的唯一键，并建立 `IF NOT EXISTS` 唯一性约束。节点按类型分开，避免同一个词在不同语义下被强行合成一个节点。

| CSV 关系 | Neo4j 模式 |
| --- | --- |
| `has_symptom` | `(Disease)-[:HAS_SYMPTOM]->(Symptom)` |
| `diagnosed_by` | `(Disease)-[:DIAGNOSED_BY]->(Check)` |
| `treated_by` | `(Disease)-[:TREATED_BY]->(Drug)` |
| `belongs_to` | `(Disease)-[:BELONGS_TO]->(Department)` |
| `has_complication` | `(Disease)-[:HAS_COMPLICATION]->(Complication)` |

候选关系不在白名单中，即使以后出现在 CSV 中，也只会计入 `skipped_candidate_records`，不会进入当前主图。

## 启动本地服务

下面是本地开发示例。密码只放在本机环境变量中，不提交到仓库。

```powershell
docker volume create medical-graphrag-neo4j-data

docker run -d `
  --name neo4j-medical `
  --restart unless-stopped `
  -p 7474:7474 `
  -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/replace-with-a-local-password `
  -v medical-graphrag-neo4j-data:/data `
  neo4j:2026.07.1
```

- `http://127.0.0.1:7474`：Neo4j Browser，可视化节点、边和查询结果。
- `bolt://127.0.0.1:7687`：Python 驱动使用的数据库连接。
- Docker named volume：容器重启或更新后保留图数据。

## 校验、导入、计数和查询

```powershell
Copy-Item .env.example .env
# 编辑 .env，或在当前 PowerShell 中设置：
$env:NEO4J_PASSWORD = "your-local-password"

# 不连接数据库，先检查全量 CSV
python -m src.retrieval.neo4j_client validate

# 建立节点唯一性约束
python -m src.retrieval.neo4j_client init

# 使用 UNWIND + MERGE 分批幂等导入
python -m src.retrieval.neo4j_client ingest --batch-size 2000

# 查看真实节点和关系数量
python -m src.retrieval.neo4j_client count

# 只扩展问题需要的关系类型
python -m src.retrieval.neo4j_client search "百日咳" `
  --relation has_symptom `
  --relation diagnosed_by `
  --relation belongs_to `
  --limit 12
```

程序会读取项目根目录的 `.env`，也可以由终端/部署系统设置环境变量，或把连接参数显式传给 CLI。密码不会出现在 JSON 结果里，`.env` 也已被 Git 忽略。

## 幂等性

节点和关系都使用 `MERGE`：同一个标签下相同 `name` 的节点不会重复创建；同一疾病、关系类型和目标实体构成的边也不会重复创建。全量导入可以安全重跑。验收应比较 `relationship_count_before` 和 `relationship_count_after`，不能只看本次提交了多少 CSV 行。

数据质量同步后，本机实测关系数为 **182,565**，节点数为 **24,237**；再次导入干净 CSV 数量保持不变。

## Browser 可视化查询

登录 Neo4j Browser 后可以直接运行：

```cypher
MATCH (d:Disease {name: '百日咳'})-[r]->(target)
RETURN d, r, target
LIMIT 50;
```

看到圆点和箭头代表图结构；圆点是节点，箭头是关系。也可先看数字：

```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label;
MATCH ()-[r]->() RETURN type(r) AS relation, count(*) AS count ORDER BY relation;
```

## 当前验收快照

| 项目 | 结果 |
| --- | ---: |
| 疾病节点 | 8,806 |
| 症状节点 | 5,997 |
| 检查节点 | 3,351 |
| 药品节点 | 3,825 |
| 科室节点 | 54 |
| 并发症节点 | 2,205 |
| 类型化节点总数 | 24,237 |
| 核心关系 | 182,565 |

`24,237` 个 typed nodes 大于离线统计的 `21,604` 个唯一文本实体，是因为同一名称如果同时作为症状、并发症或其他类型出现，会在 Neo4j 中保留为不同标签的节点。这是为了保留语义类型，不是重复导入。
