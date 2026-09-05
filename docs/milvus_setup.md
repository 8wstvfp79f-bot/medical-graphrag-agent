# Milvus 向量检索配置与运行手册

> 数据质量刷新已完成：正式集合 `medical_chunks_bge_base_zh_v15_phase5` 已使用 20,801 条干净的 `bge-base-zh-v1.5` 向量重建。逻辑实体数和存储行数均为 20,801；独立的 Hash 测试集合未修改。

## 本模块实现的功能

`src/retrieval/milvus_client.py` 负责 Milvus 侧的完整向量检索生命周期：

1. 连接指定的 Milvus 地址；
2. 创建医疗分块集合和向量索引；
3. 以有界批次读取 `chunks.jsonl`；
4. 使用指定后端生成向量；
5. 以 `chunk_id` 为主键执行 upsert，使重复导入保持逻辑幂等；
6. 同时报告当前逻辑实体数和物理行版本数；
7. 执行带元数据过滤的 Top-K 向量搜索；
8. 返回分数、完整元数据和实际命中的元数据。

模块会延迟导入 `pymilvus`。因此单元测试不需要真实数据库；正式命令缺少依赖时会给出明确错误。

## 外部依赖

正式验证需要：

- 安装了 `pymilvus` 的 Python 环境；
- 可访问的 Milvus 服务地址，或当前平台支持的 Milvus Lite 数据库路径；
- 已下载到本地的 BGE 模型目录。

仓库不会自动下载依赖、模型、容器或数据库二进制文件。凭据通过 `MILVUS_URI`、`MILVUS_TOKEN` 或命令行参数配置，不应提交到 Git。

## 集合字段结构

| 字段 | Milvus 类型 | 用途 |
|---|---|---|
| `chunk_id` | `VARCHAR`，主键 | 稳定的幂等键 |
| `doc_id` | `VARCHAR` | 将检索结果追溯到原始文档 |
| `disease_name` | `VARCHAR` | 可审计的疾病名称 |
| `text` | `VARCHAR` | 被检索的医疗文本分块 |
| `metadata` | `JSON` | `disease_name/category/department/source` 过滤条件 |
| `embedding_model` | `VARCHAR` | 向量模型版本审计 |
| `embedding` | `FLOAT_VECTOR` | 向量搜索字段 |

Hash 冒烟测试集合默认使用 384 维；正式 `bge-base-zh-v1.5` 集合使用 768 维。两者都使用 `COSINE` 和 `AUTOINDEX`。配置维度必须同时匹配向量模型和现有集合，否则在导入或查询前直接失败。

## 日常启动

Docker Desktop 就绪后启动已有容器：

```powershell
docker start milvus-standalone
docker ps --filter "name=milvus-standalone"
Test-NetConnection 127.0.0.1 -Port 19530
```

端口成功后检查集合：

```powershell
python -m src.retrieval.milvus_client count
```

如果集合已经包含 20,801 个逻辑实体，日常启动不需要重新导入。

## 首次初始化与导入

以下命令都从仓库根目录执行。

先在不连接数据库的情况下校验所有分块、向量维度、元数据对象和字段长度：

```powershell
python -m src.retrieval.milvus_client validate --backend hash --batch-size 64
```

这只能证明导入载荷能够生成，不能证明 Milvus 已经接收或持久化数据。

创建并加载集合：

```powershell
python -m src.retrieval.milvus_client --uri http://127.0.0.1:19530 init
```

先使用离线 Hash 向量写入 100 条，验证连接和字段结构：

```powershell
python -m src.retrieval.milvus_client --uri http://127.0.0.1:19530 ingest --backend hash --limit 100
```

Hash 后端只验证工程链路。正式语义检索使用独立 BGE 集合：

```powershell
python -m src.retrieval.milvus_client `
  --uri http://127.0.0.1:19530 `
  --collection medical_chunks_bge_base_zh_v15_phase5 `
  --dimension 768 `
  ingest `
  --backend bge `
  --model-path C:\path\to\bge-base-zh-v1.5 `
  --device cuda `
  --batch-size 64
```

## 计数口径

```powershell
python -m src.retrieval.milvus_client --uri http://127.0.0.1:19530 count
```

计数命令返回两个值：

- `logical_entity_count`：强一致性查询得到的当前唯一 `chunk_id` 数量，用于导入和幂等验收；
- `storage_row_count`：Milvus 的物理行版本统计。upsert 会写入新版本，因此后台压缩完成前该值可能暂时增加。

判断重复导入是否正确，应以逻辑实体数和主键抽样为准，不能只看物理行数。

## 带元数据过滤的查询

```powershell
python -m src.retrieval.milvus_client `
  --uri http://127.0.0.1:19530 `
  --collection medical_chunks_bge_base_zh_v15_phase5 `
  --dimension 768 `
  search "百日咳有哪些症状" `
  --backend bge `
  --model-path C:\path\to\bge-base-zh-v1.5 `
  --device cuda `
  --top-k 5 `
  --disease-name 百日咳 `
  --department 儿科
```

`--category` 和 `--department` 可以重复传入，表示同一字段内部使用 OR；不同字段之间使用 AND。

## 本地真实验收

Windows 本地环境已使用 Milvus 3.0.0 和 PyMilvus 3.0.1 验证：

- 成功连接 `http://127.0.0.1:19530`；
- 创建 384 维、`COSINE` 距离的 `medical_chunks_hash`；
- 对同一批 100 条 Hash 分块重复 upsert 后，强一致性查询仍只返回 100 个唯一 `chunk_id`；
- 第二次 upsert 后物理 `row_count` 曾短暂为 200，后台压缩后恢复为 100；
- Top-3 查询能够返回文本、分数、完整元数据和实际命中元数据。

正式 BGE 链路也已完成：

- 使用 `BAAI/bge-base-zh-v1.5` 生成 768 维归一化向量；
- 将 20,801 个干净分块写入独立正式集合；
- 验证逻辑实体数和存储行数均为 20,801；
- Hash 测试集合继续隔离保存 100 条逻辑记录；
- 未过滤的“百日咳”查询暴露了“顿呛”排在精确疾病前的真实语义误召回；
- 加入 `disease_name=百日咳` 和 `department=儿科` 后，只返回满足条件的证据。

BGE 适配器会为短查询添加模型建议的中文检索指令，文档分块不添加该指令。

## Milvus 3.0 本地存储重启问题

早期本地集合在 Docker 重启后遇到上游 packed stats 路径缺陷：数据文件仍存在，但加载器请求的统计文件路径不一致，导致集合无法加载。受影响集合被保留并释放，没有直接删除或原地改写。

正式集合后来使用清洗后的 20,801 个分块显式重建，并再次通过元数据过滤。生产环境应优先使用包含上游修复的 Milvus 版本或受支持的对象存储，并在发布前执行冷重启后的加载和查询验证。相关记录：

- <https://github.com/milvus-io/milvus/issues/45959>
- <https://github.com/milvus-io/milvus/issues/49419>

客户端支持 `MILVUS_LOAD_TIMEOUT_SECONDS`，默认 60 秒，使加载问题能够明确报错，而不是无限等待。

## 幂等与安全边界

- 正常导入使用 `chunk_id` 主键 upsert；重复输入保持相同逻辑实体数。
- 复用已有集合前会检查向量维度。
- 只有显式传入 `--recreate` 才会删除并重建集合。
- 第一次连接测试建议使用 `--limit`。
- `data/raw/`、`data/processed/`、Milvus token 和生成的检索结果不能提交到 Git。
- 不要对 Hash 测试集合执行正式数据重建。

## 验收标准

- 集合存在且可以加载；
- 干净全量导入后的 `logical_entity_count` 为 20,801；
- 重复导入后逻辑实体数保持不变；
- 未过滤查询返回 Top-K 文本、分数和元数据；
- 疾病与科室过滤能够排除相近疾病的错误证据；
- 导入和查询使用同一向量模型及维度；
- 容器重启后仍能成功加载并查询正式集合。

Fake Milvus 单元测试可以验证字段结构、索引参数、批处理、幂等、维度冲突保护、过滤下推和应用层二次检查，但不能替代真实数据库验收。
