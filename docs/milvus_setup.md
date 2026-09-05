# Milvus Setup and Phase 3 Runbook

> Data-quality refresh complete: the formal `medical_chunks_bge_base_zh_v15_phase5` Collection has been rebuilt with 20,801 clean `bge-base-zh-v1.5` vectors. Logical and storage row counts both equal 20,801; the separate Hash test Collection was not modified.

## What Phase 3 Implements

`src/retrieval/milvus_client.py` now owns the Milvus-side vector retrieval lifecycle:

1. connect to a configured Milvus URI;
2. create the medical chunk collection and vector index;
3. stream `chunks.jsonl` in bounded batches;
4. generate embeddings with the selected backend;
5. upsert by `chunk_id`, so rerunning ingestion is idempotent;
6. report both current logical entities and physical row versions;
7. run Top-K vector search with Metadata Filtering;
8. return `score`, complete `metadata`, and `matched_metadata`.

The module imports `pymilvus` lazily. Unit tests can therefore run without a live database, while production commands fail with a clear dependency message when `pymilvus` is missing.

## External Prerequisites

Phase 3 production verification needs both:

- a Python environment containing `pymilvus`;
- either a reachable Milvus service URI or a Milvus Lite database path supported by the local platform/runtime.

This repository does not automatically download dependencies, models, containers, or database binaries. Configure credentials through `MILVUS_URI` and `MILVUS_TOKEN` or pass `--uri` / `--token` explicitly.

## Collection Schema

| Field | Milvus type | Purpose |
| --- | --- | --- |
| `chunk_id` | `VARCHAR`, primary key | Stable idempotency key |
| `doc_id` | `VARCHAR` | Trace hit back to its document |
| `disease_name` | `VARCHAR` | Auditable disease label |
| `text` | `VARCHAR` | Retrieved medical chunk |
| `metadata` | `JSON` | `disease_name/category/department/source` filters |
| `embedding_model` | `VARCHAR` | Embedding version audit |
| `embedding` | `FLOAT_VECTOR` | Vector search field |

The default dimension is 384 for the Hash smoke-test collection. The production `bge-base-zh-v1.5` collection uses dimension 768. Both use `COSINE` and `AUTOINDEX`. The configured dimension must match both the embedding model and any existing collection. A mismatch fails before ingestion or query.

## Commands

Run every command from the repository root.

Validate every real chunk, embedding dimension, metadata object, and Milvus field length without a database connection:

```bash
python -m src.retrieval.milvus_client validate --backend hash --batch-size 64
```

This command proves that the ingestion payload can be generated. It does not prove that Milvus accepted or persisted the records.

Create/load the collection:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 init
```

Ingest a small smoke-test batch with the offline hash embedding:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 ingest --backend hash --limit 100
```

Ingest all 20,801 chunks:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 ingest --backend hash --batch-size 64
```

The hash backend validates infrastructure only. The verified production embedding model is `BAAI/bge-base-zh-v1.5` (768 dimensions), stored in a separate collection:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 --collection medical_chunks_bge_base_zh_v15 --dimension 768 ingest --backend bge --model-path /path/to/bge-base-zh-v1.5 --device cuda --batch-size 16
```

Count stored entities:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 count
```

The count command intentionally returns two values:

- `logical_entity_count`: current unique `chunk_id` values returned by a strong-consistency query iterator; use this for ingestion and idempotency acceptance.
- `storage_row_count`: Milvus' physical row-version statistic. Because upsert logically replaces an old version by writing a new version, this number can grow until background compaction removes superseded versions.

Run metadata-aware search:

```bash
python -m src.retrieval.milvus_client --uri http://localhost:19530 --collection medical_chunks_bge_base_zh_v15 --dimension 768 search "百日咳有哪些症状" --backend bge --model-path /path/to/bge-base-zh-v1.5 --device cuda --top-k 5 --disease-name 百日咳 --department 儿科
```

`--category` and `--department` can be repeated to express OR within one field. Different filter fields are combined with AND.

## Verified Local Runtime

The local Windows environment has now verified the real database path with Milvus 3.0.0 and PyMilvus 3.0.1:

- connected to `http://127.0.0.1:19530` and created `medical_chunks_hash` with dimension 384 and `COSINE`;
- upserted the same 100 Hash-embedded chunks twice;
- a strong-consistency query still returned 100 unique `chunk_id` values, so logical idempotency passed;
- Milvus' physical `row_count` was observed at 200 immediately after the second upsert and later returned to 100 after background compaction;
- a real unfiltered Top-3 query returned text, score, complete metadata, and matched metadata.

The production BGE path has also been verified on an RTX 4060 Ti with PyTorch 2.13.0 CUDA 13.0 and Sentence Transformers 6.0.1:

- downloaded the official `BAAI/bge-base-zh-v1.5` model locally and confirmed 768-dimensional normalized GPU embeddings;
- created the isolated `medical_chunks_bge_base_zh_v15` collection with `COSINE` and `AUTOINDEX`;
- rebuilt and inserted all 20,801 clean chunks in 326 batches of 64;
- confirmed `logical_entity_count = 20,801`, `storage_row_count = 20,801`, and zero rejected-term occurrences in the clean artifacts;
- kept `medical_chunks_hash` isolated at 100 logical test records;
- an unfiltered “百日咳” query returned the medically related traditional-Chinese name “顿呛” above the exact disease, exposing a real semantic Bad Case;
- adding `disease_name=百日咳` and `department=儿科` removed that ambiguity and returned only matching Top-3 evidence.

The BGE adapter prepends the model's recommended Chinese retrieval instruction to short queries while leaving document chunks unchanged.

### Milvus 3.0 local-storage restart note

After a Docker restart, the two early local-storage collections hit an upstream Milvus packed-stats path defect: the files still existed, but the loader requested `files/json_stats/...` while part of the stats had been written below a duplicated local root. The affected collections were preserved and released rather than dropped or rewritten.

For Phase 5, a separate `medical_chunks_bge_base_zh_v15_phase5` collection was initially created with 20,802 BGE chunks. A Docker restart once exposed a Milvus 3.0 Storage V2 path defect; a narrowly scoped compatibility link restored the collection and allowed the real Hybrid + Reranker + Context Budget acceptance to pass. After the field-quality refresh, the same formal collection was explicitly recreated from clean artifacts with 20,801 vectors in 326 batches of 64. Its current logical and storage counts are both 20,801, and live Metadata Filtering again passed.

This is a local recovery record, not a production deployment recommendation. Production should use an upstream-fixed Milvus version or supported object storage, then verify cold restart before release. The client now passes a configurable collection-load timeout (`MILVUS_LOAD_TIMEOUT_SECONDS`, default 60) so path failures surface as errors instead of waiting indefinitely. See `docs/hybrid_retrieval.md` and `docs/reranker_context.md`.

## Idempotency and Safety

- Normal ingestion uses Milvus `upsert` with `chunk_id` as the primary key. Running the same input twice keeps the same logical unique entities. The physical `storage_row_count` can still increase because Milvus retains superseded row versions before compaction.
- Existing collections are reused and their vector dimension is checked.
- The collection is dropped only when `--recreate` is explicitly provided.
- `--limit` is recommended for the first connectivity test.
- `data/raw/`, `data/processed/`, Milvus tokens, and generated retrieval results must remain outside Git.

## Definition of Done

Phase 3 application behavior is verified against a real server. The completed acceptance checks are:

- the collection exists and is loaded;
- `logical_entity_count` is 20,801 after clean full ingestion;
- repeating ingestion keeps `logical_entity_count` at 20,801, even if `storage_row_count` increases;
- an unfiltered query returns Top-K text, score, and metadata;
- a disease/department-filtered Bad Case returns only matching evidence;
- the same embedding model and dimension are used for ingestion and query.

Container-restart persistence under the affected Milvus 3.0.0 local-storage image required a compatibility link and remains a known runtime limitation, not a completed production-readiness claim.

The repository's Fake Milvus tests validate schema construction, index parameters, batching, idempotency, dimension mismatch protection, filter pushdown, and application-level Bad Case filtering. They do not replace the real-runtime checks above.
