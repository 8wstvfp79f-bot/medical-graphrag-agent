# BGE Reranker and Context Budget

## What Phase 6 Adds

Phase 5 produced one auditable list of Vector and Graph evidence. Phase 6 turns that candidate list into a bounded LLM input:

```text
Hybrid evidence
  → BGE Cross-Encoder scoring
  → Top-K / optional score threshold
  → Vector + Graph source quotas
  → token and item budget
  → context blocks + E1-En citations
```

This stage still outputs evidence rather than a medical answer. The completed downstream stages consume it through Function Calling, evidence-bound answer generation, FastAPI/SSE, and the fixed evaluation pipeline.

## Embedding and Reranker Are Different Jobs

| Component | Reads at one time | Role | Output |
| --- | --- | --- | --- |
| `bge-base-zh-v1.5` embedding | query or chunk separately | Fast first-stage retrieval against 20,801 clean Milvus vectors | one 768-dimensional vector |
| `bge-reranker-base` Cross-Encoder | query and one candidate passage together | More accurate second-stage relevance ordering over a small candidate set | one relevance logit, normalized here with sigmoid |

The reranker is not written into Milvus and does not replace the stored embeddings. It only runs after Hybrid Retrieval has produced a small candidate list.

## Auditable Reranking

`src/retrieval/reranker.py` defines a backend protocol and `LocalCrossEncoderReranker`. The local adapter:

- accepts only an existing local model directory and never downloads silently;
- batches query/passage pairs and supports CUDA FP16;
- uses a maximum pair length of 512 by default;
- optionally converts raw logits to a 0-1 sigmoid score.

`rerank_evidence` does not mutate its input and preserves both stages:

- `rank`: original Hybrid rank;
- `retrieval_rank`: explicit copy of the original rank;
- `rerank_rank`: new Cross-Encoder order;
- `rerank_score`: model relevance score;
- `reranker_model`: model directory name;
- `excluded_evidence`: evidence removed by `top_k` or `min_score`, including its reason.

Scores are useful for ordering and debugging but are not calibrated medical-confidence probabilities.

## Context Budget

`src/generation/context_builder.py` selects whole evidence blocks under two hard limits: `max_tokens` and `max_items`. It first tries to satisfy the default minimum quotas `vector=1` and `graph=1`, then fills remaining space in rerank order. A Hybrid evidence item with both provenance types can satisfy both quotas.

The current `estimate_tokens` is deliberately conservative and tokenizer-independent: Chinese characters and punctuation count individually, while ASCII runs are approximated. A later LLM adapter can inject the exact model tokenizer. Evidence that cannot fit in full is excluded rather than silently cut in the middle.

The returned package contains:

- rendered `context` with `[E1 | type=... | disease=... | source=...]` headers;
- selected evidence with `context_rank`, `citation_id` and `context_token_count`;
- compact `citations` mapping E1-En to evidence IDs and sources;
- excluded evidence IDs;
- used, remaining and maximum token counts;
- selected source-type counts and unmet quotas.

## Running the Full Pipeline

Set local paths in `.env` or pass them explicitly:

```env
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

## Verified Real Result

The local acceptance run used Milvus collection `medical_chunks_bge_base_zh_v15_phase5`, Neo4j core relation `has_symptom`, local `BAAI/bge-base-zh-v1.5`, and local `BAAI/bge-reranker-base` on CUDA.

| Stage | Result |
| --- | --- |
| Vector candidates | 5 |
| Graph candidates | 6 |
| Fused evidence returned | 10 |
| Reranker candidates / returned | 10 / 8 |
| Excluded by rerank Top-K | 2 |
| Context selected | 3: 2 Vector + 1 Graph |
| Context budget | 1177 used / 1200 maximum / 23 remaining |
| Citations | E1-E3 |

The direct symptom overview moved from retrieval rank 5 to rerank rank 1 with score `0.99681491`. The graph fact `百日咳 -[has_symptom]-> 吸气时有蝉鸣音` was retained as E3, so the final context contains both descriptive text and a structured symptom relation.

## Test Coverage

Unit tests cover score validation, stable tie handling, Top-K/threshold exclusion reasons, non-mutation, token-budget enforcement, source quotas, Hybrid provenance, invalid inputs, and the integrated rerank-to-context pipeline. Real-runtime acceptance additionally proves both local models, Milvus, Neo4j and CUDA work together.
