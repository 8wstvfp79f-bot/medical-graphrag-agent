from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.generation.context_builder import (
    DEFAULT_CONTEXT_MAX_ITEMS,
    DEFAULT_CONTEXT_MAX_TOKENS,
    build_context,
)
from src.retrieval.reranker import (
    DEFAULT_RERANK_TOP_K,
    RerankBackend,
    rerank_evidence,
)


def prepare_evidence_context(
    query: str,
    evidence: Sequence[Mapping[str, object]],
    *,
    reranker_backend: RerankBackend,
    rerank_top_k: int | None = DEFAULT_RERANK_TOP_K,
    rerank_min_score: float | None = None,
    context_max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    context_max_items: int = DEFAULT_CONTEXT_MAX_ITEMS,
    min_type_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Rerank Hybrid evidence, then assemble an auditable context package."""

    reranked = rerank_evidence(
        query,
        evidence,
        backend=reranker_backend,
        top_k=rerank_top_k,
        min_score=rerank_min_score,
    )
    context = build_context(
        reranked["evidence"],
        max_tokens=context_max_tokens,
        max_items=context_max_items,
        min_type_counts=min_type_counts,
    )
    return {
        "reranking": {
            "config": reranked["reranker"],
            "stats": reranked["stats"],
            "excluded_evidence": reranked["excluded_evidence"],
        },
        "reranked_evidence": reranked["evidence"],
        "context": context,
    }

