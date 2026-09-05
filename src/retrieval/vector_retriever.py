from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from src.retrieval.metadata_filter import (
    MetadataFilterInput,
    NormalizedMetadataFilter,
    build_milvus_filter_expression,
    matched_metadata,
    metadata_matches,
    normalize_metadata_filter,
)


class VectorSearchBackend(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: NormalizedMetadataFilter,
    ) -> Sequence[Mapping[str, object]]:
        """Return vector candidates ordered from best to worst."""


class MilvusVectorBackend:
    """Small adapter around a MilvusClient-compatible object.

    The collection is expected to store chunk payload fields plus a JSON field named
    ``metadata``. Keeping the adapter independent from pymilvus makes it easy to unit
    test and lets the application own connection lifecycle and credentials.
    """

    def __init__(
        self,
        client: object,
        collection_name: str,
        query_embedder: Callable[[str], Sequence[float]],
        *,
        vector_field: str = "embedding",
        metadata_field: str = "metadata",
        metric_type: str = "COSINE",
        expected_dimension: int | None = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.query_embedder = query_embedder
        self.vector_field = vector_field
        self.metadata_field = metadata_field
        self.metric_type = metric_type
        self.expected_dimension = expected_dimension

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: NormalizedMetadataFilter,
    ) -> Sequence[Mapping[str, object]]:
        search = getattr(self.client, "search", None)
        if not callable(search):
            raise TypeError("Milvus client must expose a callable search method")

        query_vector = list(self.query_embedder(query))
        if self.expected_dimension is not None and len(query_vector) != self.expected_dimension:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected {self.expected_dimension}, got {len(query_vector)}"
            )

        kwargs: dict[str, object] = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "anns_field": self.vector_field,
            "limit": top_k,
            "search_params": {"metric_type": self.metric_type, "params": {}},
            "output_fields": ["chunk_id", "doc_id", "text", self.metadata_field],
        }
        filter_expression = build_milvus_filter_expression(
            metadata_filter,
            metadata_field=self.metadata_field,
        )
        if filter_expression:
            kwargs["filter"] = filter_expression

        raw_results = search(**kwargs)
        if (
            isinstance(raw_results, Sequence)
            and raw_results
            and isinstance(raw_results[0], Sequence)
            and not isinstance(raw_results[0], (str, bytes, Mapping))
        ):
            return raw_results[0]
        return raw_results


def _candidate_payload(candidate: Mapping[str, object]) -> dict[str, object]:
    entity = candidate.get("entity")
    payload = dict(entity) if isinstance(entity, Mapping) else dict(candidate)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = candidate.get("metadata")
    payload["metadata"] = metadata if isinstance(metadata, Mapping) else {}

    if "score" not in payload:
        if "score" in candidate:
            payload["score"] = candidate["score"]
        elif "distance" in candidate:
            payload["score"] = candidate["distance"]
    return payload


def vector_search(
    query: str,
    *,
    backend: VectorSearchBackend,
    top_k: int = 5,
    metadata_filter: MetadataFilterInput | None = None,
    candidate_multiplier: int = 4,
) -> list[dict[str, object]]:
    """Run metadata-aware vector retrieval and return auditable hits.

    The filter is pushed into the backend for efficient pre-filtering and applied a
    second time here as a correctness guard. This prevents a backend/schema mismatch
    from leaking a similarly named disease or a wrong department into the context.
    """
    query = " ".join(query.split())
    if not query:
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be greater than zero")

    normalized_filter = normalize_metadata_filter(metadata_filter)
    candidate_top_k = top_k
    if normalized_filter:
        candidate_top_k = max(top_k, top_k * candidate_multiplier)

    candidates = backend.search(
        query,
        top_k=candidate_top_k,
        metadata_filter=normalized_filter,
    )
    hits: list[dict[str, object]] = []
    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        payload = _candidate_payload(candidate)
        metadata = payload["metadata"]
        if not metadata_matches(metadata, normalized_filter):
            continue

        chunk_id = str(payload.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        score_value = payload.get("score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        hits.append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(payload.get("doc_id", "")),
                "text": str(payload.get("text", "")),
                "score": score,
                "metadata": matched_metadata(metadata),
                "matched_metadata": matched_metadata(metadata),
            }
        )
        if len(hits) >= top_k:
            break

    return hits
