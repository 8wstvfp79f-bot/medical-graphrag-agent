from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.retrieval.embedding import create_embedding_model
from src.retrieval.graph_retriever import (
    GraphSearchBackend,
    Neo4jGraphBackend,
    graph_search,
)
from src.retrieval.graph_schema import CORE_RELATIONS
from src.retrieval.metadata_filter import (
    MetadataFilterInput,
    NormalizedMetadataFilter,
    normalize_metadata_filter,
)
from src.retrieval.milvus_client import (
    DEFAULT_LOAD_TIMEOUT_SECONDS,
    MedicalMilvusStore,
    MilvusCollectionError,
    MilvusSettings,
    create_milvus_client,
)
from src.retrieval.neo4j_client import (
    Neo4jSettings,
    create_neo4j_driver,
    load_project_environment,
)
from src.retrieval.reranker import LocalCrossEncoderReranker
from src.retrieval.retrieval_pipeline import prepare_evidence_context
from src.retrieval.vector_retriever import (
    MilvusVectorBackend,
    VectorSearchBackend,
    vector_search,
)


DEFAULT_COLLECTION_NAME = "medical_chunks_bge_base_zh_v15"
DEFAULT_DIMENSION = 768
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60


class HybridRetrievalError(ValueError):
    """Raised when vector and graph retrieval inputs cannot be fused safely."""


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _content_key(disease_name: str, content: str) -> str:
    return f"content:{disease_name.casefold()}:{content.casefold()}"


def _rrf_contribution(weight: float, rrf_k: int, rank: int) -> float:
    return weight / (rrf_k + rank)


def _normalize_hybrid_filter(
    disease_name: str,
    metadata_filter: MetadataFilterInput | None,
) -> NormalizedMetadataFilter:
    normalized = normalize_metadata_filter(metadata_filter)
    configured_diseases = normalized.get("disease_name")
    if configured_diseases:
        configured_keys = {value.casefold() for value in configured_diseases}
        if configured_keys != {disease_name.casefold()}:
            raise HybridRetrievalError(
                "Hybrid retrieval requires metadata_filter.disease_name to match "
                f"the graph disease_name exactly: {disease_name}"
            )
    normalized["disease_name"] = (disease_name,)
    return normalized


def _vector_candidate(
    hit: Mapping[str, object],
    *,
    rank: int,
    weight: float,
    rrf_k: int,
) -> dict[str, object]:
    chunk_id = _clean(hit.get("chunk_id"))
    content = _clean(hit.get("text"))
    metadata_value = hit.get("matched_metadata") or hit.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    disease_name = _clean(metadata.get("disease_name"))
    source = _clean(metadata.get("source"))
    if not chunk_id or not content or not disease_name or not source:
        raise HybridRetrievalError(
            "Vector evidence requires chunk_id, text, metadata.disease_name, and source"
        )

    score_value = hit.get("score")
    try:
        retrieval_score = float(score_value) if score_value is not None else None
    except (TypeError, ValueError) as exc:
        raise HybridRetrievalError("Vector evidence score must be numeric") from exc

    contribution = _rrf_contribution(weight, rrf_k, rank)
    content_key = _content_key(disease_name, content)
    return {
        "_dedupe_keys": {f"vector:{chunk_id}", content_key},
        "evidence_id": _stable_id("evidence", content_key),
        "evidence_type": "vector",
        "content": content,
        "disease_name": disease_name,
        "source": source,
        "sources": [source],
        "rank": 0,
        "best_source_rank": rank,
        "retrieval_score": retrieval_score,
        "fusion_score": contribution,
        "metadata": metadata,
        "relation": None,
        "entity_name": None,
        "entity_type": None,
        "path_text": None,
        "provenance": [
            {
                "evidence_type": "vector",
                "record_id": chunk_id,
                "source": source,
                "source_rank": rank,
                "retrieval_score": retrieval_score,
                "fusion_contribution": contribution,
            }
        ],
    }


def _graph_candidate(
    hit: Mapping[str, object],
    *,
    rank: int,
    weight: float,
    rrf_k: int,
) -> dict[str, object]:
    disease_name = _clean(hit.get("disease_name"))
    relation = _clean(hit.get("relation"))
    entity_name = _clean(hit.get("entity_name"))
    entity_type = _clean(hit.get("entity_type"))
    source = _clean(hit.get("source"))
    path_text = _clean(hit.get("path_text"))
    if not disease_name or not relation or not entity_name or not source or not path_text:
        raise HybridRetrievalError(
            "Graph evidence requires disease_name, relation, entity_name, source, "
            "and path_text"
        )

    fact_key = (
        f"graph:{disease_name.casefold()}:{relation.casefold()}:"
        f"{entity_name.casefold()}"
    )
    contribution = _rrf_contribution(weight, rrf_k, rank)
    return {
        "_dedupe_keys": {fact_key, _content_key(disease_name, path_text)},
        "evidence_id": _stable_id("evidence", fact_key),
        "evidence_type": "graph",
        "content": path_text,
        "disease_name": disease_name,
        "source": source,
        "sources": [source],
        "rank": 0,
        "best_source_rank": rank,
        "retrieval_score": None,
        "fusion_score": contribution,
        "metadata": {
            "disease_name": disease_name,
            "source": source,
            "relation": relation,
            "entity_name": entity_name,
            "entity_type": entity_type,
        },
        "relation": relation,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "path_text": path_text,
        "provenance": [
            {
                "evidence_type": "graph",
                "record_id": fact_key,
                "source": source,
                "source_rank": rank,
                "retrieval_score": None,
                "fusion_contribution": contribution,
            }
        ],
    }


def _merge_candidates(candidates: Sequence[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
    merged: list[dict[str, object]] = []
    duplicates = 0
    for candidate in candidates:
        candidate_keys = candidate["_dedupe_keys"]
        assert isinstance(candidate_keys, set)
        match = next(
            (
                existing
                for existing in merged
                if candidate_keys.intersection(existing["_dedupe_keys"])
            ),
            None,
        )
        if match is None:
            merged.append(candidate)
            continue

        duplicates += 1
        match_keys = match["_dedupe_keys"]
        assert isinstance(match_keys, set)
        match_keys.update(candidate_keys)
        match["fusion_score"] = float(match["fusion_score"]) + float(
            candidate["fusion_score"]
        )
        match["best_source_rank"] = min(
            int(match["best_source_rank"]), int(candidate["best_source_rank"])
        )

        sources = match["sources"]
        assert isinstance(sources, list)
        for source in candidate["sources"]:
            if source not in sources:
                sources.append(source)
        provenance = match["provenance"]
        assert isinstance(provenance, list)
        provenance.extend(candidate["provenance"])

        if match["evidence_type"] != candidate["evidence_type"]:
            match["evidence_type"] = "hybrid"
            if match.get("relation") is None:
                for field in ("relation", "entity_name", "entity_type", "path_text"):
                    match[field] = candidate.get(field)
        current_score = match.get("retrieval_score")
        candidate_score = candidate.get("retrieval_score")
        if candidate_score is not None and (
            current_score is None or float(candidate_score) > float(current_score)
        ):
            match["retrieval_score"] = candidate_score

    return merged, duplicates


def fuse_evidence(
    vector_hits: Sequence[Mapping[str, object]],
    graph_hits: Sequence[Mapping[str, object]],
    *,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Normalize, deduplicate, and fuse two ranked evidence lists with RRF."""
    if top_k <= 0:
        raise HybridRetrievalError("top_k must be greater than zero")
    if rrf_k <= 0:
        raise HybridRetrievalError("rrf_k must be greater than zero")
    if vector_weight <= 0 or graph_weight <= 0:
        raise HybridRetrievalError("vector_weight and graph_weight must be positive")

    candidates = [
        _vector_candidate(hit, rank=rank, weight=vector_weight, rrf_k=rrf_k)
        for rank, hit in enumerate(vector_hits, start=1)
    ]
    candidates.extend(
        _graph_candidate(hit, rank=rank, weight=graph_weight, rrf_k=rrf_k)
        for rank, hit in enumerate(graph_hits, start=1)
    )
    merged, duplicates = _merge_candidates(candidates)
    source_priority = {"hybrid": 0, "vector": 1, "graph": 2}
    merged.sort(
        key=lambda item: (
            -float(item["fusion_score"]),
            int(item["best_source_rank"]),
            source_priority[str(item["evidence_type"])],
            str(item["evidence_id"]),
        )
    )

    evidence: list[dict[str, object]] = []
    for rank, item in enumerate(merged[:top_k], start=1):
        item = dict(item)
        item.pop("_dedupe_keys", None)
        item["rank"] = rank
        item["fusion_score"] = round(float(item["fusion_score"]), 12)
        provenance = item.get("provenance", [])
        if isinstance(provenance, list):
            for entry in provenance:
                if isinstance(entry, dict):
                    entry["fusion_contribution"] = round(
                        float(entry["fusion_contribution"]), 12
                    )
        evidence.append(item)

    return evidence, {
        "vector_candidates": len(vector_hits),
        "graph_candidates": len(graph_hits),
        "candidates_before_deduplication": len(candidates),
        "duplicates_merged": duplicates,
        "candidates_after_deduplication": len(merged),
        "returned_evidence": len(evidence),
    }


def hybrid_search(
    query: str,
    disease_name: str,
    *,
    vector_backend: VectorSearchBackend,
    graph_backend: GraphSearchBackend,
    top_k: int = DEFAULT_TOP_K,
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    metadata_filter: MetadataFilterInput | None = None,
    relations: list[str] | tuple[str, ...] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
    allow_partial: bool = False,
) -> dict[str, object]:
    query = _clean(query)
    disease_name = _clean(disease_name)
    if not query:
        raise HybridRetrievalError("query must not be empty")
    if not disease_name:
        raise HybridRetrievalError("disease_name must not be empty")
    if top_k <= 0:
        raise HybridRetrievalError("top_k must be greater than zero")
    vector_limit = vector_top_k if vector_top_k is not None else top_k
    graph_limit = graph_top_k if graph_top_k is not None else top_k
    if vector_limit <= 0 or graph_limit <= 0:
        raise HybridRetrievalError("vector_top_k and graph_top_k must be positive")

    normalized_filter = _normalize_hybrid_filter(disease_name, metadata_filter)
    vector_hits: Sequence[Mapping[str, object]] = []
    graph_hits: Sequence[Mapping[str, object]] = []
    errors: dict[str, str] = {}

    try:
        vector_hits = vector_search(
            query,
            backend=vector_backend,
            top_k=vector_limit,
            metadata_filter=normalized_filter,
        )
    except Exception as exc:
        if not allow_partial:
            raise
        errors["vector"] = f"{type(exc).__name__}: {exc}"

    try:
        graph_hits = graph_search(
            disease_name,
            backend=graph_backend,
            relations=relations,
            limit=graph_limit,
        )
    except Exception as exc:
        if not allow_partial:
            raise
        errors["graph"] = f"{type(exc).__name__}: {exc}"

    evidence, stats = fuse_evidence(
        vector_hits,
        graph_hits,
        top_k=top_k,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
    )
    return {
        "query": query,
        "disease_name": disease_name,
        "metadata_filter": {
            field: list(values) for field, values in normalized_filter.items()
        },
        "relations": list(relations) if relations is not None else list(CORE_RELATIONS),
        "strategy": {
            "name": "weighted_reciprocal_rank_fusion",
            "rrf_k": rrf_k,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight,
            "note": "fusion_score is for candidate fusion, not a calibrated relevance probability",
        },
        "stats": stats,
        "errors": errors,
        "evidence": evidence,
    }


async def hybrid_search_async(
    query: str,
    disease_name: str,
    *,
    vector_backend: VectorSearchBackend,
    graph_backend: GraphSearchBackend,
    top_k: int = DEFAULT_TOP_K,
    vector_top_k: int | None = None,
    graph_top_k: int | None = None,
    metadata_filter: MetadataFilterInput | None = None,
    relations: list[str] | tuple[str, ...] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
    allow_partial: bool = False,
) -> dict[str, object]:
    """Run independent Milvus and Neo4j searches concurrently.

    Blocking database/model clients stay unchanged and are moved to worker threads.
    Fusion remains deterministic after both branches finish.
    """

    query = _clean(query)
    disease_name = _clean(disease_name)
    if not query:
        raise HybridRetrievalError("query must not be empty")
    if not disease_name:
        raise HybridRetrievalError("disease_name must not be empty")
    if top_k <= 0:
        raise HybridRetrievalError("top_k must be greater than zero")
    vector_limit = vector_top_k if vector_top_k is not None else top_k
    graph_limit = graph_top_k if graph_top_k is not None else top_k
    if vector_limit <= 0 or graph_limit <= 0:
        raise HybridRetrievalError("vector_top_k and graph_top_k must be positive")

    normalized_filter = _normalize_hybrid_filter(disease_name, metadata_filter)
    vector_result, graph_result = await asyncio.gather(
        asyncio.to_thread(
            vector_search,
            query,
            backend=vector_backend,
            top_k=vector_limit,
            metadata_filter=normalized_filter,
        ),
        asyncio.to_thread(
            graph_search,
            disease_name,
            backend=graph_backend,
            relations=relations,
            limit=graph_limit,
        ),
        return_exceptions=True,
    )

    errors: dict[str, str] = {}
    vector_hits: Sequence[Mapping[str, object]] = []
    graph_hits: Sequence[Mapping[str, object]] = []
    if isinstance(vector_result, BaseException):
        if not allow_partial:
            raise vector_result
        errors["vector"] = f"{type(vector_result).__name__}: {vector_result}"
    else:
        vector_hits = vector_result
    if isinstance(graph_result, BaseException):
        if not allow_partial:
            raise graph_result
        errors["graph"] = f"{type(graph_result).__name__}: {graph_result}"
    else:
        graph_hits = graph_result

    evidence, stats = fuse_evidence(
        vector_hits,
        graph_hits,
        top_k=top_k,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
    )
    return {
        "query": query,
        "disease_name": disease_name,
        "metadata_filter": {
            field: list(values) for field, values in normalized_filter.items()
        },
        "relations": list(relations) if relations is not None else list(CORE_RELATIONS),
        "strategy": {
            "name": "async_weighted_reciprocal_rank_fusion",
            "rrf_k": rrf_k,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight,
            "execution": "vector_and_graph_concurrent",
            "note": "fusion_score is for candidate fusion, not a calibrated relevance probability",
        },
        "stats": stats,
        "errors": errors,
        "evidence": evidence,
    }


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Milvus vector evidence with Neo4j graph evidence"
    )
    parser.add_argument("query")
    parser.add_argument("--disease-name", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--vector-top-k", type=int)
    parser.add_argument("--graph-top-k", type=int)
    parser.add_argument("--relation", action="append", choices=CORE_RELATIONS)
    parser.add_argument("--category", action="append")
    parser.add_argument("--department", action="append")
    parser.add_argument("--source")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight", type=float, default=1.0)
    parser.add_argument("--allow-partial", action="store_true")

    parser.add_argument(
        "--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MILVUS_COLLECTION", DEFAULT_COLLECTION_NAME),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.getenv("MILVUS_DIMENSION", str(DEFAULT_DIMENSION))),
    )
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument(
        "--milvus-load-timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "MILVUS_LOAD_TIMEOUT_SECONDS", str(DEFAULT_LOAD_TIMEOUT_SECONDS)
            )
        ),
    )
    parser.add_argument("--model-path", type=Path, default=_optional_path(os.getenv("BGE_MODEL_PATH")))
    parser.add_argument("--device", default=os.getenv("BGE_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--reranker-model-path",
        type=Path,
        default=_optional_path(os.getenv("BGE_RERANKER_MODEL_PATH")),
    )
    parser.add_argument(
        "--reranker-device", default=os.getenv("BGE_RERANKER_DEVICE")
    )
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--rerank-min-score", type=float)
    parser.add_argument("--context-max-tokens", type=int, default=2000)
    parser.add_argument("--context-max-items", type=int, default=6)

    parser.add_argument(
        "--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    )
    parser.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    return parser.parse_args()


def _metadata_filter_from_args(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {"disease_name": args.disease_name}
    if args.category:
        result["category"] = args.category
    if args.department:
        result["department"] = args.department
    if args.source:
        result["source"] = args.source
    return result


def main() -> None:
    configure_stdout()
    load_project_environment()
    args = parse_args()
    if args.model_path is None:
        raise SystemExit(
            "A local BGE model is required. Set BGE_MODEL_PATH or pass --model-path."
        )

    embedding_model = create_embedding_model(
        "bge",
        dimension=args.dimension,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    if embedding_model.dimension != args.dimension:
        raise SystemExit(
            "BGE model dimension does not match --dimension: "
            f"model={embedding_model.dimension}, configured={args.dimension}"
        )

    milvus_settings = MilvusSettings(
        uri=args.milvus_uri,
        token=args.milvus_token,
        collection_name=args.collection,
        dimension=args.dimension,
        load_timeout_seconds=args.milvus_load_timeout_seconds,
    )
    neo4j_settings = Neo4jSettings(
        uri=args.neo4j_uri,
        username=args.neo4j_username,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    milvus_client = create_milvus_client(milvus_settings)
    neo4j_driver = create_neo4j_driver(neo4j_settings)
    try:
        store = MedicalMilvusStore(milvus_client, milvus_settings)
        if not store.has_collection():
            raise MilvusCollectionError(
                f"Milvus collection does not exist: {milvus_settings.collection_name}"
            )
        store.validate_existing_collection()
        store.load_collection()
        vector_backend = MilvusVectorBackend(
            milvus_client,
            milvus_settings.collection_name,
            embedding_model.embed_query,
            vector_field=milvus_settings.vector_field,
            metadata_field=milvus_settings.metadata_field,
            metric_type=milvus_settings.metric_type,
            expected_dimension=milvus_settings.dimension,
        )
        graph_backend = Neo4jGraphBackend(
            neo4j_driver, database=neo4j_settings.database
        )
        result = hybrid_search(
            args.query,
            args.disease_name,
            vector_backend=vector_backend,
            graph_backend=graph_backend,
            top_k=args.top_k,
            vector_top_k=args.vector_top_k,
            graph_top_k=args.graph_top_k,
            metadata_filter=_metadata_filter_from_args(args),
            relations=args.relation,
            rrf_k=args.rrf_k,
            vector_weight=args.vector_weight,
            graph_weight=args.graph_weight,
            allow_partial=args.allow_partial,
        )
        if args.reranker_model_path is not None:
            reranker = LocalCrossEncoderReranker(
                args.reranker_model_path,
                device=args.reranker_device or args.device,
                batch_size=args.reranker_batch_size,
                max_length=args.reranker_max_length,
            )
            result["post_retrieval"] = prepare_evidence_context(
                args.query,
                result["evidence"],
                reranker_backend=reranker,
                rerank_top_k=args.rerank_top_k,
                rerank_min_score=args.rerank_min_score,
                context_max_tokens=args.context_max_tokens,
                context_max_items=args.context_max_items,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        close = getattr(milvus_client, "close", None)
        if callable(close):
            close()
        close = getattr(neo4j_driver, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
