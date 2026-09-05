from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from src.agent.query_planner import (
    DEFAULT_DOCUMENTS_PATH,
    DeterministicMedicalQueryPlanner,
    DiseaseCatalog,
    QueryPlan,
)
from src.agent.tool_schemas import (
    MEDICAL_TOOL_SCHEMAS,
    ValidatedToolCall,
    validate_tool_call,
)
from src.retrieval.embedding import create_embedding_model
from src.retrieval.graph_retriever import (
    GraphSearchBackend,
    Neo4jGraphBackend,
    graph_search,
)
from src.retrieval.hybrid_retriever import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DIMENSION,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    fuse_evidence,
    hybrid_search,
    hybrid_search_async,
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
from src.retrieval.reranker import LocalCrossEncoderReranker, RerankBackend
from src.retrieval.retrieval_pipeline import prepare_evidence_context
from src.retrieval.vector_retriever import (
    MilvusVectorBackend,
    VectorSearchBackend,
    vector_search,
)


class AgentExecutionError(RuntimeError):
    """Raised when a validated plan lacks a required runtime backend."""


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


class GraphRAGAgent:
    """Plan one medical query and execute the existing evidence pipeline.

    The class deliberately ends at ``evidence_ready``. Answer generation belongs
    to the surrounding API service, so retrieval stays independently testable.
    """

    def __init__(
        self,
        planner: DeterministicMedicalQueryPlanner,
        *,
        vector_backend: VectorSearchBackend | None = None,
        graph_backend: GraphSearchBackend | None = None,
        reranker_backend: RerankBackend | None = None,
        rerank_top_k: int | None = 8,
        rerank_min_score: float | None = None,
        context_max_tokens: int = 2000,
        context_max_items: int = 6,
        rrf_k: int = DEFAULT_RRF_K,
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
    ) -> None:
        self.planner = planner
        self.vector_backend = vector_backend
        self.graph_backend = graph_backend
        self.reranker_backend = reranker_backend
        self.rerank_top_k = rerank_top_k
        self.rerank_min_score = rerank_min_score
        self.context_max_tokens = context_max_tokens
        self.context_max_items = context_max_items
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def run(
        self,
        query: str,
        *,
        disease_name: str | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        allow_partial: bool | None = None,
    ) -> dict[str, object]:
        return self.run_plan(
            self.planner.plan(
                query,
                disease_name=disease_name,
                metadata_filter=metadata_filter,
                top_k=top_k,
                vector_top_k=vector_top_k,
                graph_top_k=graph_top_k,
                allow_partial=allow_partial,
            )
        )

    async def arun(
        self,
        query: str,
        *,
        disease_name: str | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        allow_partial: bool | None = None,
    ) -> dict[str, object]:
        plan = self.planner.plan(
            query,
            disease_name=disease_name,
            metadata_filter=metadata_filter,
            top_k=top_k,
            vector_top_k=vector_top_k,
            graph_top_k=graph_top_k,
            allow_partial=allow_partial,
        )
        return await self.arun_plan(plan)

    def run_plan(self, plan: QueryPlan) -> dict[str, object]:
        if plan.status != "ready":
            return {
                "status": plan.status,
                "request_id": f"req_{uuid.uuid4().hex}",
                "query_plan": plan.to_dict(),
                "tool_calls": [],
                "retrieval": None,
                "post_retrieval": None,
            }
        if not isinstance(plan.tool_call, Mapping):
            raise AgentExecutionError("A ready query plan must contain one tool call")
        name = plan.tool_call.get("name")
        arguments = plan.tool_call.get("arguments")
        result = self.execute_tool_call(name, arguments)
        result["query_plan"] = plan.to_dict()
        return result

    async def arun_plan(self, plan: QueryPlan) -> dict[str, object]:
        if plan.status != "ready":
            return {
                "status": plan.status,
                "request_id": f"req_{uuid.uuid4().hex}",
                "query_plan": plan.to_dict(),
                "tool_calls": [],
                "retrieval": None,
                "post_retrieval": None,
            }
        if not isinstance(plan.tool_call, Mapping):
            raise AgentExecutionError("A ready query plan must contain one tool call")
        result = await self.execute_tool_call_async(
            plan.tool_call.get("name"), plan.tool_call.get("arguments")
        )
        result["query_plan"] = plan.to_dict()
        return result

    def execute_tool_call(self, name: object, arguments: object) -> dict[str, object]:
        """Validate and execute one planner- or LLM-produced Function Call."""

        validated = validate_tool_call(name, arguments)
        started = time.perf_counter()
        retrieval = self._retrieve(validated)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

        return self._prepare_result(validated, retrieval, elapsed_ms)

    async def execute_tool_call_async(
        self, name: object, arguments: object
    ) -> dict[str, object]:
        validated = validate_tool_call(name, arguments)
        started = time.perf_counter()
        retrieval = await self._retrieve_async(validated)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        return await asyncio.to_thread(
            self._prepare_result, validated, retrieval, elapsed_ms
        )

    def _prepare_result(
        self,
        validated: ValidatedToolCall,
        retrieval: dict[str, object],
        elapsed_ms: float,
    ) -> dict[str, object]:
        if self.reranker_backend is None:
            raise AgentExecutionError(
                "A reranker backend is required before evidence can be prepared"
            )
        quotas = {
            "vector_search": {"vector": 1, "graph": 0},
            "graph_search": {"vector": 0, "graph": 1},
            "hybrid_search": {"vector": 1, "graph": 1},
        }[validated.name]
        post_retrieval = prepare_evidence_context(
            str(validated.arguments["query"]),
            retrieval["evidence"],
            reranker_backend=self.reranker_backend,
            rerank_top_k=self.rerank_top_k,
            rerank_min_score=self.rerank_min_score,
            context_max_tokens=self.context_max_tokens,
            context_max_items=self.context_max_items,
            min_type_counts=quotas,
        )
        return {
            "status": "evidence_ready",
            "request_id": f"req_{uuid.uuid4().hex}",
            "query_plan": None,
            "tool_calls": [
                {
                    "name": validated.name,
                    "arguments": validated.arguments,
                    "status": "completed",
                    "elapsed_ms": elapsed_ms,
                    "retrieval_stats": retrieval["stats"],
                }
            ],
            "retrieval": retrieval,
            "post_retrieval": post_retrieval,
        }

    async def _retrieve_async(self, call: ValidatedToolCall) -> dict[str, object]:
        if call.name != "hybrid_search":
            return await asyncio.to_thread(self._retrieve, call)
        arguments = call.arguments
        if self.vector_backend is None or self.graph_backend is None:
            raise AgentExecutionError(
                "hybrid_search requires both vector and graph backends"
            )
        configured_filter = arguments.get("metadata_filter")
        runtime_filter = (
            {
                key: value
                for key, value in configured_filter.items()
                if value is not None
            }
            if isinstance(configured_filter, Mapping)
            else {}
        )
        return await hybrid_search_async(
            str(arguments["query"]),
            str(arguments["disease_name"]),
            vector_backend=self.vector_backend,
            graph_backend=self.graph_backend,
            top_k=int(arguments["top_k"]),
            vector_top_k=int(arguments["vector_top_k"]),
            graph_top_k=int(arguments["graph_top_k"]),
            metadata_filter=runtime_filter,
            relations=arguments["relations"],
            rrf_k=self.rrf_k,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight,
            allow_partial=bool(arguments["allow_partial"]),
        )

    def _retrieve(self, call: ValidatedToolCall) -> dict[str, object]:
        arguments = call.arguments
        configured_filter = arguments.get("metadata_filter")
        runtime_filter = (
            {
                key: value
                for key, value in configured_filter.items()
                if value is not None
            }
            if isinstance(configured_filter, Mapping)
            else {}
        )
        if call.name == "vector_search":
            if self.vector_backend is None:
                raise AgentExecutionError("vector_search requires a vector backend")
            hits = vector_search(
                str(arguments["query"]),
                backend=self.vector_backend,
                top_k=int(arguments["top_k"]),
                metadata_filter=runtime_filter,
            )
            evidence, stats = fuse_evidence(
                hits,
                [],
                top_k=int(arguments["top_k"]),
                rrf_k=self.rrf_k,
                vector_weight=self.vector_weight,
                graph_weight=self.graph_weight,
            )
            diseases = runtime_filter["disease_name"]
            return {
                "query": arguments["query"],
                "disease_name": diseases[0],
                "metadata_filter": runtime_filter,
                "relations": [],
                "strategy": {
                    "name": "metadata_filtered_vector_search",
                    "note": "fusion_score only normalizes evidence for downstream stages",
                },
                "stats": stats,
                "errors": {},
                "evidence": evidence,
            }

        if call.name == "graph_search":
            if self.graph_backend is None:
                raise AgentExecutionError("graph_search requires a graph backend")
            hits = graph_search(
                str(arguments["disease_name"]),
                backend=self.graph_backend,
                relations=arguments["relations"],
                limit=int(arguments["limit"]),
            )
            evidence, stats = fuse_evidence(
                [],
                hits,
                top_k=int(arguments["limit"]),
                rrf_k=self.rrf_k,
                vector_weight=self.vector_weight,
                graph_weight=self.graph_weight,
            )
            return {
                "query": arguments["query"],
                "disease_name": arguments["disease_name"],
                "metadata_filter": {},
                "relations": arguments["relations"],
                "strategy": {
                    "name": "allowlisted_graph_expansion",
                    "note": "fusion_score only normalizes evidence for downstream stages",
                },
                "stats": stats,
                "errors": {},
                "evidence": evidence,
            }

        if self.vector_backend is None or self.graph_backend is None:
            raise AgentExecutionError(
                "hybrid_search requires both vector and graph backends"
            )
        return hybrid_search(
            str(arguments["query"]),
            str(arguments["disease_name"]),
            vector_backend=self.vector_backend,
            graph_backend=self.graph_backend,
            top_k=int(arguments["top_k"]),
            vector_top_k=int(arguments["vector_top_k"]),
            graph_top_k=int(arguments["graph_top_k"]),
            metadata_filter=runtime_filter,
            relations=arguments["relations"],
            rrf_k=self.rrf_k,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight,
            allow_partial=bool(arguments["allow_partial"]),
        )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a natural-language medical query, execute GraphRAG retrieval, "
            "and return evidence with citations"
        )
    )
    parser.add_argument("query", nargs="?")
    parser.add_argument("--disease-name", help="Optional exact entity override")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--show-tool-schemas", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--vector-top-k", type=int, default=6)
    parser.add_argument("--graph-top-k", type=int, default=6)
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
    parser.add_argument(
        "--model-path", type=Path, default=_optional_path(os.getenv("BGE_MODEL_PATH"))
    )
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


def main() -> None:
    configure_stdout()
    load_project_environment()
    args = parse_args()
    if args.show_tool_schemas:
        print(json.dumps(MEDICAL_TOOL_SCHEMAS, ensure_ascii=False, indent=2))
        return
    if not args.query:
        raise SystemExit("query is required unless --show-tool-schemas is used")

    catalog = DiseaseCatalog.from_documents(args.documents)
    planner = DeterministicMedicalQueryPlanner(
        catalog,
        top_k=args.top_k,
        vector_top_k=args.vector_top_k,
        graph_top_k=args.graph_top_k,
        allow_partial=args.allow_partial,
    )
    plan = planner.plan(args.query, disease_name=args.disease_name)
    if args.plan_only or plan.status != "ready":
        print(
            json.dumps(
                {
                    "status": plan.status,
                    "query_plan": plan.to_dict(),
                    "tool_calls": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.model_path is None:
        raise SystemExit(
            "A local BGE model is required. Set BGE_MODEL_PATH or pass --model-path."
        )
    if args.reranker_model_path is None:
        raise SystemExit(
            "A local reranker is required. Set BGE_RERANKER_MODEL_PATH or pass "
            "--reranker-model-path."
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
    reranker = LocalCrossEncoderReranker(
        args.reranker_model_path,
        device=args.reranker_device or args.device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )

    tool_name = str(plan.selected_tool)
    needs_vector = tool_name in {"vector_search", "hybrid_search"}
    needs_graph = tool_name in {"graph_search", "hybrid_search"}
    milvus_client: object | None = None
    neo4j_driver: object | None = None
    vector_backend: VectorSearchBackend | None = None
    graph_backend: GraphSearchBackend | None = None
    try:
        if needs_vector:
            milvus_settings = MilvusSettings(
                uri=args.milvus_uri,
                token=args.milvus_token,
                collection_name=args.collection,
                dimension=args.dimension,
                load_timeout_seconds=args.milvus_load_timeout_seconds,
            )
            milvus_client = create_milvus_client(milvus_settings)
            store = MedicalMilvusStore(milvus_client, milvus_settings)
            if not store.has_collection():
                raise MilvusCollectionError(
                    f"Milvus collection does not exist: {args.collection}"
                )
            store.validate_existing_collection()
            store.load_collection()
            vector_backend = MilvusVectorBackend(
                milvus_client,
                args.collection,
                embedding_model.embed_query,
                vector_field=milvus_settings.vector_field,
                metadata_field=milvus_settings.metadata_field,
                metric_type=milvus_settings.metric_type,
                expected_dimension=args.dimension,
            )
        if needs_graph:
            neo4j_settings = Neo4jSettings(
                uri=args.neo4j_uri,
                username=args.neo4j_username,
                password=args.neo4j_password,
                database=args.neo4j_database,
            )
            neo4j_driver = create_neo4j_driver(neo4j_settings)
            graph_backend = Neo4jGraphBackend(
                neo4j_driver, database=neo4j_settings.database
            )

        agent = GraphRAGAgent(
            planner,
            vector_backend=vector_backend,
            graph_backend=graph_backend,
            reranker_backend=reranker,
            rerank_top_k=args.rerank_top_k,
            rerank_min_score=args.rerank_min_score,
            context_max_tokens=args.context_max_tokens,
            context_max_items=args.context_max_items,
        )
        print(json.dumps(agent.run_plan(plan), ensure_ascii=False, indent=2))
    finally:
        for resource in (milvus_client, neo4j_driver):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main()
