from __future__ import annotations

import os
from pathlib import Path

from src.agent.orchestrator import GraphRAGAgent
from src.agent.query_planner import DiseaseCatalog, DeterministicMedicalQueryPlanner
from src.api.service import MedicalChatService
from src.generation.answer_generator import (
    ExtractiveCitationAnswerGenerator,
    GroundedLLMAnswerGenerator,
)
from src.llm.client import OpenAICompatibleChatClient
from src.retrieval.embedding import create_embedding_model
from src.retrieval.graph_retriever import Neo4jGraphBackend
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
from src.retrieval.vector_retriever import MilvusVectorBackend


def _required_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"Configured local model directory does not exist: {path}")
    return path


class LocalMedicalRuntime:
    """Load heavyweight local resources once for the FastAPI application lifespan."""

    def __init__(self) -> None:
        self.milvus_client: object | None = None
        self.neo4j_driver: object | None = None
        self.service: MedicalChatService | None = None

    def start(self) -> MedicalChatService:
        load_project_environment()
        dimension = int(os.getenv("MILVUS_DIMENSION", "768"))
        model_path = _required_path("BGE_MODEL_PATH")
        reranker_path = _required_path("BGE_RERANKER_MODEL_PATH")
        device = os.getenv("BGE_DEVICE")

        embedding_model = create_embedding_model(
            "bge",
            dimension=dimension,
            model_path=model_path,
            device=device,
            batch_size=int(os.getenv("BGE_BATCH_SIZE", "32")),
        )
        if embedding_model.dimension != dimension:
            raise RuntimeError(
                "BGE model dimension does not match MILVUS_DIMENSION: "
                f"model={embedding_model.dimension}, configured={dimension}"
            )
        reranker = LocalCrossEncoderReranker(
            reranker_path,
            device=os.getenv("BGE_RERANKER_DEVICE") or device,
            batch_size=int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8")),
            max_length=int(os.getenv("BGE_RERANKER_MAX_LENGTH", "512")),
        )

        milvus_settings = MilvusSettings(
            uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.getenv("MILVUS_TOKEN"),
            collection_name=os.getenv(
                "MILVUS_COLLECTION", "medical_chunks_bge_base_zh_v15"
            ),
            dimension=dimension,
            load_timeout_seconds=float(
                os.getenv(
                    "MILVUS_LOAD_TIMEOUT_SECONDS",
                    str(DEFAULT_LOAD_TIMEOUT_SECONDS),
                )
            ),
        )
        self.milvus_client = create_milvus_client(milvus_settings)
        store = MedicalMilvusStore(self.milvus_client, milvus_settings)
        if not store.has_collection():
            raise MilvusCollectionError(
                f"Milvus collection does not exist: {milvus_settings.collection_name}"
            )
        store.validate_existing_collection()
        store.load_collection()
        vector_backend = MilvusVectorBackend(
            self.milvus_client,
            milvus_settings.collection_name,
            embedding_model.embed_query,
            vector_field=milvus_settings.vector_field,
            metadata_field=milvus_settings.metadata_field,
            metric_type=milvus_settings.metric_type,
            expected_dimension=dimension,
        )

        neo4j_settings = Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
        self.neo4j_driver = create_neo4j_driver(neo4j_settings)
        graph_backend = Neo4jGraphBackend(
            self.neo4j_driver, database=neo4j_settings.database
        )
        planner = DeterministicMedicalQueryPlanner(DiseaseCatalog.from_documents())
        agent = GraphRAGAgent(
            planner,
            vector_backend=vector_backend,
            graph_backend=graph_backend,
            reranker_backend=reranker,
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "8")),
            context_max_tokens=int(os.getenv("CONTEXT_MAX_TOKENS", "2000")),
            context_max_items=int(os.getenv("CONTEXT_MAX_ITEMS", "6")),
        )
        extractive_generator = ExtractiveCitationAnswerGenerator()
        generator_backend = os.getenv(
            "ANSWER_GENERATOR_BACKEND", "extractive"
        ).strip().casefold()
        if generator_backend == "extractive":
            answer_generator = extractive_generator
        elif generator_backend in {"openai-compatible", "openai_compatible"}:
            base_url = os.getenv("GENERATION_LLM_BASE_URL") or os.getenv(
                "LLM_BASE_URL"
            )
            model = os.getenv("GENERATION_LLM_MODEL") or os.getenv("LLM_MODEL")
            if not base_url or not model:
                raise RuntimeError(
                    "LLM answer generation requires GENERATION_LLM_BASE_URL/"
                    "LLM_BASE_URL and GENERATION_LLM_MODEL/LLM_MODEL"
                )
            client = OpenAICompatibleChatClient(
                base_url=base_url,
                model=model,
                api_key=os.getenv("GENERATION_LLM_API_KEY")
                or os.getenv("LLM_API_KEY"),
                timeout_seconds=float(os.getenv("GENERATION_LLM_TIMEOUT", "90")),
            )
            allow_fallback = os.getenv(
                "GENERATION_LLM_FALLBACK", "true"
            ).strip().casefold() in {"1", "true", "yes", "on"}
            answer_generator = GroundedLLMAnswerGenerator(
                client,
                fallback=extractive_generator if allow_fallback else None,
            )
        else:
            raise RuntimeError(
                "ANSWER_GENERATOR_BACKEND must be extractive or openai-compatible"
            )
        self.service = MedicalChatService(
            agent,
            answer_generator,
            max_concurrent_requests=int(os.getenv("API_MAX_CONCURRENCY", "1")),
        )
        return self.service

    def close(self) -> None:
        for resource in (self.milvus_client, self.neo4j_driver):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        self.service = None
