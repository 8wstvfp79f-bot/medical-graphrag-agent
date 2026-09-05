from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol

from src.retrieval.embedding import (
    DEFAULT_CHUNKS_PATH,
    EmbeddingModel,
    create_embedding_model,
    embed_chunk_records,
    iter_chunk_records,
)
from src.retrieval.metadata_filter import (
    MetadataFilterInput,
    normalize_chunk_metadata,
)
from src.retrieval.vector_retriever import MilvusVectorBackend, vector_search


DEFAULT_COLLECTION_NAME = "medical_chunks"
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_VECTOR_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_COUNT_BATCH_SIZE = 1024
DEFAULT_LOAD_TIMEOUT_SECONDS = 60.0
SUPPORTED_METRICS = frozenset({"COSINE", "IP", "L2"})


class MilvusDependencyError(RuntimeError):
    """Raised when the optional pymilvus runtime is unavailable."""


class MilvusCollectionError(RuntimeError):
    """Raised when a collection is missing or incompatible with the application."""


class DataTypes(Protocol):
    VARCHAR: object
    JSON: object
    FLOAT_VECTOR: object


@dataclass(frozen=True)
class MilvusSettings:
    uri: str = DEFAULT_MILVUS_URI
    token: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    dimension: int = DEFAULT_VECTOR_DIMENSION
    metric_type: str = "COSINE"
    vector_field: str = "embedding"
    metadata_field: str = "metadata"
    text_max_length: int = 16_384
    load_timeout_seconds: float = DEFAULT_LOAD_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("Milvus uri must not be empty")
        if not self.collection_name.strip():
            raise ValueError("Milvus collection_name must not be empty")
        if self.dimension <= 0:
            raise ValueError("Milvus vector dimension must be greater than zero")
        metric_type = self.metric_type.upper()
        if metric_type not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported Milvus metric_type: {self.metric_type}. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_METRICS))}"
            )
        object.__setattr__(self, "metric_type", metric_type)
        if self.text_max_length <= 0:
            raise ValueError("text_max_length must be greater than zero")
        if self.load_timeout_seconds <= 0:
            raise ValueError("Milvus load_timeout_seconds must be greater than zero")


@dataclass(frozen=True)
class IngestionReport:
    collection_name: str
    collection_created: bool
    embedding_model: str
    embedding_dimension: int
    attempted_records: int
    batches: int
    logical_entity_count_before: int | None
    logical_entity_count_after: int | None
    storage_row_count_before: int | None
    storage_row_count_after: int | None


@dataclass(frozen=True)
class MilvusValidationReport:
    input_path: str
    embedding_model: str
    embedding_dimension: int
    validated_records: int
    batches: int
    max_text_bytes: int


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def create_milvus_client(settings: MilvusSettings) -> object:
    """Create a real pymilvus client without importing pymilvus at module import time."""
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise MilvusDependencyError(
            "pymilvus is not installed. Install project requirements in an allowed "
            "environment, then provide a running Milvus URI or a supported Milvus Lite path."
        ) from exc

    kwargs: dict[str, object] = {"uri": settings.uri}
    if settings.token:
        kwargs["token"] = settings.token
    return MilvusClient(**kwargs)


def load_milvus_data_types() -> DataTypes:
    try:
        from pymilvus import DataType
    except ImportError as exc:
        raise MilvusDependencyError(
            "pymilvus is required to create a Milvus collection schema"
        ) from exc
    return DataType


def _require_text(record: Mapping[str, object], field: str) -> str:
    value = str(record.get(field, "")).strip()
    if not value:
        raise ValueError(f"Milvus record must contain non-empty '{field}'")
    return value


def _check_varchar_length(value: str, field: str, max_length: int) -> None:
    byte_length = len(value.encode("utf-8"))
    if byte_length > max_length:
        raise ValueError(
            f"Milvus field '{field}' exceeds max UTF-8 length "
            f"{max_length}: {byte_length}"
        )


class MedicalMilvusStore:
    """Own the medical chunk collection lifecycle and bounded batch ingestion."""

    def __init__(
        self,
        client: object,
        settings: MilvusSettings,
        *,
        data_types: DataTypes | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self._data_types = data_types

    def _method(self, name: str):
        method = getattr(self.client, name, None)
        if not callable(method):
            raise TypeError(f"Milvus client must expose a callable '{name}' method")
        return method

    def has_collection(self) -> bool:
        return bool(
            self._method("has_collection")(
                collection_name=self.settings.collection_name
            )
        )

    def _existing_vector_dimension(self) -> int | None:
        describe = getattr(self.client, "describe_collection", None)
        if not callable(describe):
            return None
        description = describe(collection_name=self.settings.collection_name)
        if not isinstance(description, Mapping):
            return None
        fields = description.get("fields")
        if not isinstance(fields, Sequence):
            return None
        for field in fields:
            if not isinstance(field, Mapping):
                continue
            field_name = field.get("name") or field.get("field_name")
            if field_name != self.settings.vector_field:
                continue
            params = field.get("params") or field.get("type_params") or {}
            dimension = field.get("dim")
            if isinstance(params, Mapping):
                dimension = params.get("dim", dimension)
            if dimension is None:
                return None
            return int(dimension)
        return None

    def validate_existing_collection(self) -> None:
        existing_dimension = self._existing_vector_dimension()
        if (
            existing_dimension is not None
            and existing_dimension != self.settings.dimension
        ):
            raise MilvusCollectionError(
                "Existing Milvus vector dimension does not match the embedding model: "
                f"collection={existing_dimension}, configured={self.settings.dimension}"
            )

    def load_collection(self) -> None:
        load = getattr(self.client, "load_collection", None)
        if callable(load):
            load(
                collection_name=self.settings.collection_name,
                timeout=self.settings.load_timeout_seconds,
            )

    def ensure_collection(self, *, recreate: bool = False) -> bool:
        """Create the collection once; recreation only occurs behind an explicit flag."""
        exists = self.has_collection()
        if exists and recreate:
            self._method("drop_collection")(
                collection_name=self.settings.collection_name
            )
            exists = False

        if exists:
            self.validate_existing_collection()
            self.load_collection()
            return False

        data_types = self._data_types or load_milvus_data_types()
        schema = self._method("create_schema")(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=data_types.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="doc_id",
            datatype=data_types.VARCHAR,
            max_length=128,
        )
        schema.add_field(
            field_name="disease_name",
            datatype=data_types.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="text",
            datatype=data_types.VARCHAR,
            max_length=self.settings.text_max_length,
        )
        schema.add_field(
            field_name=self.settings.metadata_field,
            datatype=data_types.JSON,
        )
        schema.add_field(
            field_name="embedding_model",
            datatype=data_types.VARCHAR,
            max_length=256,
        )
        schema.add_field(
            field_name=self.settings.vector_field,
            datatype=data_types.FLOAT_VECTOR,
            dim=self.settings.dimension,
        )

        index_params = self._method("prepare_index_params")()
        index_params.add_index(
            field_name=self.settings.vector_field,
            index_name="medical_embedding_index",
            index_type="AUTOINDEX",
            metric_type=self.settings.metric_type,
        )
        self._method("create_collection")(
            collection_name=self.settings.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.load_collection()
        return True

    def build_entity(self, record: Mapping[str, object]) -> dict[str, object]:
        chunk_id = _require_text(record, "chunk_id")
        doc_id = _require_text(record, "doc_id")
        text = _require_text(record, "text")
        metadata_value = record.get("metadata")
        if not isinstance(metadata_value, Mapping):
            raise ValueError("Milvus record must contain an object 'metadata'")
        metadata = normalize_chunk_metadata(metadata_value)
        disease_name = str(metadata["disease_name"])
        source = str(metadata["source"])
        if not disease_name or not source:
            raise ValueError(
                "Milvus metadata must contain non-empty disease_name and source"
            )

        vector_value = record.get("embedding")
        if not isinstance(vector_value, Sequence) or isinstance(
            vector_value, (str, bytes)
        ):
            raise ValueError("Milvus record must contain a numeric embedding sequence")
        vector = [float(value) for value in vector_value]
        if len(vector) != self.settings.dimension:
            raise ValueError(
                "Milvus embedding dimension mismatch: "
                f"expected {self.settings.dimension}, got {len(vector)}"
            )

        embedding_metadata = record.get("embedding_metadata")
        embedding_model = "unknown"
        if isinstance(embedding_metadata, Mapping):
            embedding_model = str(embedding_metadata.get("model", "unknown"))

        _check_varchar_length(chunk_id, "chunk_id", 128)
        _check_varchar_length(doc_id, "doc_id", 128)
        _check_varchar_length(disease_name, "disease_name", 512)
        _check_varchar_length(text, "text", self.settings.text_max_length)
        _check_varchar_length(embedding_model, "embedding_model", 256)
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "disease_name": disease_name,
            "text": text,
            self.settings.metadata_field: metadata,
            "embedding_model": embedding_model,
            self.settings.vector_field: vector,
        }

    def storage_row_count(self) -> int | None:
        """Return Milvus' physical row-version statistic.

        Milvus upsert is implemented as a logical replacement. Until background
        compaction removes superseded versions, ``row_count`` can grow even though
        queries still expose only one current entity for each primary key.
        """
        stats_method = getattr(self.client, "get_collection_stats", None)
        if not callable(stats_method) or not self.has_collection():
            return None
        stats = stats_method(collection_name=self.settings.collection_name)
        if not isinstance(stats, Mapping):
            return None
        value = stats.get("row_count", stats.get("rowCount"))
        if value is None:
            return None
        return int(value)

    def logical_entity_count(
        self,
        *,
        batch_size: int = DEFAULT_COUNT_BATCH_SIZE,
    ) -> int | None:
        """Count current logical entities by streaming and deduplicating primary keys."""
        if batch_size <= 0:
            raise ValueError("count batch_size must be greater than zero")
        if not self.has_collection():
            return None

        self.load_collection()
        iterator = self._method("query_iterator")(
            collection_name=self.settings.collection_name,
            filter="",
            output_fields=["chunk_id"],
            batch_size=batch_size,
            consistency_level="Strong",
        )
        next_page = getattr(iterator, "next", None)
        if not callable(next_page):
            raise TypeError("Milvus query iterator must expose a callable 'next' method")

        chunk_ids: set[str] = set()
        try:
            while True:
                page = next_page()
                if not page:
                    break
                if not isinstance(page, Sequence) or isinstance(page, (str, bytes)):
                    raise TypeError("Milvus query iterator pages must be sequences")
                for entity in page:
                    if not isinstance(entity, Mapping):
                        raise TypeError("Milvus query iterator entities must be mappings")
                    chunk_id = str(entity.get("chunk_id", "")).strip()
                    if not chunk_id:
                        raise ValueError("Milvus entity is missing primary key 'chunk_id'")
                    chunk_ids.add(chunk_id)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return len(chunk_ids)

    def entity_count(self) -> int | None:
        """Backward-compatible alias for the logical unique-entity count."""
        return self.logical_entity_count()

    def count_summary(self) -> dict[str, int | None]:
        return {
            "logical_entity_count": self.logical_entity_count(),
            "storage_row_count": self.storage_row_count(),
        }

    def _upsert_batch(self, entities: list[dict[str, object]]) -> None:
        if not entities:
            return
        self._method("upsert")(
            collection_name=self.settings.collection_name,
            data=entities,
        )

    def ingest_chunks(
        self,
        chunks_path: Path,
        embedding_model: EmbeddingModel,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        limit: int | None = None,
        recreate: bool = False,
    ) -> IngestionReport:
        if embedding_model.dimension != self.settings.dimension:
            raise ValueError(
                "Embedding model dimension must match Milvus collection dimension: "
                f"model={embedding_model.dimension}, collection={self.settings.dimension}"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero when provided")

        created = self.ensure_collection(recreate=recreate)
        logical_count_before = self.logical_entity_count()
        storage_count_before = self.storage_row_count()
        records = iter_chunk_records(chunks_path)
        if limit is not None:
            records = islice(records, limit)

        attempted = 0
        batches = 0
        entity_batch: list[dict[str, object]] = []
        for embedded_record in embed_chunk_records(
            records,
            embedding_model,
            batch_size=batch_size,
        ):
            entity_batch.append(self.build_entity(embedded_record))
            attempted += 1
            if len(entity_batch) >= batch_size:
                self._upsert_batch(entity_batch)
                batches += 1
                entity_batch = []
        if entity_batch:
            self._upsert_batch(entity_batch)
            batches += 1

        flush = getattr(self.client, "flush", None)
        if callable(flush):
            flush(collection_name=self.settings.collection_name)
        logical_count_after = self.logical_entity_count()
        storage_count_after = self.storage_row_count()
        return IngestionReport(
            collection_name=self.settings.collection_name,
            collection_created=created,
            embedding_model=embedding_model.model_name,
            embedding_dimension=embedding_model.dimension,
            attempted_records=attempted,
            batches=batches,
            logical_entity_count_before=logical_count_before,
            logical_entity_count_after=logical_count_after,
            storage_row_count_before=storage_count_before,
            storage_row_count_after=storage_count_after,
        )

    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        *,
        top_k: int = 5,
        metadata_filter: MetadataFilterInput | None = None,
    ) -> list[dict[str, object]]:
        if not self.has_collection():
            raise MilvusCollectionError(
                f"Milvus collection does not exist: {self.settings.collection_name}"
            )
        if embedding_model.dimension != self.settings.dimension:
            raise ValueError(
                "Query embedding dimension must match Milvus collection dimension"
            )
        self.load_collection()
        backend = MilvusVectorBackend(
            self.client,
            self.settings.collection_name,
            embedding_model.embed_query,
            vector_field=self.settings.vector_field,
            metadata_field=self.settings.metadata_field,
            metric_type=self.settings.metric_type,
            expected_dimension=self.settings.dimension,
        )
        return vector_search(
            query,
            backend=backend,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )


def validate_chunks_for_milvus(
    chunks_path: Path,
    embedding_model: EmbeddingModel,
    settings: MilvusSettings,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> MilvusValidationReport:
    """Validate the real ingestion payload without connecting to Milvus."""
    if embedding_model.dimension != settings.dimension:
        raise ValueError(
            "Embedding model dimension must match Milvus collection dimension: "
            f"model={embedding_model.dimension}, collection={settings.dimension}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when provided")

    records = iter_chunk_records(chunks_path)
    if limit is not None:
        records = islice(records, limit)
    validator = MedicalMilvusStore(object(), settings)
    count = 0
    max_text_bytes = 0
    for embedded_record in embed_chunk_records(
        records,
        embedding_model,
        batch_size=batch_size,
    ):
        entity = validator.build_entity(embedded_record)
        max_text_bytes = max(
            max_text_bytes,
            len(str(entity["text"]).encode("utf-8")),
        )
        count += 1
    batches = (count + batch_size - 1) // batch_size if count else 0
    return MilvusValidationReport(
        input_path=str(chunks_path.resolve()),
        embedding_model=embedding_model.model_name,
        embedding_dimension=embedding_model.dimension,
        validated_records=count,
        batches=batches,
        max_text_bytes=max_text_bytes,
    )


def _settings_from_args(args: argparse.Namespace) -> MilvusSettings:
    return MilvusSettings(
        uri=args.uri,
        token=args.token,
        collection_name=args.collection,
        dimension=args.dimension,
        metric_type=args.metric,
        load_timeout_seconds=args.load_timeout_seconds,
    )


def _embedding_from_args(args: argparse.Namespace) -> EmbeddingModel:
    return create_embedding_model(
        args.backend,
        dimension=args.dimension,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )


def _metadata_filter_from_args(
    args: argparse.Namespace,
) -> dict[str, str | Sequence[str]]:
    metadata_filter: dict[str, str | Sequence[str]] = {}
    if args.disease_name:
        metadata_filter["disease_name"] = args.disease_name
    if args.category:
        metadata_filter["category"] = args.category
    if args.department:
        metadata_filter["department"] = args.department
    if args.source:
        metadata_filter["source"] = args.source
    return metadata_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the medical Milvus collection")
    parser.add_argument(
        "--uri",
        default=os.getenv("MILVUS_URI", DEFAULT_MILVUS_URI),
        help="Milvus server URI or a supported Milvus Lite database path",
    )
    parser.add_argument("--token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--dimension", type=int, default=DEFAULT_VECTOR_DIMENSION)
    parser.add_argument("--metric", choices=sorted(SUPPORTED_METRICS), default="COSINE")
    parser.add_argument(
        "--load-timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "MILVUS_LOAD_TIMEOUT_SECONDS", str(DEFAULT_LOAD_TIMEOUT_SECONDS)
            )
        ),
        help="Fail instead of waiting forever when a collection cannot finish loading",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate all ingestion records without connecting to Milvus",
    )
    validate_parser.add_argument("--input", type=Path, default=DEFAULT_CHUNKS_PATH)
    validate_parser.add_argument("--backend", choices=("hash", "bge"), default="hash")
    validate_parser.add_argument("--model-path", type=Path)
    validate_parser.add_argument("--device")
    validate_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    validate_parser.add_argument("--limit", type=int)

    init_parser = subparsers.add_parser("init", help="Create and load the collection")
    init_parser.add_argument("--recreate", action="store_true")

    ingest_parser = subparsers.add_parser("ingest", help="Embed and upsert chunks")
    ingest_parser.add_argument("--input", type=Path, default=DEFAULT_CHUNKS_PATH)
    ingest_parser.add_argument("--backend", choices=("hash", "bge"), default="hash")
    ingest_parser.add_argument("--model-path", type=Path)
    ingest_parser.add_argument("--device")
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ingest_parser.add_argument("--limit", type=int)
    ingest_parser.add_argument("--recreate", action="store_true")

    subparsers.add_parser("count", help="Print the current collection entity count")

    search_parser = subparsers.add_parser("search", help="Run metadata-aware search")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--backend", choices=("hash", "bge"), default="hash")
    search_parser.add_argument("--model-path", type=Path)
    search_parser.add_argument("--device")
    search_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    search_parser.add_argument("--disease-name")
    search_parser.add_argument("--category", action="append")
    search_parser.add_argument("--department", action="append")
    search_parser.add_argument("--source")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    settings = _settings_from_args(args)
    if args.command == "validate":
        model = _embedding_from_args(args)
        result = asdict(
            validate_chunks_for_milvus(
                args.input,
                model,
                settings,
                batch_size=args.batch_size,
                limit=args.limit,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    try:
        client = create_milvus_client(settings)
    except MilvusDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    store = MedicalMilvusStore(client, settings)
    try:
        if args.command == "init":
            created = store.ensure_collection(recreate=args.recreate)
            result = {
                "collection": settings.collection_name,
                "created": created,
                "dimension": settings.dimension,
                "metric_type": settings.metric_type,
                **store.count_summary(),
            }
        elif args.command == "ingest":
            model = _embedding_from_args(args)
            result = asdict(
                store.ingest_chunks(
                    args.input,
                    model,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    recreate=args.recreate,
                )
            )
        elif args.command == "count":
            if not store.has_collection():
                raise MilvusCollectionError(
                    f"Milvus collection does not exist: {settings.collection_name}"
                )
            result = {
                "collection": settings.collection_name,
                **store.count_summary(),
                "storage_row_count_note": (
                    "May include superseded row versions created by upsert until "
                    "Milvus compaction removes them."
                ),
            }
        else:
            model = _embedding_from_args(args)
            result = {
                "query": args.query,
                "top_k": args.top_k,
                "metadata_filter": _metadata_filter_from_args(args),
                "hits": store.search(
                    args.query,
                    model,
                    top_k=args.top_k,
                    metadata_filter=_metadata_filter_from_args(args),
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
