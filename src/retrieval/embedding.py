from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+", re.IGNORECASE)
DEFAULT_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class EmbeddingModel(Protocol):
    """Minimal interface shared by offline and production embedding backends."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _clean_text(text: object) -> str:
    value = text if isinstance(text, str) else str(text or "")
    return " ".join(value.replace("\u3000", " ").split())


def _validate_texts(texts: Sequence[str]) -> list[str]:
    cleaned = [_clean_text(text) for text in texts]
    empty_indexes = [index for index, text in enumerate(cleaned) if not text]
    if empty_indexes:
        raise ValueError(f"Embedding input contains empty text at indexes: {empty_indexes}")
    return cleaned


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for segment in TOKEN_PATTERN.findall(text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens or [text.casefold()]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@dataclass(frozen=True)
class HashEmbeddingModel:
    """Deterministic offline embedding for development and tests.

    This backend is not a semantic model and must not be used for final retrieval
    quality claims. It gives the pipeline stable vectors without downloading a model.
    """

    dimension: int = 384
    model_name: str = "hash-embedding-v1"

    def __post_init__(self) -> None:
        if self.dimension < 32:
            raise ValueError("Hash embedding dimension must be at least 32")

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], byteorder="big") % self.dimension
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = _validate_texts(texts)
        return [self._embed_one(text) for text in cleaned]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class LocalSentenceTransformerEmbeddingModel:
    """Sentence-Transformers adapter that only accepts an existing local model path."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str | None = None,
    ) -> None:
        resolved_path = model_path.expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(
                f"Local embedding model directory not found: {resolved_path}"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local BGE embedding requires the optional sentence-transformers package"
            ) from exc

        self._model = SentenceTransformer(str(resolved_path), device=device)
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if not callable(get_dimension):
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if not dimension:
            raise ValueError("Unable to determine the local embedding model dimension")
        self._dimension = int(dimension)
        self._model_name = resolved_path.name
        self._batch_size = batch_size
        self._query_instruction = (query_instruction or "").strip()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = _validate_texts(texts)
        if not cleaned:
            return []
        embeddings = self._model.encode(
            cleaned,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        cleaned_query = _validate_texts([query])[0]
        if self._query_instruction:
            cleaned_query = f"{self._query_instruction}{cleaned_query}"
        return self.embed_texts([cleaned_query])[0]


def create_embedding_model(
    backend: str = "hash",
    *,
    dimension: int = 384,
    model_path: Path | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> EmbeddingModel:
    normalized_backend = backend.strip().casefold()
    if normalized_backend == "hash":
        return HashEmbeddingModel(dimension=dimension)
    if normalized_backend in {"sentence-transformers", "sentence_transformers", "bge"}:
        if model_path is None:
            raise ValueError("model_path is required for the local sentence-transformers backend")
        return LocalSentenceTransformerEmbeddingModel(
            model_path,
            device=device,
            batch_size=batch_size,
            query_instruction=(
                DEFAULT_BGE_QUERY_INSTRUCTION if normalized_backend == "bge" else None
            ),
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")


def iter_chunk_records(path: Path) -> Iterator[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield record


def _embed_record_batch(
    records: Sequence[Mapping[str, object]],
    model: EmbeddingModel,
) -> list[dict[str, object]]:
    texts = [_clean_text(record.get("text")) for record in records]
    vectors = model.embed_texts(texts)
    if len(vectors) != len(records):
        raise ValueError("Embedding backend returned a different number of vectors")

    embedded_records: list[dict[str, object]] = []
    for record, vector in zip(records, vectors):
        if len(vector) != model.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {model.dimension}, got {len(vector)}"
            )
        embedded = dict(record)
        embedded["embedding"] = vector
        embedded["embedding_metadata"] = {
            "model": model.model_name,
            "dimension": model.dimension,
            "normalized": True,
        }
        embedded_records.append(embedded)
    return embedded_records


def embed_chunk_records(
    records: Iterable[Mapping[str, object]],
    model: EmbeddingModel,
    *,
    batch_size: int = 64,
) -> Iterator[dict[str, object]]:
    """Embed chunks in bounded batches for the later Milvus ingestion phase."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    batch: list[Mapping[str, object]] = []
    for record in records:
        chunk_id = _clean_text(record.get("chunk_id"))
        if not chunk_id:
            raise ValueError("Every chunk record must contain chunk_id")
        batch.append(record)
        if len(batch) >= batch_size:
            yield from _embed_record_batch(batch, model)
            batch = []
    if batch:
        yield from _embed_record_batch(batch, model)


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test chunk embeddings offline")
    parser.add_argument("--input", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--backend", choices=("hash", "sentence-transformers", "bge"), default="hash")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    model = create_embedding_model(
        args.backend,
        dimension=args.dimension,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    records = list(islice(iter_chunk_records(args.input), args.sample_size))
    if not records:
        raise ValueError(f"No chunks found in: {args.input}")
    embedded = list(embed_chunk_records(records, model, batch_size=args.batch_size))
    summary = {
        "backend": args.backend,
        "model": model.model_name,
        "dimension": model.dimension,
        "embedded_chunks": len(embedded),
        "chunk_ids": [record["chunk_id"] for record in embedded],
        "first_vector_norm": round(vector_norm(embedded[0]["embedding"]), 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
