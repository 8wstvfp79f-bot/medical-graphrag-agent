from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("\u3000", " ").split())


def normalize_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(normalize_items(item))
        return list(dict.fromkeys(items))

    text = clean_text(value)
    return [text] if text else []


def iter_documents(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(document, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            yield document


def build_chunk_metadata(document: dict) -> dict:
    document_metadata = document.get("metadata")
    if not isinstance(document_metadata, dict):
        document_metadata = {}

    disease_name = clean_text(
        document.get("disease_name") or document_metadata.get("disease_name")
    )
    source = clean_text(document.get("source") or document_metadata.get("source"))
    if not disease_name:
        raise ValueError(f"Document {document.get('doc_id', '<unknown>')} has no disease_name")
    if not source:
        raise ValueError(f"Document {document.get('doc_id', '<unknown>')} has no source")

    return {
        "disease_name": disease_name,
        "category": normalize_items(document_metadata.get("category")),
        "department": normalize_items(document_metadata.get("department")),
        "source": source,
    }


def split_text(text: str, max_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in the range [0, max_chars)")

    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + max_chars, len(normalized))
        end = hard_end
        if hard_end < len(normalized):
            newline = normalized.rfind("\n", start, hard_end)
            sentence = max(
                normalized.rfind("。", start, hard_end),
                normalized.rfind("；", start, hard_end),
            )
            boundary = max(newline, sentence)
            if boundary > start + max_chars // 2:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def build_chunks(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    max_chars: int = 800,
    overlap_chars: int = 100,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Documents file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for document in iter_documents(input_path):
            doc_id = clean_text(document.get("doc_id"))
            if not doc_id:
                raise ValueError("Every document must contain doc_id")

            metadata = build_chunk_metadata(document)
            chunks = split_text(
                clean_text(document.get("text")),
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            for chunk_index, chunk_text in enumerate(chunks, start=1):
                chunk_count += 1
                row = {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index:03d}",
                    "doc_id": doc_id,
                    "disease_name": metadata["disease_name"],
                    "text": chunk_text,
                    "metadata": metadata,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    return chunk_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata-rich RAG chunks")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    count = build_chunks(
        args.input,
        args.output,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"Generated {count} chunks: {args.output}")


if __name__ == "__main__":
    main()
