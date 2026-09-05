from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
DEFAULT_TRIPLES_PATH = PROJECT_ROOT / "data" / "processed" / "triples.csv"
DEFAULT_CANDIDATE_TRIPLES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "candidate_triples.csv"
)
CORE_RELATIONS = (
    "has_symptom",
    "diagnosed_by",
    "treated_by",
    "belongs_to",
    "has_complication",
)


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def count_documents(path: Path) -> tuple[int, set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {path}")

    count = 0
    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            count += 1
            doc_id = str(document.get("doc_id", "")).strip()
            if doc_id:
                doc_ids.add(doc_id)
    return count, doc_ids


def calculate_stats(
    documents_path: Path,
    triples_path: Path,
    additional_triples_paths: Sequence[Path] = (),
) -> dict[str, object]:
    triples_paths = (triples_path, *additional_triples_paths)
    for path in triples_paths:
        if not path.exists():
            raise FileNotFoundError(f"Triples file not found: {path}")

    document_count, doc_ids = count_documents(documents_path)
    relation_distribution: Counter[str] = Counter()
    core_entities: set[str] = set()
    core_diseases: set[str] = set()
    all_entities: set[str] = set()
    total_triples = 0
    core_triples = 0
    candidate_triples = 0

    for path in triples_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            expected_fields = {"head", "relation", "tail", "source"}
            if not reader.fieldnames or not expected_fields.issubset(reader.fieldnames):
                raise ValueError(
                    "Triples CSV must contain columns: "
                    f"{', '.join(sorted(expected_fields))}"
                )

            for row in reader:
                head = (row.get("head") or "").strip()
                relation = (row.get("relation") or "").strip()
                tail = (row.get("tail") or "").strip()
                if not head or not relation or not tail:
                    continue

                total_triples += 1
                relation_distribution[relation] += 1
                all_entities.update((head, tail))
                if relation in CORE_RELATIONS:
                    core_triples += 1
                    core_entities.update((head, tail))
                    core_diseases.add(head)
                else:
                    candidate_triples += 1

    ordered_distribution = {
        relation: relation_distribution[relation]
        for relation in sorted(relation_distribution)
    }
    return {
        "documents": document_count,
        "documents_with_unique_doc_id": len(doc_ids),
        "total_triples": total_triples,
        "core_triples": core_triples,
        "candidate_triples": candidate_triples,
        "core_entities": len(core_entities),
        "core_disease_entities": len(core_diseases),
        "all_entities": len(all_entities),
        "relation_distribution": ordered_distribution,
    }


def render_text(stats: dict[str, object]) -> str:
    lines = [
        f"Documents: {stats['documents']}",
        f"Core triples: {stats['core_triples']}",
        f"Candidate triples: {stats['candidate_triples']}",
        f"Core entities: {stats['core_entities']}",
        f"Core disease entities: {stats['core_disease_entities']}",
        "Relation distribution:",
    ]
    distribution = stats["relation_distribution"]
    if isinstance(distribution, dict):
        lines.extend(f"  {relation}: {count}" for relation, count in distribution.items())
    return "\n".join(lines)


def render_markdown(stats: dict[str, object]) -> str:
    lines = [
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Documents | {stats['documents']:,} |",
        f"| Core triples | {stats['core_triples']:,} |",
        f"| Candidate triples | {stats['candidate_triples']:,} |",
        f"| Core + candidate triples | {stats['total_triples']:,} |",
        f"| Core entities | {stats['core_entities']:,} |",
        f"| Core disease entities | {stats['core_disease_entities']:,} |",
        "",
        "| Relation | Count |",
        "| --- | ---: |",
    ]
    distribution = stats["relation_distribution"]
    if isinstance(distribution, dict):
        lines.extend(f"| `{relation}` | {count:,} |" for relation, count in distribution.items())
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report GraphRAG relation scale")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS_PATH)
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES_PATH)
    parser.add_argument(
        "--candidate-triples",
        type=Path,
        help=(
            "Optional isolated candidate CSV to combine with core scale statistics"
        ),
    )
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    additional = (args.candidate_triples,) if args.candidate_triples else ()
    stats = calculate_stats(args.documents, args.triples, additional)
    if args.format == "json":
        rendered = json.dumps(stats, ensure_ascii=False, indent=2)
    elif args.format == "markdown":
        rendered = render_markdown(stats)
    else:
        rendered = render_text(stats)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote relation statistics: {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
