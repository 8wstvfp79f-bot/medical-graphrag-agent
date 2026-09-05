from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.data.build_triples import (
    RAW_PATH,
    SOURCE_NAME,
    is_valid_record,
    is_valid_tail,
    iter_json_records,
)
from src.data.medical_quality import clean_text, load_default_quality_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_triples.csv"
DEFAULT_STATS_PATH = (
    PROJECT_ROOT / "results" / "data" / "candidate_relation_stats.json"
)

# These relations deliberately stay outside the production Neo4j allowlist.  They
# are useful for scale experiments, but several values are long text fragments or
# fine-grained product names that would add noise to the main retrieval graph.
LIST_FIELD_RELATIONS = (
    ("cure_way", "treated_with_method"),
    ("do_eat", "recommended_food"),
    ("recommand_eat", "recommended_food"),
    ("not_eat", "avoid_food"),
    ("drug_detail", "has_brand_drug"),
)
TEXT_FIELD_RELATIONS = (
    ("cause", "caused_by"),
    ("prevent", "prevented_by"),
)
SCALAR_FIELD_RELATIONS = (
    ("get_way", "transmitted_by"),
    ("cure_lasttime", "has_duration"),
    ("cost_money", "has_cost_range"),
    ("cured_prob", "has_cure_probability"),
    ("get_prob", "has_prevalence"),
    ("easy_get", "affects_population"),
    ("yibao_status", "has_insurance_status"),
)

_SEGMENT_SPLIT = re.compile(r"[。！？!?；;\n]+|(?=\s*\d+[、.．])")
_CLAUSE_SPLIT = re.compile(r"[，,：:]+")


def _segments(value: object, *, max_chars: int = 220) -> list[str]:
    """Turn long prose into bounded, auditable candidate graph values."""

    text = clean_text(value)
    if not text:
        return []
    result: list[str] = []
    for sentence in _SEGMENT_SPLIT.split(text):
        sentence = clean_text(sentence)
        sentence = re.sub(r"^\d+[、.．]\s*", "", sentence)
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            result.append(sentence)
            continue
        clauses = [clean_text(item) for item in _CLAUSE_SPLIT.split(sentence)]
        clauses = [item for item in clauses if item]
        if len(clauses) > 1:
            result.extend(item[:max_chars] for item in clauses)
        else:
            result.extend(
                sentence[start : start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
    return result


def build_candidate_triples(
    raw_path: Path,
    output_path: Path,
    *,
    source_name: str = SOURCE_NAME,
) -> dict[str, object]:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    quality_policy = load_default_quality_policy()
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, str]] = []
    relation_distribution: Counter[str] = Counter()
    rejected_reason_distribution: Counter[str] = Counter()
    rejected_values = 0
    duplicate_edges = 0
    valid_records = 0

    def add(head: str, field: str, relation: str, value: object) -> None:
        nonlocal duplicate_edges, rejected_values
        accepted, rejected = quality_policy.filter_items(
            field,
            value,
            disease_name=head,
            source=source_name,
        )
        rejected_values += len(rejected)
        rejected_reason_distribution.update(item.reason for item in rejected)
        for tail in accepted:
            tail = clean_text(tail)
            if not is_valid_tail(tail):
                rejected_values += 1
                rejected_reason_distribution["invalid_tail"] += 1
                continue
            key = (head, relation, tail, source_name)
            if key in seen:
                duplicate_edges += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "source": source_name,
                    "field": field,
                    "quality_tier": "candidate",
                }
            )
            relation_distribution[relation] += 1

    for record in iter_json_records(raw_path):
        if not is_valid_record(record):
            continue
        valid_records += 1
        head = clean_text(record.get("name"))
        for field, relation in LIST_FIELD_RELATIONS:
            add(head, field, relation, record.get(field))
        for field, relation in TEXT_FIELD_RELATIONS:
            add(head, field, relation, _segments(record.get(field)))
        for field, relation in SCALAR_FIELD_RELATIONS:
            add(head, field, relation, record.get(field))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "head",
                "relation",
                "tail",
                "source",
                "field",
                "quality_tier",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "raw_path": str(raw_path),
        "output_path": str(output_path),
        "valid_records": valid_records,
        "candidate_triples": len(rows),
        "rejected_values": rejected_values,
        "duplicate_edges_removed": duplicate_edges,
        "rejected_reason_distribution": {
            reason: rejected_reason_distribution[reason]
            for reason in sorted(rejected_reason_distribution)
        },
        "relation_distribution": {
            relation: relation_distribution[relation]
            for relation in sorted(relation_distribution)
        },
        "ingestion_policy": (
            "Candidate relations are not imported by the production Neo4j "
            "core-relation allowlist."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated candidate-relation pool from medical.json"
    )
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS_PATH)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    stats = build_candidate_triples(args.raw, args.output)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
