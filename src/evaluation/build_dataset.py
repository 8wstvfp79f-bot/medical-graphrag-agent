from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from src.data.medical_quality import load_default_quality_policy
from src.evaluation.schemas import INTENT_RELATIONS, load_evaluation_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIPLES = PROJECT_ROOT / "data" / "processed" / "triples.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "medical_qa_eval.jsonl"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "evaluation" / "medical_qa_eval.manifest.json"
)
DEFAULT_DISEASE_COUNT = 50
DEFAULT_SEED = "medical-graphrag-eval-v2"

QUESTION_TEMPLATES = {
    "symptom": "{disease}有哪些典型症状？",
    "diagnosis": "{disease}通常需要做哪些检查？",
    "treatment": "{disease}常用哪些药物治疗？",
    "department": "{disease}应该挂什么科？",
    "complication": "{disease}可能有哪些并发症？",
}

# Keep familiar interview/demo diseases when all five gold relation types exist.
PRIORITY_DISEASES = (
    "百日咳",
    "流行性感冒",
    "高血压",
    "糖尿病",
    "支气管哮喘",
    "肺炎",
)


def _stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _load_relation_gold(
    triples_path: Path,
) -> dict[str, dict[str, list[str]]]:
    if not triples_path.is_file():
        raise FileNotFoundError(f"Core triples not found: {triples_path}")
    grouped: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[tuple[str, str, str]] = set()
    with triples_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            disease = str(row.get("head", "")).strip()
            relation = str(row.get("relation", "")).strip()
            entity = str(row.get("tail", "")).strip()
            if (
                not disease
                or relation not in INTENT_RELATIONS.values()
                or not entity
                or (disease, relation, entity) in seen
            ):
                continue
            seen.add((disease, relation, entity))
            grouped[disease][relation].append(entity)
    return grouped


def build_evaluation_dataset(
    triples_path: Path,
    output_path: Path,
    *,
    disease_count: int = DEFAULT_DISEASE_COUNT,
    seed: str = DEFAULT_SEED,
    gold_per_case: int = 3,
) -> dict[str, object]:
    if disease_count <= 0 or gold_per_case <= 0:
        raise ValueError("disease_count and gold_per_case must be greater than zero")
    grouped = _load_relation_gold(triples_path)
    required_relations = set(INTENT_RELATIONS.values())
    eligible = [
        disease
        for disease, relations in grouped.items()
        if required_relations.issubset(relations)
    ]
    priority = [disease for disease in PRIORITY_DISEASES if disease in eligible]
    remaining = sorted(
        (disease for disease in eligible if disease not in priority),
        key=lambda disease: _stable_key(seed, disease),
    )
    selected = (priority + remaining)[:disease_count]
    if len(selected) < disease_count:
        raise ValueError(
            f"Only {len(selected)} diseases contain all five core relations; "
            f"requested {disease_count}"
        )

    policy = load_default_quality_policy()
    forbidden_entities = sorted(
        {
            value
            for values in policy.rejected_exact.values()
            for value in values
        }
    )
    rows: list[dict[str, object]] = []
    for disease_index, disease in enumerate(selected, start=1):
        for intent, relation in INTENT_RELATIONS.items():
            values = grouped[disease][relation]
            ordered = sorted(
                values,
                key=lambda entity: _stable_key(
                    seed, f"{disease}:{relation}:{entity}"
                ),
            )
            rows.append(
                {
                    "case_id": f"case_{disease_index:03d}_{intent}",
                    "question": QUESTION_TEMPLATES[intent].format(disease=disease),
                    "disease_name": disease,
                    "intent": intent,
                    "relation": relation,
                    "gold_entities": ordered[:gold_per_case],
                    "forbidden_entities": forbidden_entities,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    # Parse the generated file through the production schema before reporting it.
    validated = load_evaluation_cases(output_path)
    distribution = {
        intent: sum(case.intent == intent for case in validated)
        for intent in INTENT_RELATIONS
    }
    return {
        "dataset": str(output_path),
        "seed": seed,
        "eligible_diseases": len(eligible),
        "selected_diseases": len(selected),
        "case_count": len(validated),
        "gold_entity_references": sum(
            len(case.gold_entities) for case in validated
        ),
        "intent_distribution": distribution,
        "provenance": (
            "Deterministically derived from cleaned core triples; not independently "
            "clinician-annotated."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic 200+ case medical retrieval evaluation set"
    )
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--disease-count", type=int, default=DEFAULT_DISEASE_COUNT)
    parser.add_argument("--gold-per-case", type=int, default=3)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    manifest = build_evaluation_dataset(
        args.triples,
        args.output,
        disease_count=args.disease_count,
        seed=args.seed,
        gold_per_case=args.gold_per_case,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
