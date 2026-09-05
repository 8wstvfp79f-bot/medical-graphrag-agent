from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


INTENT_RELATIONS = {
    "symptom": "has_symptom",
    "diagnosis": "diagnosed_by",
    "treatment": "treated_by",
    "department": "belongs_to",
    "complication": "has_complication",
}


class EvaluationDatasetError(ValueError):
    """Raised when the fixed evaluation set is malformed."""


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def _string_list(value: object, *, field: str, required: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvaluationDatasetError(f"{field} must be an array")
    values = tuple(dict.fromkeys(_clean(item) for item in value if _clean(item)))
    if required and not values:
        raise EvaluationDatasetError(f"{field} must contain at least one value")
    return values


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    disease_name: str
    intent: str
    relation: str
    gold_entities: tuple[str, ...]
    forbidden_entities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: object, *, location: str) -> "EvaluationCase":
        if not isinstance(value, dict):
            raise EvaluationDatasetError(f"Expected an object at {location}")
        case_id = _clean(value.get("case_id"))
        question = _clean(value.get("question"))
        disease_name = _clean(value.get("disease_name"))
        intent = _clean(value.get("intent"))
        relation = _clean(value.get("relation"))
        if not case_id or not question or not disease_name:
            raise EvaluationDatasetError(
                f"case_id, question, and disease_name are required at {location}"
            )
        expected_relation = INTENT_RELATIONS.get(intent)
        if expected_relation is None:
            raise EvaluationDatasetError(
                f"Unsupported intent '{intent}' at {location}"
            )
        if relation != expected_relation:
            raise EvaluationDatasetError(
                f"Intent '{intent}' requires relation '{expected_relation}' at {location}"
            )
        return cls(
            case_id=case_id,
            question=question,
            disease_name=disease_name,
            intent=intent,
            relation=relation,
            gold_entities=_string_list(
                value.get("gold_entities"), field="gold_entities", required=True
            ),
            forbidden_entities=_string_list(
                value.get("forbidden_entities", []),
                field="forbidden_entities",
                required=False,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "disease_name": self.disease_name,
            "intent": self.intent,
            "relation": self.relation,
            "gold_entities": list(self.gold_entities),
            "forbidden_entities": list(self.forbidden_entities),
        }


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_no}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationDatasetError(f"Invalid JSON at {location}: {exc}") from exc
            case = EvaluationCase.from_mapping(value, location=location)
            if case.case_id in seen_ids:
                raise EvaluationDatasetError(
                    f"Duplicate case_id '{case.case_id}' at {location}"
                )
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise EvaluationDatasetError("Evaluation dataset must not be empty")
    return cases
