from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationSpec:
    relation: str
    relationship_type: str
    tail_label: str
    description: str


RELATION_SPECS: dict[str, RelationSpec] = {
    "has_symptom": RelationSpec(
        relation="has_symptom",
        relationship_type="HAS_SYMPTOM",
        tail_label="Symptom",
        description="疾病具有某个症状",
    ),
    "diagnosed_by": RelationSpec(
        relation="diagnosed_by",
        relationship_type="DIAGNOSED_BY",
        tail_label="Check",
        description="疾病可通过某项检查辅助诊断",
    ),
    "treated_by": RelationSpec(
        relation="treated_by",
        relationship_type="TREATED_BY",
        tail_label="Drug",
        description="疾病常用或推荐某种药品治疗",
    ),
    "belongs_to": RelationSpec(
        relation="belongs_to",
        relationship_type="BELONGS_TO",
        tail_label="Department",
        description="疾病属于某个就诊科室",
    ),
    "has_complication": RelationSpec(
        relation="has_complication",
        relationship_type="HAS_COMPLICATION",
        tail_label="Complication",
        description="疾病可能伴随某种并发症",
    ),
}

CORE_RELATIONS = tuple(RELATION_SPECS)
RELATIONSHIP_TO_RELATION = {
    spec.relationship_type: spec.relation for spec in RELATION_SPECS.values()
}
NODE_LABELS = (
    "Disease",
    "Symptom",
    "Check",
    "Drug",
    "Department",
    "Complication",
)


def normalize_relations(relations: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if relations is None:
        return CORE_RELATIONS

    normalized: list[str] = []
    seen: set[str] = set()
    for value in relations:
        relation = "_".join(str(value).strip().lower().split())
        if relation not in RELATION_SPECS:
            raise ValueError(
                f"Unsupported graph relation: {value}. Expected one of: "
                f"{', '.join(CORE_RELATIONS)}"
            )
        if relation not in seen:
            seen.add(relation)
            normalized.append(relation)
    if not normalized:
        raise ValueError("At least one graph relation is required")
    return tuple(normalized)
