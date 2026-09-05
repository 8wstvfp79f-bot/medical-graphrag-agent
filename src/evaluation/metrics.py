from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from src.evaluation.schemas import EvaluationCase


_CITATION = re.compile(r"\[(E\d+)\]")
_NUMBERED_STATEMENT = re.compile(r"^\s*\d+[.、]\s*(.+)$")


def normalize_text(value: object) -> str:
    text = str(value or "").casefold()
    return re.sub(r"[\s\u3000，,。；;：:、…\"'“”‘’（）()\[\]]+", "", text)


def _evidence_text(item: Mapping[str, object]) -> str:
    metadata = item.get("metadata")
    metadata_text = ""
    if isinstance(metadata, Mapping):
        metadata_text = " ".join(str(value) for value in metadata.values())
    return " ".join(
        str(item.get(field, ""))
        for field in ("content", "disease_name", "entity_name", "path_text")
    ) + " " + metadata_text


def _matched_gold(
    case: EvaluationCase, evidence: Sequence[Mapping[str, object]]
) -> list[str]:
    searchable = normalize_text(" ".join(_evidence_text(item) for item in evidence))
    return [gold for gold in case.gold_entities if normalize_text(gold) in searchable]


def evaluate_retrieval(
    case: EvaluationCase,
    evidence: Sequence[Mapping[str, object]],
    *,
    k: int = 3,
) -> dict[str, object]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    top = list(evidence[:k])
    matched = _matched_gold(case, top)
    relation_matches = [
        item
        for item in top
        if normalize_text(item.get("disease_name"))
        == normalize_text(case.disease_name)
        and normalize_text(item.get("relation")) == normalize_text(case.relation)
    ]
    first_rank: int | None = None
    for rank, item in enumerate(evidence, start=1):
        correct_graph_relation = (
            normalize_text(item.get("disease_name"))
            == normalize_text(case.disease_name)
            and normalize_text(item.get("relation"))
            == normalize_text(case.relation)
        )
        if _matched_gold(case, [item]) or correct_graph_relation:
            first_rank = rank
            break

    metadata_matches = [
        normalize_text(item.get("disease_name")) == normalize_text(case.disease_name)
        for item in top
    ]
    graph_relation_hit = bool(relation_matches)
    forbidden_hits = [
        term
        for term in case.forbidden_entities
        if normalize_text(term)
        and normalize_text(term)
        in normalize_text(" ".join(_evidence_text(item) for item in top))
    ]
    return {
        "k": k,
        "returned": len(top),
        "hit_at_k": bool(matched or relation_matches),
        "gold_recall_at_k": round(len(matched) / len(case.gold_entities), 6),
        "matched_gold_entities": matched,
        "first_relevant_rank": first_rank,
        "reciprocal_rank": round(1 / first_rank, 6) if first_rank else 0.0,
        "metadata_accuracy_at_k": (
            round(sum(metadata_matches) / len(metadata_matches), 6)
            if metadata_matches
            else 0.0
        ),
        "graph_relation_hit_at_k": graph_relation_hit,
        "forbidden_hits": forbidden_hits,
    }


def _numbered_statements(answer: str) -> list[str]:
    statements: list[str] = []
    for line in answer.splitlines():
        match = _NUMBERED_STATEMENT.match(line)
        if match:
            statements.append(match.group(1).strip())
    return statements


def _claim_supported(
    statement: str,
    cited_ids: Sequence[str],
    evidence_by_citation: Mapping[str, Mapping[str, object]],
) -> bool:
    claim = _CITATION.sub("", statement)
    claim = re.sub(r"^资料显示[：:]", "", claim).strip().rstrip("。…")
    normalized_claim = normalize_text(claim)
    for citation_id in cited_ids:
        item = evidence_by_citation.get(citation_id)
        if item is None:
            continue
        evidence_type = str(item.get("evidence_type", ""))
        if evidence_type in {"graph", "hybrid"}:
            entity = normalize_text(item.get("entity_name"))
            disease = normalize_text(item.get("disease_name"))
            if entity and entity in normalized_claim and disease in normalized_claim:
                return True
        content = normalize_text(item.get("content"))
        if normalized_claim and normalized_claim in content:
            return True
    return False


def evaluate_answer(
    case: EvaluationCase,
    answer: str,
    citations: Sequence[Mapping[str, object]],
    context_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    valid_ids = {
        str(item.get("citation_id"))
        for item in citations
        if str(item.get("citation_id", "")).strip()
    }
    evidence_by_citation = {
        str(item.get("citation_id")): item
        for item in context_evidence
        if str(item.get("citation_id", "")).strip()
    }
    used_ids = _CITATION.findall(answer)
    invalid_ids = sorted(
        citation_id
        for citation_id in set(used_ids)
        if citation_id not in valid_ids or citation_id not in evidence_by_citation
    )
    statements = _numbered_statements(answer)
    cited_statements = [statement for statement in statements if _CITATION.search(statement)]
    unsupported: list[str] = []
    for statement in statements:
        statement_ids = _CITATION.findall(statement)
        if not statement_ids or not _claim_supported(
            statement, statement_ids, evidence_by_citation
        ):
            unsupported.append(statement)

    answer_text = normalize_text(answer)
    matched_gold = [
        gold for gold in case.gold_entities if normalize_text(gold) in answer_text
    ]
    forbidden_hits = [
        term
        for term in case.forbidden_entities
        if normalize_text(term) and normalize_text(term) in answer_text
    ]
    citation_coverage = (
        len(cited_statements) / len(statements) if statements else 1.0
    )
    unsupported_rate = len(unsupported) / len(statements) if statements else 0.0
    return {
        "statement_count": len(statements),
        "used_citation_ids": list(dict.fromkeys(used_ids)),
        "invalid_citation_ids": invalid_ids,
        "citation_validity": not invalid_ids and bool(used_ids),
        "statement_citation_coverage": round(citation_coverage, 6),
        "unsupported_claim_rate": round(unsupported_rate, 6),
        "unsupported_statements": unsupported,
        "gold_entity_coverage": round(
            len(matched_gold) / len(case.gold_entities), 6
        ),
        "matched_gold_entities": matched_gold,
        "forbidden_hits": forbidden_hits,
        "metric_note": (
            "unsupported_claim_rate is a deterministic citation/evidence proxy, "
            "not a semantic LLM hallucination score"
        ),
    }
