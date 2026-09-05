from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence


DEFAULT_CONTEXT_MAX_TOKENS = 2000
DEFAULT_CONTEXT_MAX_ITEMS = 6
SUPPORTED_EVIDENCE_TYPES = frozenset({"vector", "graph", "hybrid"})


class ContextBudgetError(ValueError):
    """Raised when evidence cannot be assembled under a valid context budget."""


_TOKEN_PARTS = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s\w]",
    flags=re.UNICODE,
)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without depending on a target LLM.

    Each CJK character and punctuation mark counts as one token. ASCII word-like
    runs count as roughly one token per four characters. A later LLM adapter can
    inject its exact tokenizer through `token_counter`.
    """

    count = 0
    for part in _TOKEN_PARTS.findall(str(text or "")):
        if part.isascii() and (part.isalnum() or "_" in part):
            count += max(1, math.ceil(len(part) / 4))
        else:
            count += 1
    return count


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def _evidence_types(item: Mapping[str, object]) -> set[str]:
    evidence_type = _clean(item.get("evidence_type"))
    types = {evidence_type} if evidence_type in {"vector", "graph"} else set()
    if evidence_type == "hybrid":
        types.update({"vector", "graph"})
    provenance = item.get("provenance")
    if isinstance(provenance, list):
        for source in provenance:
            if isinstance(source, Mapping):
                source_type = _clean(source.get("evidence_type"))
                if source_type in {"vector", "graph"}:
                    types.add(source_type)
    return types


def _source_text(item: Mapping[str, object]) -> str:
    sources = item.get("sources")
    if isinstance(sources, list):
        cleaned = [_clean(source) for source in sources if _clean(source)]
        if cleaned:
            return ",".join(dict.fromkeys(cleaned))
    return _clean(item.get("source")) or "unknown"


def _render_block(item: Mapping[str, object], citation_id: str) -> str:
    evidence_type = _clean(item.get("evidence_type"))
    disease_name = _clean(item.get("disease_name"))
    relation = _clean(item.get("relation"))
    header_parts = [
        citation_id,
        f"type={evidence_type}",
        f"disease={disease_name}",
        f"source={_source_text(item)}",
    ]
    if relation:
        header_parts.append(f"relation={relation}")
    return f"[{' | '.join(header_parts)}]\n{_clean(item.get('content'))}"


def build_context(
    evidence: Sequence[Mapping[str, object]],
    *,
    max_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    max_items: int = DEFAULT_CONTEXT_MAX_ITEMS,
    min_type_counts: Mapping[str, int] | None = None,
    token_counter: Callable[[str], int] = estimate_tokens,
) -> dict[str, object]:
    """Select reranked evidence under token and evidence-type budgets."""

    if max_tokens <= 0:
        raise ContextBudgetError("max_tokens must be greater than zero")
    if max_items <= 0:
        raise ContextBudgetError("max_items must be greater than zero")

    requested_quotas = dict(min_type_counts or {"vector": 1, "graph": 1})
    for evidence_type, count in requested_quotas.items():
        if evidence_type not in {"vector", "graph"}:
            raise ContextBudgetError(
                f"Unsupported evidence type quota: {evidence_type}"
            )
        if not isinstance(count, int) or count < 0:
            raise ContextBudgetError("Evidence type quotas must be non-negative integers")

    prepared: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for position, source_item in enumerate(evidence, start=1):
        item = dict(source_item)
        evidence_id = _clean(item.get("evidence_id"))
        content = _clean(item.get("content"))
        evidence_type = _clean(item.get("evidence_type"))
        if not evidence_id or not content:
            raise ContextBudgetError(
                "Every evidence item requires evidence_id and content"
            )
        if evidence_id in seen_ids:
            raise ContextBudgetError(f"Duplicate evidence_id: {evidence_id}")
        if evidence_type not in SUPPORTED_EVIDENCE_TYPES:
            raise ContextBudgetError(f"Unsupported evidence_type: {evidence_type}")
        seen_ids.add(evidence_id)

        rank_value = item.get("rerank_rank", item.get("rank", position))
        try:
            priority_rank = int(rank_value)
        except (TypeError, ValueError) as exc:
            raise ContextBudgetError("Evidence rank must be an integer") from exc
        if priority_rank <= 0:
            raise ContextBudgetError("Evidence rank must be greater than zero")
        item["_priority_rank"] = priority_rank
        item["_types"] = _evidence_types(item)
        prepared.append(item)

    prepared.sort(
        key=lambda item: (
            int(item["_priority_rank"]),
            str(item["evidence_id"]),
        )
    )

    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    used_tokens = 0

    def try_select(item: dict[str, object]) -> bool:
        nonlocal used_tokens
        evidence_id = str(item["evidence_id"])
        if evidence_id in selected_ids or len(selected) >= max_items:
            return False
        conservative_block = _render_block(item, "EVIDENCE")
        token_count = int(token_counter(conservative_block))
        if token_count <= 0:
            raise ContextBudgetError("token_counter must return a positive integer")
        if used_tokens + token_count > max_tokens:
            return False
        selected.append(item)
        selected_ids.add(evidence_id)
        item["_context_token_count"] = token_count
        used_tokens += token_count
        return True

    quota_selected: Counter[str] = Counter()
    for evidence_type, required_count in requested_quotas.items():
        for item in prepared:
            if quota_selected[evidence_type] >= required_count:
                break
            item_types = item["_types"]
            assert isinstance(item_types, set)
            if evidence_type not in item_types:
                continue
            already_selected = str(item["evidence_id"]) in selected_ids
            if already_selected or try_select(item):
                quota_selected[evidence_type] += 1

    for item in prepared:
        try_select(item)

    selected.sort(
        key=lambda item: (
            int(item["_priority_rank"]),
            str(item["evidence_id"]),
        )
    )

    context_blocks: list[str] = []
    citations: list[dict[str, object]] = []
    output_evidence: list[dict[str, object]] = []
    final_used_tokens = 0
    final_type_counts: Counter[str] = Counter()
    for context_rank, item in enumerate(selected, start=1):
        citation_id = f"E{context_rank}"
        block = _render_block(item, citation_id)
        token_count = int(token_counter(block))
        final_used_tokens += token_count

        output_item = {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        }
        output_item["context_rank"] = context_rank
        output_item["citation_id"] = citation_id
        output_item["context_token_count"] = token_count
        output_evidence.append(output_item)
        context_blocks.append(block)

        item_types = item["_types"]
        assert isinstance(item_types, set)
        final_type_counts.update(item_types)
        citations.append(
            {
                "citation_id": citation_id,
                "evidence_id": item["evidence_id"],
                "evidence_type": item["evidence_type"],
                "sources": item.get("sources", [item.get("source")]),
            }
        )

    if final_used_tokens > max_tokens:
        raise ContextBudgetError(
            "Rendered context exceeded max_tokens; token_counter is not stable"
        )

    unmet_quotas = {
        evidence_type: max(0, required - final_type_counts[evidence_type])
        for evidence_type, required in requested_quotas.items()
        if final_type_counts[evidence_type] < required
    }
    excluded_ids = [
        str(item["evidence_id"])
        for item in prepared
        if str(item["evidence_id"]) not in selected_ids
    ]
    return {
        "context": "\n\n".join(context_blocks),
        "evidence": output_evidence,
        "citations": citations,
        "excluded_evidence_ids": excluded_ids,
        "budget": {
            "token_counter": getattr(token_counter, "__name__", type(token_counter).__name__),
            "max_tokens": max_tokens,
            "used_tokens": final_used_tokens,
            "remaining_tokens": max_tokens - final_used_tokens,
            "max_items": max_items,
            "selected_items": len(output_evidence),
            "requested_type_quotas": requested_quotas,
            "selected_type_counts": dict(final_type_counts),
            "unmet_type_quotas": unmet_quotas,
        },
    }

