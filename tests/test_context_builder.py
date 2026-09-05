from __future__ import annotations

import unittest

from src.generation.context_builder import (
    ContextBudgetError,
    build_context,
    estimate_tokens,
)


def evidence(
    evidence_id: str,
    content: str,
    *,
    rerank_rank: int,
    evidence_type: str,
) -> dict[str, object]:
    provenance_types = (
        ["vector", "graph"] if evidence_type == "hybrid" else [evidence_type]
    )
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "content": content,
        "disease_name": "百日咳",
        "source": "test",
        "sources": ["test"],
        "rerank_rank": rerank_rank,
        "provenance": [
            {"evidence_type": source_type} for source_type in provenance_types
        ],
    }


def word_counter(text: str) -> int:
    return len(text.split())


class ContextBuilderTests(unittest.TestCase):
    def test_estimator_counts_chinese_and_ascii_conservatively(self) -> None:
        self.assertGreaterEqual(estimate_tokens("百日咳 symptom"), 4)
        self.assertEqual(estimate_tokens(""), 0)

    def test_budget_is_never_exceeded(self) -> None:
        result = build_context(
            [
                evidence("v1", "one two three four", rerank_rank=1, evidence_type="vector"),
                evidence("g1", "five six", rerank_rank=2, evidence_type="graph"),
                evidence("v2", "seven eight nine", rerank_rank=3, evidence_type="vector"),
            ],
            max_tokens=28,
            max_items=3,
            token_counter=word_counter,
        )
        self.assertLessEqual(result["budget"]["used_tokens"], 28)
        self.assertEqual(result["budget"]["unmet_type_quotas"], {})

    def test_type_quotas_keep_vector_and_graph_sources(self) -> None:
        result = build_context(
            [
                evidence("v1", "向量一", rerank_rank=1, evidence_type="vector"),
                evidence("v2", "向量二", rerank_rank=2, evidence_type="vector"),
                evidence("g1", "图谱一", rerank_rank=3, evidence_type="graph"),
            ],
            max_tokens=1000,
            max_items=2,
        )

        self.assertEqual(
            [item["evidence_id"] for item in result["evidence"]], ["v1", "g1"]
        )
        self.assertEqual(result["budget"]["selected_type_counts"], {"vector": 1, "graph": 1})

    def test_hybrid_evidence_can_satisfy_both_type_quotas(self) -> None:
        result = build_context(
            [evidence("h1", "共同证据", rerank_rank=1, evidence_type="hybrid")],
            max_tokens=100,
            max_items=1,
        )
        self.assertEqual(result["budget"]["unmet_type_quotas"], {})
        self.assertEqual(result["budget"]["selected_type_counts"], {"vector": 1, "graph": 1})

    def test_unfittable_evidence_is_reported_as_excluded(self) -> None:
        result = build_context(
            [evidence("large", "one two three four five", rerank_rank=1, evidence_type="vector")],
            max_tokens=2,
            token_counter=word_counter,
        )
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["excluded_evidence_ids"], ["large"])
        self.assertEqual(result["budget"]["unmet_type_quotas"], {"vector": 1, "graph": 1})

    def test_invalid_budget_and_type_are_rejected(self) -> None:
        with self.assertRaises(ContextBudgetError):
            build_context([], max_tokens=0)
        with self.assertRaises(ContextBudgetError):
            build_context(
                [evidence("x", "证据", rerank_rank=1, evidence_type="unknown")]
            )


if __name__ == "__main__":
    unittest.main()

