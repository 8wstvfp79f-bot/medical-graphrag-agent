from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.retrieval.reranker import (
    LocalCrossEncoderReranker,
    RerankerError,
    rerank_evidence,
)


def evidence(
    evidence_id: str,
    content: str,
    *,
    rank: int,
    evidence_type: str = "vector",
) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "content": content,
        "disease_name": "百日咳",
        "source": "test",
        "sources": ["test"],
        "rank": rank,
        "provenance": [],
    }


class FakeReranker:
    model_name = "fake-cross-encoder"
    normalized_scores = True

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.last_query = ""
        self.last_passages: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.last_query = query
        self.last_passages = list(passages)
        return self.scores


class RerankerTests(unittest.TestCase):
    def test_direct_symptom_evidence_moves_above_pathogen_text(self) -> None:
        rows = [
            evidence("toxin", "百日咳杆菌会产生多种毒素", rank=1),
            evidence("symptom", "百日咳典型症状是痉挛性咳嗽", rank=2),
            evidence(
                "graph",
                "百日咳 -[has_symptom]-> 吸气时有蝉鸣音",
                rank=3,
                evidence_type="graph",
            ),
        ]
        backend = FakeReranker([0.1, 0.95, 0.9])

        result = rerank_evidence(
            "百日咳有哪些典型症状？", rows, backend=backend, top_k=3
        )

        self.assertEqual(
            [item["evidence_id"] for item in result["evidence"]],
            ["symptom", "graph", "toxin"],
        )
        self.assertEqual(result["evidence"][0]["retrieval_rank"], 2)
        self.assertEqual(result["evidence"][0]["rerank_rank"], 1)
        self.assertNotIn("rerank_score", rows[0])

    def test_top_k_and_threshold_are_auditable(self) -> None:
        result = rerank_evidence(
            "问题",
            [
                evidence("a", "甲", rank=1),
                evidence("b", "乙", rank=2),
                evidence("c", "丙", rank=3),
            ],
            backend=FakeReranker([0.9, 0.6, 0.2]),
            top_k=1,
            min_score=0.5,
        )

        self.assertEqual(result["stats"]["returned_evidence"], 1)
        self.assertEqual(result["stats"]["below_min_score"], 1)
        reasons = {row["evidence_id"]: row["reason"] for row in result["excluded_evidence"]}
        self.assertEqual(reasons, {"b": "top_k", "c": "below_min_score"})

    def test_ties_keep_retrieval_order(self) -> None:
        result = rerank_evidence(
            "问题",
            [evidence("second", "乙", rank=2), evidence("first", "甲", rank=1)],
            backend=FakeReranker([0.5, 0.5]),
            top_k=None,
        )
        self.assertEqual(
            [item["evidence_id"] for item in result["evidence"]],
            ["first", "second"],
        )

    def test_invalid_backend_scores_are_rejected(self) -> None:
        with self.assertRaises(RerankerError):
            rerank_evidence(
                "问题",
                [evidence("a", "甲", rank=1)],
                backend=FakeReranker([]),
            )
        with self.assertRaises(RerankerError):
            rerank_evidence(
                "问题",
                [evidence("a", "甲", rank=1)],
                backend=FakeReranker([math.nan]),
            )

    def test_local_model_path_must_exist_without_network_access(self) -> None:
        with self.assertRaises(RerankerError):
            LocalCrossEncoderReranker(Path("definitely-missing-reranker"))


if __name__ == "__main__":
    unittest.main()

