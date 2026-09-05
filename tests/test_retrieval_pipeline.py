from __future__ import annotations

import unittest

from src.retrieval.retrieval_pipeline import prepare_evidence_context


class FakeReranker:
    model_name = "fake-cross-encoder"
    normalized_scores = True

    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [0.95 if "症状" in passage else 0.2 for passage in passages]


class RetrievalPipelineTests(unittest.TestCase):
    def test_reranking_and_context_budget_form_one_pipeline(self) -> None:
        rows = [
            {
                "evidence_id": "mechanism",
                "evidence_type": "vector",
                "content": "百日咳杆菌毒素机制",
                "disease_name": "百日咳",
                "source": "text",
                "sources": ["text"],
                "rank": 1,
                "provenance": [{"evidence_type": "vector"}],
            },
            {
                "evidence_id": "symptom",
                "evidence_type": "graph",
                "content": "百日咳症状是痉挛性咳嗽",
                "disease_name": "百日咳",
                "source": "graph",
                "sources": ["graph"],
                "rank": 2,
                "provenance": [{"evidence_type": "graph"}],
            },
        ]

        result = prepare_evidence_context(
            "百日咳有哪些症状？",
            rows,
            reranker_backend=FakeReranker(),
            rerank_top_k=2,
            context_max_tokens=300,
            context_max_items=2,
        )

        self.assertEqual(result["reranked_evidence"][0]["evidence_id"], "symptom")
        self.assertEqual(result["context"]["budget"]["selected_items"], 2)
        self.assertIn("百日咳症状是痉挛性咳嗽", result["context"]["context"])


if __name__ == "__main__":
    unittest.main()

