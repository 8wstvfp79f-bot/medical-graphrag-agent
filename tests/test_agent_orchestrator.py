from __future__ import annotations

import unittest

from src.agent.orchestrator import GraphRAGAgent
from src.agent.query_planner import (
    DeterministicMedicalQueryPlanner,
    DiseaseCatalog,
)


class FakeVectorBackend:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query, *, top_k, metadata_filter):
        self.calls += 1
        disease_name = metadata_filter["disease_name"][0]
        return [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "text": f"{disease_name}的典型症状包括痉挛性咳嗽。",
                "score": 0.91,
                "metadata": {
                    "disease_name": disease_name,
                    "category": ["疾病百科"],
                    "department": ["呼吸内科"],
                    "source": "sample",
                },
            }
        ]


class FakeGraphBackend:
    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.calls = 0
        self.fail_if_called = fail_if_called

    def search(self, disease_name, *, relations, limit):
        del limit
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("graph backend should not be called")
        return [
            {
                "disease_name": disease_name,
                "relationship_type": "HAS_SYMPTOM",
                "entity_name": "痉挛性咳嗽",
                "entity_type": "Symptom",
                "source": "sample",
            }
        ] if "has_symptom" in relations else []


class FakeReranker:
    model_name = "fake-reranker"
    normalized_scores = True

    def score(self, query, passages):
        del query
        return [0.9 - index * 0.1 for index, _ in enumerate(passages)]


class AgentOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog = DiseaseCatalog(["百日咳", "流感"])
        self.planner = DeterministicMedicalQueryPlanner(
            catalog, top_k=4, vector_top_k=3, graph_top_k=3
        )

    def test_natural_question_executes_full_hybrid_evidence_pipeline(self) -> None:
        vector = FakeVectorBackend()
        graph = FakeGraphBackend()
        agent = GraphRAGAgent(
            self.planner,
            vector_backend=vector,
            graph_backend=graph,
            reranker_backend=FakeReranker(),
            context_max_tokens=300,
            context_max_items=3,
        )

        result = agent.run("百日咳有哪些典型症状？")

        self.assertEqual(result["status"], "evidence_ready")
        self.assertEqual(result["query_plan"]["selected_tool"], "hybrid_search")
        self.assertEqual(result["tool_calls"][0]["name"], "hybrid_search")
        self.assertEqual(vector.calls, 1)
        self.assertEqual(graph.calls, 1)
        self.assertEqual(result["retrieval"]["stats"]["vector_candidates"], 1)
        self.assertEqual(result["retrieval"]["stats"]["graph_candidates"], 1)
        self.assertGreaterEqual(
            result["post_retrieval"]["context"]["budget"]["selected_items"], 1
        )
        self.assertIn("[E1", result["post_retrieval"]["context"]["context"])

    def test_text_only_intent_does_not_call_graph(self) -> None:
        vector = FakeVectorBackend()
        graph = FakeGraphBackend(fail_if_called=True)
        agent = GraphRAGAgent(
            self.planner,
            vector_backend=vector,
            graph_backend=graph,
            reranker_backend=FakeReranker(),
        )

        result = agent.run("百日咳的病因是什么？")

        self.assertEqual(result["query_plan"]["selected_tool"], "vector_search")
        self.assertEqual(vector.calls, 1)
        self.assertEqual(graph.calls, 0)
        self.assertEqual(result["retrieval"]["stats"]["graph_candidates"], 0)

    def test_missing_disease_stops_before_any_backend(self) -> None:
        vector = FakeVectorBackend()
        graph = FakeGraphBackend()
        agent = GraphRAGAgent(
            self.planner,
            vector_backend=vector,
            graph_backend=graph,
        )

        result = agent.run("有哪些典型症状？")

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(vector.calls, 0)
        self.assertEqual(graph.calls, 0)


if __name__ == "__main__":
    unittest.main()
