from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator, Mapping

from fastapi.testclient import TestClient

from src.agent.orchestrator import GraphRAGAgent
from src.agent.query_planner import DeterministicMedicalQueryPlanner, DiseaseCatalog
from src.api.app import create_app
from src.api.service import MedicalChatService
from src.generation.answer_generator import ExtractiveCitationAnswerGenerator


class FakeVectorBackend:
    def search(self, query, *, top_k, metadata_filter):
        del query, top_k
        disease_name = metadata_filter["disease_name"][0]
        return [
            {
                "chunk_id": "chunk-api",
                "doc_id": "doc-api",
                "text": f"{disease_name}的典型症状包括阵发性痉挛性咳嗽。",
                "score": 0.94,
                "metadata": {
                    "disease_name": disease_name,
                    "category": ["疾病百科"],
                    "department": ["呼吸内科"],
                    "source": "medical.json",
                },
            }
        ]


class FakeGraphBackend:
    def search(self, disease_name, *, relations, limit):
        del limit
        return [
            {
                "disease_name": disease_name,
                "relationship_type": "HAS_SYMPTOM",
                "entity_name": "痉挛性咳嗽",
                "entity_type": "Symptom",
                "source": "triples.csv",
            }
        ] if "has_symptom" in relations else []


class FakeReranker:
    model_name = "fake-reranker"
    normalized_scores = True

    def score(self, query, passages):
        del query
        return [0.95 - index * 0.05 for index, _ in enumerate(passages)]


def build_service() -> MedicalChatService:
    planner = DeterministicMedicalQueryPlanner(DiseaseCatalog(["百日咳"]))
    agent = GraphRAGAgent(
        planner,
        vector_backend=FakeVectorBackend(),
        graph_backend=FakeGraphBackend(),
        reranker_backend=FakeReranker(),
        context_max_tokens=400,
        context_max_items=3,
    )
    return MedicalChatService(agent, ExtractiveCitationAnswerGenerator())


class ChatServiceTests(unittest.TestCase):
    def test_service_runs_plan_retrieval_and_cited_answer(self) -> None:
        response = asyncio.run(
            build_service().chat({"query": "百日咳有哪些典型症状？"})
        )

        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["query_plan"]["selected_tool"], "hybrid_search")
        self.assertEqual(
            response["retrieval_stats"]["vector_candidates"], 1
        )
        self.assertEqual(response["retrieval_stats"]["graph_candidates"], 1)
        self.assertIn("[E1]", response["answer"])
        self.assertTrue(response["citations"])

    def test_service_stops_and_clarifies_without_a_disease(self) -> None:
        response = asyncio.run(build_service().chat({"query": "有哪些典型症状？"}))

        self.assertEqual(response["status"], "needs_clarification")
        self.assertEqual(response["tool_calls"], [])
        self.assertEqual(response["evidence"], [])

    def test_stream_has_observable_pipeline_events(self) -> None:
        async def collect() -> list[dict[str, object]]:
            return [
                item
                async for item in build_service().stream_chat(
                    {"query": "百日咳有哪些典型症状？"}
                )
            ]

        events = asyncio.run(collect())
        names = [event["event"] for event in events]
        self.assertEqual(names[0], "plan")
        self.assertIn("tool_start", names)
        self.assertIn("tool_result", names)
        self.assertIn("citation", names)
        self.assertIn("token", names)
        self.assertEqual(names[-1], "done")


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(create_app(service=build_service()))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health_and_json_chat_contract(self) -> None:
        self.assertEqual(
            self.client.get("/health").json(),
            {"status": "ok", "service_ready": True},
        )
        response = self.client.post(
            "/chat", json={"query": "百日咳有哪些典型症状？"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertIn("answer", payload)
        self.assertIn("citations", payload)
        self.assertIn("query_plan", payload)

    def test_frontend_and_assets_are_served_by_fastapi(self) -> None:
        page = self.client.get("/")
        script = self.client.get("/assets/app.js")

        self.assertEqual(page.status_code, 200)
        self.assertIn("Medical GraphRAG", page.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn('fetch("/chat"', script.text)

    def test_request_schema_rejects_bad_limits_and_unknown_fields(self) -> None:
        bad_limit = self.client.post(
            "/chat", json={"query": "百日咳有什么症状？", "top_k": 0}
        )
        unknown = self.client.post(
            "/chat", json={"query": "百日咳有什么症状？", "secret": True}
        )

        self.assertEqual(bad_limit.status_code, 422)
        self.assertEqual(unknown.status_code, 422)

    def test_sse_chat_uses_named_events_and_finishes(self) -> None:
        response = self.client.post(
            "/chat",
            json={"query": "百日咳有哪些典型症状？", "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: plan", response.text)
        self.assertIn("event: citation", response.text)
        self.assertIn("event: done", response.text)


if __name__ == "__main__":
    unittest.main()
