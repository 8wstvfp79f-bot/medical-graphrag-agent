from __future__ import annotations

import asyncio
import threading
import unittest

from src.retrieval.hybrid_retriever import hybrid_search_async


class ConcurrentVectorBackend:
    def __init__(self, vector_started: threading.Event, graph_started: threading.Event):
        self.vector_started = vector_started
        self.graph_started = graph_started

    def search(self, query, *, top_k, metadata_filter):
        del query, top_k
        self.vector_started.set()
        if not self.graph_started.wait(timeout=1):
            raise AssertionError("graph search did not start concurrently")
        disease_name = metadata_filter["disease_name"][0]
        return [
            {
                "chunk_id": "chunk-async",
                "doc_id": "doc-async",
                "text": f"{disease_name}可出现痉挛性咳嗽。",
                "score": 0.9,
                "metadata": {
                    "disease_name": disease_name,
                    "category": ["疾病百科"],
                    "department": ["呼吸内科"],
                    "source": "medical.json",
                },
            }
        ]


class ConcurrentGraphBackend:
    def __init__(self, vector_started: threading.Event, graph_started: threading.Event):
        self.vector_started = vector_started
        self.graph_started = graph_started

    def search(self, disease_name, *, relations, limit):
        del limit
        self.graph_started.set()
        if not self.vector_started.wait(timeout=1):
            raise AssertionError("vector search did not start concurrently")
        return [
            {
                "disease_name": disease_name,
                "relationship_type": "HAS_SYMPTOM",
                "entity_name": "痉挛性咳嗽",
                "entity_type": "Symptom",
                "source": "triples.csv",
            }
        ] if "has_symptom" in relations else []


class FailingVectorBackend:
    def search(self, *args, **kwargs):
        del args, kwargs
        raise ConnectionError("Milvus unavailable")


class AsyncHybridTests(unittest.TestCase):
    def test_vector_and_graph_branches_really_start_concurrently(self) -> None:
        vector_started = threading.Event()
        graph_started = threading.Event()
        result = asyncio.run(
            hybrid_search_async(
                "百日咳有哪些症状？",
                "百日咳",
                vector_backend=ConcurrentVectorBackend(vector_started, graph_started),
                graph_backend=ConcurrentGraphBackend(vector_started, graph_started),
                relations=["has_symptom"],
                top_k=4,
            )
        )

        self.assertTrue(vector_started.is_set())
        self.assertTrue(graph_started.is_set())
        self.assertEqual(
            result["strategy"]["execution"], "vector_and_graph_concurrent"
        )
        self.assertEqual(result["stats"]["vector_candidates"], 1)
        self.assertEqual(result["stats"]["graph_candidates"], 1)

    def test_partial_mode_keeps_the_successful_async_branch(self) -> None:
        started = threading.Event()
        result = asyncio.run(
            hybrid_search_async(
                "百日咳有哪些症状？",
                "百日咳",
                vector_backend=FailingVectorBackend(),
                graph_backend=ConcurrentGraphBackend(started, started),
                relations=["has_symptom"],
                allow_partial=True,
            )
        )

        self.assertIn("ConnectionError", result["errors"]["vector"])
        self.assertEqual(result["stats"]["graph_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
