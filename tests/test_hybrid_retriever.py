from __future__ import annotations

import unittest

from src.retrieval.hybrid_retriever import (
    HybridRetrievalError,
    fuse_evidence,
    hybrid_search,
)


def vector_hit(
    chunk_id: str,
    text: str,
    *,
    disease_name: str = "百日咳",
    score: float = 0.9,
) -> dict[str, object]:
    metadata = {
        "disease_name": disease_name,
        "category": ["疾病百科"],
        "department": ["儿科"],
        "source": "xywy_disease_encyclopedia",
    }
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc_{chunk_id}",
        "text": text,
        "score": score,
        "metadata": metadata,
        "matched_metadata": metadata,
    }


def graph_hit(
    entity_name: str,
    *,
    relation: str = "has_symptom",
    path_text: str | None = None,
) -> dict[str, object]:
    return {
        "evidence_type": "graph",
        "disease_name": "百日咳",
        "relation": relation,
        "entity_name": entity_name,
        "entity_type": "Symptom",
        "source": "xywy",
        "path_text": path_text or f"百日咳 -[{relation}]-> {entity_name}",
    }


class FakeVectorBackend:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.last_filter: dict[str, tuple[str, ...]] | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, tuple[str, ...]],
    ) -> list[dict[str, object]]:
        del query
        self.last_filter = metadata_filter
        candidates = []
        for row in self.rows:
            payload = dict(row)
            score = payload.pop("score")
            candidates.append({"entity": payload, "distance": score})
        return candidates[:top_k]


class FakeGraphBackend:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def search(
        self,
        disease_name: str,
        *,
        relations: tuple[str, ...],
        limit: int,
    ) -> list[dict[str, object]]:
        relationship_types = {
            "has_symptom": "HAS_SYMPTOM",
            "diagnosed_by": "DIAGNOSED_BY",
            "treated_by": "TREATED_BY",
            "belongs_to": "BELONGS_TO",
            "has_complication": "HAS_COMPLICATION",
        }
        results = []
        for row in self.rows:
            if row["relation"] not in relations:
                continue
            results.append(
                {
                    "disease_name": disease_name,
                    "relationship_type": relationship_types[str(row["relation"])],
                    "entity_name": row["entity_name"],
                    "entity_type": row["entity_type"],
                    "source": row["source"],
                }
            )
        return results[:limit]


class FailingVectorBackend:
    def search(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        raise ConnectionError("Milvus unavailable")


class HybridRetrieverTests(unittest.TestCase):
    def test_equal_weight_rrf_interleaves_vector_and_graph(self) -> None:
        evidence, stats = fuse_evidence(
            [
                vector_hit("v1", "百日咳文本证据一", score=0.95),
                vector_hit("v2", "百日咳文本证据二", score=0.85),
            ],
            [graph_hit("痉挛性咳嗽"), graph_hit("低热")],
            top_k=4,
        )

        self.assertEqual(
            [item["evidence_type"] for item in evidence],
            ["vector", "graph", "vector", "graph"],
        )
        self.assertEqual([item["rank"] for item in evidence], [1, 2, 3, 4])
        self.assertEqual(stats["returned_evidence"], 4)

    def test_disease_filter_is_injected_and_auditable(self) -> None:
        vector_backend = FakeVectorBackend(
            [vector_hit("v1", "百日咳的症状包括痉挛性咳嗽")]
        )
        result = hybrid_search(
            "百日咳有哪些症状",
            "百日咳",
            vector_backend=vector_backend,
            graph_backend=FakeGraphBackend([graph_hit("痉挛性咳嗽")]),
            metadata_filter={"department": "儿科"},
            relations=["has_symptom"],
            top_k=2,
        )

        assert vector_backend.last_filter is not None
        self.assertEqual(vector_backend.last_filter["disease_name"], ("百日咳",))
        self.assertEqual(vector_backend.last_filter["department"], ("儿科",))
        self.assertEqual(result["metadata_filter"]["disease_name"], ["百日咳"])
        self.assertEqual(result["stats"]["vector_candidates"], 1)
        self.assertEqual(result["stats"]["graph_candidates"], 1)

    def test_conflicting_vector_and_graph_disease_is_rejected(self) -> None:
        vector_backend = FakeVectorBackend([])
        with self.assertRaises(HybridRetrievalError):
            hybrid_search(
                "百日咳有哪些症状",
                "百日咳",
                vector_backend=vector_backend,
                graph_backend=FakeGraphBackend([]),
                metadata_filter={"disease_name": "顿咳"},
            )
        self.assertIsNone(vector_backend.last_filter)

    def test_exact_duplicate_text_is_merged_with_provenance(self) -> None:
        duplicate_text = "百日咳 -[has_symptom]-> 痉挛性咳嗽"
        evidence, stats = fuse_evidence(
            [vector_hit("v1", duplicate_text)],
            [graph_hit("痉挛性咳嗽", path_text=duplicate_text)],
            top_k=5,
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["evidence_type"], "hybrid")
        self.assertEqual(len(evidence[0]["provenance"]), 2)
        self.assertEqual(
            evidence[0]["sources"], ["xywy_disease_encyclopedia", "xywy"]
        )
        self.assertEqual(stats["duplicates_merged"], 1)

    def test_duplicate_vector_chunks_do_not_consume_two_slots(self) -> None:
        evidence, stats = fuse_evidence(
            [
                vector_hit("v1", "完全相同的文本"),
                vector_hit("v2", "完全相同的文本", score=0.8),
            ],
            [],
            top_k=5,
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(len(evidence[0]["provenance"]), 2)
        self.assertEqual(evidence[0]["retrieval_score"], 0.9)
        self.assertEqual(stats["duplicates_merged"], 1)

    def test_allow_partial_keeps_graph_evidence_and_reports_error(self) -> None:
        result = hybrid_search(
            "百日咳有哪些症状",
            "百日咳",
            vector_backend=FailingVectorBackend(),
            graph_backend=FakeGraphBackend([graph_hit("痉挛性咳嗽")]),
            relations=["has_symptom"],
            allow_partial=True,
        )

        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["evidence_type"], "graph")
        self.assertIn("ConnectionError", result["errors"]["vector"])

    def test_invalid_fusion_configuration_is_rejected(self) -> None:
        with self.assertRaises(HybridRetrievalError):
            fuse_evidence([], [], top_k=0)
        with self.assertRaises(HybridRetrievalError):
            fuse_evidence([], [], graph_weight=0)


if __name__ == "__main__":
    unittest.main()
