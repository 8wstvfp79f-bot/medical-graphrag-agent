from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_chunks import build_chunks
from src.data.build_documents import SOURCE_NAME, build_document
from src.retrieval.metadata_filter import (
    MetadataFilterError,
    build_milvus_filter_expression,
)
from src.retrieval.vector_retriever import vector_search


class FakeBackend:
    def __init__(self, candidates: list[dict[str, object]]) -> None:
        self.candidates = candidates
        self.last_top_k = 0
        self.last_filter: dict[str, tuple[str, ...]] = {}

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, tuple[str, ...]],
    ) -> list[dict[str, object]]:
        self.last_top_k = top_k
        self.last_filter = metadata_filter
        return self.candidates[:top_k]


class MetadataPipelineTests(unittest.TestCase):
    def test_document_and_chunk_metadata_have_required_fields(self) -> None:
        document = build_document(
            {
                "name": "百日咳",
                "desc": "一种呼吸道传染病",
                "category": ["疾病百科", "传染科"],
                "cure_department": ["传染科"],
            },
            1,
        )
        self.assertEqual(document["metadata"]["disease_name"], "百日咳")
        self.assertEqual(document["metadata"]["source"], SOURCE_NAME)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "documents.jsonl"
            output_path = Path(temp_dir) / "chunks.jsonl"
            input_path.write_text(
                json.dumps(document, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            count = build_chunks(input_path, output_path, max_chars=40, overlap_chars=5)
            chunks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, len(chunks))
        self.assertGreater(count, 0)
        self.assertEqual(
            set(chunks[0]["metadata"]),
            {"disease_name", "category", "department", "source"},
        )

    def test_vector_search_filters_similar_disease_bad_case(self) -> None:
        candidates = [
            {
                "chunk_id": "pertussis",
                "doc_id": "d1",
                "text": "百日咳常见阵发性痉挛性咳嗽。",
                "score": 0.86,
                "metadata": {
                    "disease_name": "百日咳",
                    "category": ["疾病百科", "传染科"],
                    "department": ["传染科"],
                    "source": "xywy_disease_encyclopedia",
                },
            },
            {
                "chunk_id": "cough",
                "doc_id": "d2",
                "text": "咳嗽可能由多种原因引起。",
                "score": 0.93,
                "metadata": {
                    "disease_name": "咳嗽",
                    "category": ["疾病百科", "呼吸内科"],
                    "department": ["呼吸内科"],
                    "source": "xywy_disease_encyclopedia",
                },
            },
        ]
        backend = FakeBackend(candidates)

        hits = vector_search(
            "百日咳有什么症状",
            backend=backend,
            top_k=3,
            metadata_filter={"disease_name": "百日咳", "department": "传染科"},
        )

        self.assertEqual([hit["chunk_id"] for hit in hits], ["pertussis"])
        self.assertEqual(hits[0]["matched_metadata"]["disease_name"], "百日咳")
        self.assertEqual(hits[0]["matched_metadata"]["department"], ["传染科"])
        self.assertEqual(backend.last_filter["disease_name"], ("百日咳",))
        self.assertEqual(backend.last_top_k, 12)

    def test_milvus_expression_and_unknown_field_validation(self) -> None:
        expression = build_milvus_filter_expression(
            {"disease_name": "百日咳", "category": ["疾病百科", "传染科"]}
        )
        self.assertIn('metadata["disease_name"] in ["百日咳"]', expression)
        self.assertIn("json_contains_any", expression)

        with self.assertRaises(MetadataFilterError):
            build_milvus_filter_expression({"model_number": "ABC-100"})


if __name__ == "__main__":
    unittest.main()
