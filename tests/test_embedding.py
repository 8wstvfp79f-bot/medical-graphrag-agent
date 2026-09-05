from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.retrieval.embedding import (
    HashEmbeddingModel,
    create_embedding_model,
    embed_chunk_records,
)


class EmbeddingTests(unittest.TestCase):
    def test_hash_embedding_is_deterministic_normalized_and_fixed_size(self) -> None:
        first_model = HashEmbeddingModel(dimension=64)
        second_model = HashEmbeddingModel(dimension=64)
        texts = ["百日咳常见痉挛性咳嗽", "需要进行血常规检查"]

        first_vectors = first_model.embed_texts(texts)
        second_vectors = second_model.embed_texts(texts)

        self.assertEqual(first_vectors, second_vectors)
        self.assertEqual([len(vector) for vector in first_vectors], [64, 64])
        for vector in first_vectors:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0)
        self.assertNotEqual(first_vectors[0], first_vectors[1])

    def test_embed_query_matches_single_text_embedding(self) -> None:
        model = HashEmbeddingModel(dimension=48)
        query = "百日咳有哪些症状"
        self.assertEqual(model.embed_query(query), model.embed_texts([query])[0])

    def test_chunk_batch_embedding_preserves_metadata(self) -> None:
        model = HashEmbeddingModel(dimension=32)
        chunks = [
            {
                "chunk_id": "d1_chunk_001",
                "doc_id": "d1",
                "text": "百日咳症状",
                "metadata": {
                    "disease_name": "百日咳",
                    "category": ["疾病百科"],
                    "department": ["儿科"],
                    "source": "xywy_disease_encyclopedia",
                },
            },
            {
                "chunk_id": "d2_chunk_001",
                "doc_id": "d2",
                "text": "苯中毒检查",
                "metadata": {
                    "disease_name": "苯中毒",
                    "category": ["疾病百科"],
                    "department": ["急诊科"],
                    "source": "xywy_disease_encyclopedia",
                },
            },
        ]

        embedded = list(embed_chunk_records(chunks, model, batch_size=1))

        self.assertEqual(len(embedded), 2)
        self.assertEqual(embedded[0]["metadata"], chunks[0]["metadata"])
        self.assertEqual(len(embedded[0]["embedding"]), 32)
        self.assertEqual(embedded[0]["embedding_metadata"]["model"], "hash-embedding-v1")
        self.assertTrue(embedded[0]["embedding_metadata"]["normalized"])

    def test_empty_text_and_invalid_configuration_are_rejected(self) -> None:
        model = HashEmbeddingModel(dimension=32)
        with self.assertRaises(ValueError):
            model.embed_texts([" "])
        with self.assertRaises(ValueError):
            HashEmbeddingModel(dimension=16)
        with self.assertRaises(ValueError):
            create_embedding_model("bge")

    def test_local_backend_rejects_missing_model_directory_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing-bge-model"
            with self.assertRaises(FileNotFoundError):
                create_embedding_model("bge", model_path=missing_path)


if __name__ == "__main__":
    unittest.main()
