from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.retrieval.milvus_client import (
    MedicalMilvusStore,
    MilvusCollectionError,
    MilvusSettings,
    validate_chunks_for_milvus,
)


class FakeDataTypes:
    VARCHAR = "VARCHAR"
    JSON = "JSON"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict[str, object]] = []

    def add_field(self, **kwargs: object) -> None:
        self.fields.append(dict(kwargs))


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, object]] = []

    def add_index(self, **kwargs: object) -> None:
        self.indexes.append(dict(kwargs))


class FakeQueryIterator:
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        output_fields: list[str],
        batch_size: int,
    ) -> None:
        self.rows = rows
        self.output_fields = output_fields
        self.batch_size = batch_size
        self.offset = 0
        self.closed = False

    def next(self) -> list[dict[str, object]]:
        batch = self.rows[self.offset : self.offset + self.batch_size]
        self.offset += len(batch)
        return [
            {field: entity[field] for field in self.output_fields}
            for entity in batch
        ]

    def close(self) -> None:
        self.closed = True


class FakeMilvusClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.last_search: dict[str, object] = {}

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_schema(self, **_: object) -> FakeSchema:
        return FakeSchema()

    def prepare_index_params(self) -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: FakeSchema,
        index_params: FakeIndexParams,
    ) -> None:
        self.collections[collection_name] = {
            "schema": schema,
            "index_params": index_params,
            "rows": {},
            "storage_row_count": 0,
            "loaded": False,
        }

    def drop_collection(self, *, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def describe_collection(self, *, collection_name: str) -> dict[str, object]:
        schema = self.collections[collection_name]["schema"]
        assert isinstance(schema, FakeSchema)
        fields: list[dict[str, object]] = []
        for field in schema.fields:
            fields.append(
                {
                    "field_name": field["field_name"],
                    "params": {"dim": field["dim"]} if "dim" in field else {},
                }
            )
        return {"fields": fields}

    def load_collection(self, *, collection_name: str, timeout: float) -> None:
        self.collections[collection_name]["loaded"] = True
        self.collections[collection_name]["load_timeout"] = timeout

    def upsert(
        self,
        *,
        collection_name: str,
        data: list[dict[str, object]],
    ) -> dict[str, int]:
        rows = self.collections[collection_name]["rows"]
        assert isinstance(rows, dict)
        for entity in data:
            rows[str(entity["chunk_id"])] = dict(entity)
        storage_row_count = self.collections[collection_name]["storage_row_count"]
        assert isinstance(storage_row_count, int)
        self.collections[collection_name]["storage_row_count"] = (
            storage_row_count + len(data)
        )
        return {"upsert_count": len(data)}

    def flush(self, *, collection_name: str) -> None:
        if collection_name not in self.collections:
            raise KeyError(collection_name)

    def get_collection_stats(self, *, collection_name: str) -> dict[str, str]:
        storage_row_count = self.collections[collection_name]["storage_row_count"]
        assert isinstance(storage_row_count, int)
        return {"row_count": str(storage_row_count)}

    def query_iterator(
        self,
        *,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        batch_size: int,
        consistency_level: str,
    ) -> FakeQueryIterator:
        del filter, consistency_level
        rows = self.collections[collection_name]["rows"]
        assert isinstance(rows, dict)
        return FakeQueryIterator(
            list(rows.values()),
            output_fields=output_fields,
            batch_size=batch_size,
        )

    def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
        self.last_search = dict(kwargs)
        collection_name = str(kwargs["collection_name"])
        vectors = kwargs["data"]
        assert isinstance(vectors, list)
        query_vector = vectors[0]
        assert isinstance(query_vector, list)
        limit = int(kwargs["limit"])
        output_fields = kwargs["output_fields"]
        assert isinstance(output_fields, list)
        rows = self.collections[collection_name]["rows"]
        assert isinstance(rows, dict)

        hits: list[dict[str, object]] = []
        for entity in rows.values():
            assert isinstance(entity, dict)
            vector = entity["embedding"]
            assert isinstance(vector, list)
            numerator = sum(a * b for a, b in zip(query_vector, vector))
            denominator = math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(
                sum(b * b for b in vector)
            )
            score = numerator / denominator if denominator else 0.0
            payload = {field: entity[field] for field in output_fields}
            hits.append({"distance": score, "entity": payload})
        hits.sort(key=lambda hit: float(hit["distance"]), reverse=True)
        return [hits[:limit]]


class TinyEmbeddingModel:
    model_name = "tiny-test-embedding"
    dimension = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "普通咳嗽" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "百日咳" in text:
                vectors.append([0.9, 0.1, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def chunk(
    chunk_id: str,
    text: str,
    disease_name: str,
    department: str,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"doc_{chunk_id}",
        "disease_name": disease_name,
        "text": text,
        "metadata": {
            "disease_name": disease_name,
            "category": ["疾病百科"],
            "department": [department],
            "source": "xywy_disease_encyclopedia",
        },
    }


class MilvusClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeMilvusClient()
        self.settings = MilvusSettings(
            uri="fake://milvus",
            collection_name="medical_test",
            dimension=3,
        )
        self.store = MedicalMilvusStore(
            self.client,
            self.settings,
            data_types=FakeDataTypes,
        )
        self.model = TinyEmbeddingModel()

    def test_collection_schema_and_index_are_created_once(self) -> None:
        self.assertTrue(self.store.ensure_collection())
        self.assertFalse(self.store.ensure_collection())

        collection = self.client.collections["medical_test"]
        schema = collection["schema"]
        index_params = collection["index_params"]
        self.assertIsInstance(schema, FakeSchema)
        self.assertIsInstance(index_params, FakeIndexParams)
        self.assertEqual(collection["load_timeout"], 60.0)
        assert isinstance(schema, FakeSchema)
        assert isinstance(index_params, FakeIndexParams)
        self.assertEqual(
            [field["field_name"] for field in schema.fields],
            [
                "chunk_id",
                "doc_id",
                "disease_name",
                "text",
                "metadata",
                "embedding_model",
                "embedding",
            ],
        )
        self.assertEqual(index_params.indexes[0]["metric_type"], "COSINE")
        self.assertEqual(index_params.indexes[0]["index_type"], "AUTOINDEX")

    def test_batch_ingestion_is_idempotent_by_chunk_id(self) -> None:
        records = [
            chunk("wrong", "普通咳嗽说明", "咳嗽", "呼吸内科"),
            chunk("right", "百日咳证据", "百日咳", "儿科"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in records
                )
                + "\n",
                encoding="utf-8",
            )
            first = self.store.ingest_chunks(path, self.model, batch_size=1)
            second = self.store.ingest_chunks(path, self.model, batch_size=2)

        self.assertTrue(first.collection_created)
        self.assertEqual(first.attempted_records, 2)
        self.assertEqual(first.batches, 2)
        self.assertEqual(first.logical_entity_count_before, 0)
        self.assertEqual(first.logical_entity_count_after, 2)
        self.assertEqual(first.storage_row_count_before, 0)
        self.assertEqual(first.storage_row_count_after, 2)
        self.assertFalse(second.collection_created)
        self.assertEqual(second.logical_entity_count_before, 2)
        self.assertEqual(second.logical_entity_count_after, 2)
        self.assertEqual(second.storage_row_count_before, 2)
        self.assertEqual(second.storage_row_count_after, 4)
        self.assertEqual(self.store.entity_count(), 2)

    def test_search_filters_a_higher_scoring_wrong_disease(self) -> None:
        records = [
            chunk("wrong", "普通咳嗽说明", "咳嗽", "呼吸内科"),
            chunk("right", "百日咳证据", "百日咳", "儿科"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in records
                )
                + "\n",
                encoding="utf-8",
            )
            self.store.ingest_chunks(path, self.model, batch_size=2)

        hits = self.store.search(
            "百日咳有什么症状",
            self.model,
            top_k=1,
            metadata_filter={"disease_name": "百日咳", "department": "儿科"},
        )

        self.assertEqual([hit["chunk_id"] for hit in hits], ["right"])
        self.assertEqual(hits[0]["matched_metadata"]["disease_name"], "百日咳")
        self.assertIn("filter", self.client.last_search)
        self.assertIn("json_contains_any", str(self.client.last_search["filter"]))
        self.assertEqual(
            self.client.last_search["search_params"],
            {"metric_type": "COSINE", "params": {}},
        )

    def test_full_payload_validation_needs_no_milvus_client(self) -> None:
        records = [
            chunk("one", "百日咳证据", "百日咳", "儿科"),
            chunk("two", "普通咳嗽说明", "咳嗽", "呼吸内科"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.jsonl"
            path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
                + "\n",
                encoding="utf-8",
            )
            report = validate_chunks_for_milvus(
                path,
                self.model,
                self.settings,
                batch_size=1,
            )

        self.assertEqual(report.validated_records, 2)
        self.assertEqual(report.batches, 2)
        self.assertGreater(report.max_text_bytes, 0)

    def test_existing_collection_dimension_mismatch_is_rejected(self) -> None:
        self.store.ensure_collection()
        incompatible_store = MedicalMilvusStore(
            self.client,
            MilvusSettings(
                uri="fake://milvus",
                collection_name="medical_test",
                dimension=4,
            ),
            data_types=FakeDataTypes,
        )
        with self.assertRaises(MilvusCollectionError):
            incompatible_store.ensure_collection()

    def test_collection_load_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "load_timeout_seconds"):
            MilvusSettings(load_timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
