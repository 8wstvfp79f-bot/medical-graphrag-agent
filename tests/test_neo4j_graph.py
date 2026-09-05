from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.retrieval.graph_retriever import Neo4jGraphBackend, graph_search
from src.retrieval.neo4j_client import (
    MedicalNeo4jStore,
    Neo4jSettings,
    validate_triples,
)


class FakeNeo4jDriver:
    def __init__(self) -> None:
        self.nodes: dict[str, set[str]] = {}
        self.relationships: dict[tuple[str, str, str], str] = {}
        self.constraints: set[str] = set()
        self.calls: list[tuple[str, dict[str, object], str]] = []
        self.verified = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def execute_query(
        self,
        query: str,
        *,
        parameters_: dict[str, object],
        database_: str,
    ) -> SimpleNamespace:
        self.calls.append((query, parameters_, database_))
        constraint_match = re.search(
            r"medical_graph:constraint:([A-Za-z]+)", query
        )
        if constraint_match:
            self.constraints.add(constraint_match.group(1))
            return SimpleNamespace(records=[])

        ingest_match = re.search(
            r"medical_graph:ingest:([A-Z_]+):([A-Za-z]+)", query
        )
        if ingest_match:
            relationship_type, tail_label = ingest_match.groups()
            rows = parameters_["rows"]
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, dict)
                head = str(row["head"])
                tail = str(row["tail"])
                source = str(row["source"])
                self.nodes.setdefault("Disease", set()).add(head)
                self.nodes.setdefault(tail_label, set()).add(tail)
                self.relationships[(head, relationship_type, tail)] = source
            return SimpleNamespace(records=[{"processed": len(rows)}])

        prune_match = re.search(
            r"medical_graph:prune:([A-Z_]+):([A-Za-z]+)", query
        )
        if prune_match:
            relationship_type, _ = prune_match.groups()
            source = str(parameters_["source"])
            values = {str(value) for value in parameters_["values"]}
            matched = [
                key
                for key, relationship_source in self.relationships.items()
                if key[1] == relationship_type
                and key[2] in values
                and relationship_source == source
            ]
            for key in matched:
                del self.relationships[key]
            return SimpleNamespace(
                records=[{"deleted_relationships": len(matched)}]
            )

        orphan_match = re.search(
            r"medical_graph:prune_orphans:([A-Za-z]+)", query
        )
        if orphan_match:
            label = orphan_match.group(1)
            values = {str(value) for value in parameters_["values"]}
            connected = {
                name
                for head, _, tail in self.relationships
                for name in (head, tail)
            }
            orphans = {
                value
                for value in values
                if value in self.nodes.get(label, set()) and value not in connected
            }
            self.nodes.setdefault(label, set()).difference_update(orphans)
            return SimpleNamespace(records=[{"deleted_orphan_nodes": len(orphans)}])

        if "medical_graph:count:nodes" in query:
            return SimpleNamespace(
                records=[
                    {"label": label, "count": len(names)}
                    for label, names in sorted(self.nodes.items())
                ]
            )

        if "medical_graph:count:relationships" in query:
            counts: dict[str, int] = {}
            for _, relationship_type, _ in self.relationships:
                counts[relationship_type] = counts.get(relationship_type, 0) + 1
            return SimpleNamespace(
                records=[
                    {"relationship_type": relationship_type, "count": count}
                    for relationship_type, count in sorted(counts.items())
                ]
            )

        if "medical_graph:search" in query:
            disease_name = str(parameters_["disease_name"])
            relationship_types = parameters_["relationship_types"]
            limit = int(parameters_["limit"])
            assert isinstance(relationship_types, list)
            tail_labels = {
                "HAS_SYMPTOM": "Symptom",
                "DIAGNOSED_BY": "Check",
                "TREATED_BY": "Drug",
                "BELONGS_TO": "Department",
                "HAS_COMPLICATION": "Complication",
            }
            records = []
            for (head, relationship_type, tail), source in sorted(
                self.relationships.items()
            ):
                if head != disease_name or relationship_type not in relationship_types:
                    continue
                records.append(
                    {
                        "disease_name": head,
                        "relationship_type": relationship_type,
                        "entity_name": tail,
                        "entity_type": tail_labels[relationship_type],
                        "source": source,
                    }
                )
            return SimpleNamespace(records=records[:limit])
        raise AssertionError(f"Unexpected Cypher query: {query}")


def write_triples(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["head", "relation", "tail", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)


class Neo4jGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = FakeNeo4jDriver()
        self.settings = Neo4jSettings(
            uri="bolt://fake:7687",
            username="neo4j",
            password="test-password",
        )
        self.store = MedicalNeo4jStore(self.driver, self.settings)

    def test_schema_and_ingestion_are_idempotent(self) -> None:
        rows = [
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "痉挛性咳嗽",
                "source": "xywy",
            },
            {
                "head": "百日咳",
                "relation": "belongs_to",
                "tail": "儿科",
                "source": "xywy",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triples.csv"
            write_triples(path, rows)
            first = self.store.ingest_triples(path, batch_size=1)
            second = self.store.ingest_triples(path, batch_size=2)

        self.assertEqual(len(self.driver.constraints), 6)
        self.assertEqual(first.node_count_before, 0)
        self.assertEqual(first.node_count_after, 3)
        self.assertEqual(first.relationship_count_after, 2)
        self.assertEqual(second.relationship_count_before, 2)
        self.assertEqual(second.relationship_count_after, 2)
        self.assertEqual(len(self.driver.relationships), 2)

    def test_candidate_relation_is_counted_but_not_imported(self) -> None:
        rows = [
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "咳嗽",
                "source": "xywy",
            },
            {
                "head": "百日咳",
                "relation": "caused_by",
                "tail": "百日咳杆菌",
                "source": "xywy",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triples.csv"
            write_triples(path, rows)
            validation = validate_triples(path)
            report = self.store.ingest_triples(path)

        self.assertEqual(validation.scanned_records, 2)
        self.assertEqual(validation.core_records, 1)
        self.assertEqual(validation.skipped_candidate_records, 1)
        self.assertEqual(report.submitted_core_records, 1)
        self.assertEqual(report.skipped_candidate_records, 1)
        self.assertEqual(report.relationship_count_after, 1)

    def test_only_confirmed_audit_items_are_pruned(self) -> None:
        rows = [
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "痉挛性咳嗽",
                "source": "xywy",
            },
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "闫鹏辉",
                "source": "xywy",
            },
        ]
        audit_rows = [
            {
                "disease_name": "百日咳",
                "field": "symptom",
                "value": "闫鹏辉",
                "reason": "confirmed_non_medical_entity",
                "source": "xywy",
            },
            {
                "disease_name": "百日咳",
                "field": "symptom",
                "value": "痉挛性咳嗽",
                "reason": "duplicate_after_normalization",
                "source": "xywy",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            triples_path = root / "triples.csv"
            audit_path = root / "rejected.jsonl"
            write_triples(triples_path, rows)
            audit_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in audit_rows)
                + "\n",
                encoding="utf-8",
            )
            self.store.ingest_triples(triples_path)
            report = self.store.prune_rejected_items(audit_path)

        self.assertEqual(report.rejected_values, 1)
        self.assertEqual(report.deleted_relationships, 1)
        self.assertEqual(report.deleted_orphan_nodes, 1)
        self.assertIn(
            ("百日咳", "HAS_SYMPTOM", "痉挛性咳嗽"),
            self.driver.relationships,
        )
        self.assertNotIn(
            ("百日咳", "HAS_SYMPTOM", "闫鹏辉"),
            self.driver.relationships,
        )

    def test_graph_search_returns_typed_evidence_and_relation_filter(self) -> None:
        rows = [
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "痉挛性咳嗽",
                "source": "xywy",
            },
            {
                "head": "百日咳",
                "relation": "belongs_to",
                "tail": "儿科",
                "source": "xywy",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triples.csv"
            write_triples(path, rows)
            self.store.ingest_triples(path)

        backend = Neo4jGraphBackend(self.driver)
        evidence = graph_search(
            "百日咳",
            backend=backend,
            relations=["has_symptom"],
            limit=5,
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["relation"], "has_symptom")
        self.assertEqual(evidence[0]["entity_type"], "Symptom")
        self.assertEqual(
            evidence[0]["path_text"], "百日咳 -[has_symptom]-> 痉挛性咳嗽"
        )

    def test_unknown_relation_is_rejected_before_query(self) -> None:
        backend = Neo4jGraphBackend(self.driver)
        with self.assertRaises(ValueError):
            graph_search("百日咳", backend=backend, relations=["caused_by"])

    def test_missing_core_field_reports_csv_line(self) -> None:
        rows = [
            {
                "head": "百日咳",
                "relation": "has_symptom",
                "tail": "",
                "source": "xywy",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "triples.csv"
            write_triples(path, rows)
            with self.assertRaisesRegex(ValueError, r"triples.csv:2"):
                validate_triples(path)


if __name__ == "__main__":
    unittest.main()
