from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.stat_relation_scale import calculate_stats


class RelationStatsTests(unittest.TestCase):
    def test_core_and_candidate_relations_are_counted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            documents = root / "documents.jsonl"
            triples = root / "triples.csv"
            documents.write_text(
                "\n".join(
                    json.dumps({"doc_id": doc_id}) for doc_id in ("d1", "d2")
                )
                + "\n",
                encoding="utf-8",
            )
            with triples.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["head", "relation", "tail", "source"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "head": "百日咳",
                            "relation": "has_symptom",
                            "tail": "咳嗽",
                            "source": "xywy",
                        },
                        {
                            "head": "百日咳",
                            "relation": "recommended_food",
                            "tail": "梨",
                            "source": "xywy",
                        },
                    ]
                )

            stats = calculate_stats(documents, triples)

        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["core_triples"], 1)
        self.assertEqual(stats["candidate_triples"], 1)
        self.assertEqual(stats["core_entities"], 2)
        self.assertEqual(stats["all_entities"], 3)

    def test_separate_candidate_csv_can_be_combined_without_becoming_core(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            documents = root / "documents.jsonl"
            core = root / "triples.csv"
            candidates = root / "candidate_triples.csv"
            documents.write_text('{"doc_id":"d1"}\n', encoding="utf-8")
            for path, relation, tail in (
                (core, "has_symptom", "咳嗽"),
                (candidates, "prevented_by", "保持通风"),
            ):
                with path.open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=["head", "relation", "tail", "source"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "head": "百日咳",
                            "relation": relation,
                            "tail": tail,
                            "source": "xywy",
                        }
                    )

            stats = calculate_stats(documents, core, (candidates,))

        self.assertEqual(stats["core_triples"], 1)
        self.assertEqual(stats["candidate_triples"], 1)
        self.assertEqual(stats["total_triples"], 2)
        self.assertEqual(stats["core_entities"], 2)
        self.assertEqual(stats["all_entities"], 3)


if __name__ == "__main__":
    unittest.main()
