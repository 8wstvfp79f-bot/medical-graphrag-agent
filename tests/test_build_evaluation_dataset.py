from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluation.build_dataset import build_evaluation_dataset
from src.evaluation.schemas import INTENT_RELATIONS, load_evaluation_cases


class BuildEvaluationDatasetTests(unittest.TestCase):
    def test_builds_five_intents_per_eligible_disease_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            triples = root / "triples.csv"
            output = root / "eval.jsonl"
            with triples.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["head", "relation", "tail", "source"],
                )
                writer.writeheader()
                for disease in ("甲病", "乙病"):
                    for relation in INTENT_RELATIONS.values():
                        for index in range(1, 5):
                            writer.writerow(
                                {
                                    "head": disease,
                                    "relation": relation,
                                    "tail": f"{relation}-{index}",
                                    "source": "xywy",
                                }
                            )

            first = build_evaluation_dataset(
                triples, output, disease_count=2, seed="fixed"
            )
            first_text = output.read_text(encoding="utf-8")
            second = build_evaluation_dataset(
                triples, output, disease_count=2, seed="fixed"
            )
            cases = load_evaluation_cases(output)

        self.assertEqual(first, second)
        self.assertEqual(first_text.count("\n"), 10)
        self.assertEqual(len(cases), 10)
        self.assertEqual(first["gold_entity_references"], 30)
        self.assertEqual({case.intent for case in cases}, set(INTENT_RELATIONS))


if __name__ == "__main__":
    unittest.main()
