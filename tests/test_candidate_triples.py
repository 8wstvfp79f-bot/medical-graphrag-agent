from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_candidate_triples import build_candidate_triples


class CandidateTripleTests(unittest.TestCase):
    def test_candidate_relations_are_generated_in_a_separate_file(self) -> None:
        record = {
            "name": "示例疾病",
            "desc": "用于测试的疾病介绍。",
            "cure_way": ["药物治疗"],
            "do_eat": ["苹果"],
            "recommand_eat": ["苹果", "梨"],
            "not_eat": ["辣椒"],
            "drug_detail": ["示例品牌(示例药)"],
            "cause": "感染因素。遗传因素；",
            "prevent": "保持通风。注意卫生。",
            "get_way": "呼吸道传播",
            "cure_lasttime": "约两周",
            "cost_money": "约100元",
            "cured_prob": "约90%",
            "get_prob": "0.1%",
            "easy_get": "儿童",
            "yibao_status": "是",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "medical.json"
            output = root / "candidate_triples.csv"
            raw.write_text(
                json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            stats = build_candidate_triples(raw, output)
            with output.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(stats["valid_records"], 1)
        self.assertEqual(stats["candidate_triples"], len(rows))
        self.assertGreaterEqual(len(rows), 14)
        self.assertEqual({row["quality_tier"] for row in rows}, {"candidate"})
        self.assertEqual(
            len(
                [
                    row
                    for row in rows
                    if row["relation"] == "recommended_food"
                    and row["tail"] == "苹果"
                ]
            ),
            1,
        )
        self.assertIn("caused_by", {row["relation"] for row in rows})
        self.assertIn("has_brand_drug", {row["relation"] for row in rows})


if __name__ == "__main__":
    unittest.main()
