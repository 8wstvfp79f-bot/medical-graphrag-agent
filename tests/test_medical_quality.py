from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.build_documents import build_document
from src.data.medical_quality import (
    MedicalQualityPolicy,
    load_default_quality_policy,
    write_rejection_audit,
)


class MedicalQualityTests(unittest.TestCase):
    def test_confirmed_non_medical_symptom_is_rejected_at_document_boundary(self) -> None:
        rejected = []
        document = build_document(
            {
                "name": "百日咳",
                "desc": "一种呼吸道传染病",
                "symptom": ["痉挛性咳嗽", "闫鹏辉", "低热"],
            },
            1,
            rejected_items=rejected,
        )

        self.assertIn("痉挛性咳嗽", document["text"])
        self.assertNotIn("闫鹏辉", document["text"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].field, "symptom")
        self.assertIn("non_medical_entity", rejected[0].reason)

    def test_policy_supports_alias_normalization_and_deduplication(self) -> None:
        policy = MedicalQualityPolicy(
            aliases={"symptom": {"鸡鸣样吸气声": "吸气性鸡鸣样吼声"}}
        )

        accepted, rejected = policy.filter_items(
            "symptom",
            ["鸡鸣样吸气声", "吸气性鸡鸣样吼声"],
            disease_name="百日咳",
            source="test",
        )

        self.assertEqual(accepted, ["吸气性鸡鸣样吼声"])
        self.assertEqual([item.reason for item in rejected], ["duplicate_after_normalization"])

    def test_rejection_audit_is_deterministic_and_deduplicated(self) -> None:
        policy = load_default_quality_policy()
        _, rejected = policy.filter_items(
            "symptom",
            ["闫鹏辉", "闫鹏辉"],
            disease_name="百日咳",
            source="xywy",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rejected.jsonl"
            count = write_rejection_audit(path, rejected)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["value"], "闫鹏辉")
        self.assertEqual(rows[0]["disease_name"], "百日咳")


if __name__ == "__main__":
    unittest.main()
