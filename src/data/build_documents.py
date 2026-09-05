from __future__ import annotations

import json
import sys
from pathlib import Path

from src.data.medical_quality import (
    DEFAULT_REJECTION_AUDIT_PATH,
    MedicalQualityPolicy,
    RejectedMedicalItem,
    clean_text,
    load_default_quality_policy,
    normalize_items,
    sanitize_medical_record,
    write_rejection_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "xywy" / "medical.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"
INVALID_TITLE = "无标题文档"
SOURCE_NAME = "xywy_disease_encyclopedia"


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def join_items(value: object) -> str:
    return "、".join(normalize_items(value))


def is_valid_record(record: dict) -> bool:
    name = clean_text(record.get("name"))
    desc = clean_text(record.get("desc"))
    return bool(name) and name != INVALID_TITLE and bool(desc)


def iter_json_records(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_text(record: dict) -> str:
    sections = [
        ("疾病名称", clean_text(record.get("name"))),
        ("疾病简介", clean_text(record.get("desc"))),
        ("所属分类", join_items(record.get("category"))),
        ("病因", join_items(record.get("cause"))),
        ("预防", join_items(record.get("prevent"))),
        ("症状", join_items(record.get("symptom"))),
        ("检查", join_items(record.get("check"))),
        ("治疗科室", join_items(record.get("cure_department"))),
        ("治疗方式", join_items(record.get("cure_way"))),
        ("常用药品", join_items(record.get("common_drug"))),
        ("推荐药品", join_items(record.get("recommand_drug"))),
        ("并发症", join_items(record.get("acompany"))),
    ]
    return "\n".join(f"{label}：{value}" for label, value in sections if value)


def build_document(
    record: dict,
    index: int,
    *,
    quality_policy: MedicalQualityPolicy | None = None,
    rejected_items: list[RejectedMedicalItem] | None = None,
) -> dict:
    sanitized, rejected = sanitize_medical_record(
        record,
        policy=quality_policy,
        source=SOURCE_NAME,
    )
    if rejected_items is not None:
        rejected_items.extend(rejected)
    disease_name = clean_text(sanitized.get("name"))
    return {
        "doc_id": f"disease_{index:06d}",
        "disease_name": disease_name,
        "source": SOURCE_NAME,
        "text": build_text(sanitized),
        "metadata": {
            "disease_name": disease_name,
            "category": normalize_items(sanitized.get("category")),
            "department": normalize_items(sanitized.get("cure_department")),
            "source": SOURCE_NAME,
        },
    }


def main() -> None:
    configure_stdout()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    quality_policy = load_default_quality_policy()
    rejected_items: list[RejectedMedicalItem] = []
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as output:
        for record in iter_json_records(RAW_PATH):
            if not is_valid_record(record):
                continue

            count += 1
            document = build_document(
                record,
                count,
                quality_policy=quality_policy,
                rejected_items=rejected_items,
            )
            output.write(json.dumps(document, ensure_ascii=False) + "\n")

    rejected_count = write_rejection_audit(
        DEFAULT_REJECTION_AUDIT_PATH,
        rejected_items,
    )
    print(f"Generated {count} documents: {OUTPUT_PATH}")
    print(
        f"Audited {rejected_count} rejected medical items: "
        f"{DEFAULT_REJECTION_AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()
