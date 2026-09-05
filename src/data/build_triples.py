from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from src.data.medical_quality import (
    DEFAULT_REJECTION_AUDIT_PATH,
    RejectedMedicalItem,
    clean_text,
    load_default_quality_policy,
    normalize_items,
    sanitize_medical_record,
    write_rejection_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "xywy" / "medical.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "triples.csv"
INVALID_TITLE = "无标题文档"
SOURCE_NAME = "xywy"
CONTENT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
MEANINGFUL_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
PLACEHOLDERS = {
    "无",
    "暂无",
    "未知",
    "不详",
    "其他",
    "none",
    "null",
    "nan",
    "[]",
}

FIELD_RELATIONS = [
    ("symptom", "has_symptom"),
    ("check", "diagnosed_by"),
    ("recommand_drug", "treated_by"),
    ("common_drug", "treated_by"),
    ("cure_department", "belongs_to"),
    ("acompany", "has_complication"),
]


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def is_valid_record(record: dict) -> bool:
    name = clean_text(record.get("name"))
    desc = clean_text(record.get("desc"))
    return bool(name) and name != INVALID_TITLE and bool(desc)


def is_valid_tail(value: str) -> bool:
    text = clean_text(value)
    if not text:
        return False

    compact = text.strip(" \t\r\n,，.。;；:：、|/\\()（）[]【】{}<>《》-_=+*#@!！?？'\"")
    if not compact:
        return False

    if compact.lower() in PLACEHOLDERS:
        return False

    if not CONTENT_RE.search(compact):
        return False

    meaningful_chars = MEANINGFUL_RE.findall(compact)
    return len(meaningful_chars) >= 2


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


def main() -> None:
    configure_stdout()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, str]] = []
    quality_policy = load_default_quality_policy()
    rejected_items: list[RejectedMedicalItem] = []

    for record in iter_json_records(RAW_PATH):
        if not is_valid_record(record):
            continue

        record, rejected = sanitize_medical_record(
            record,
            policy=quality_policy,
            source=SOURCE_NAME,
        )
        rejected_items.extend(rejected)

        head = clean_text(record.get("name"))
        for field, relation in FIELD_RELATIONS:
            for tail in normalize_items(record.get(field)):
                tail = clean_text(tail)
                if not is_valid_tail(tail):
                    continue

                key = (head, relation, tail, SOURCE_NAME)
                if key in seen:
                    continue

                seen.add(key)
                rows.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "source": SOURCE_NAME,
                    }
                )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["head", "relation", "tail", "source"])
        writer.writeheader()
        writer.writerows(rows)

    rejected_count = write_rejection_audit(
        DEFAULT_REJECTION_AUDIT_PATH,
        rejected_items,
    )
    print(f"Generated {len(rows)} triples: {OUTPUT_PATH}")
    print(
        f"Audited {rejected_count} rejected medical items: "
        f"{DEFAULT_REJECTION_AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()
