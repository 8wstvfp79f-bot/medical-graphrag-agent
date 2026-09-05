from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "xywy" / "medical.json"
INVALID_TITLE = "无标题文档"


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("\u3000", " ").split())


def normalize_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(normalize_items(item))
        return items

    text = clean_text(value)
    return [text] if text else []


def has_value(value: object) -> bool:
    return bool(normalize_items(value))


def is_valid_record(record: dict) -> bool:
    name = clean_text(record.get("name"))
    desc = clean_text(record.get("desc"))
    return bool(name) and name != INVALID_TITLE and bool(desc)


def iter_json_records(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, exc


def main() -> None:
    configure_stdout()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    stats = {
        "total_records": 0,
        "valid_records": 0,
        "name_non_empty": 0,
        "desc_non_empty": 0,
        "symptom_non_empty": 0,
        "check_non_empty": 0,
        "recommand_drug_non_empty": 0,
    }
    parse_errors = 0
    samples: list[dict] = []

    for line_no, record, error in iter_json_records(RAW_PATH):
        stats["total_records"] += 1
        if error is not None:
            parse_errors += 1
            print(f"JSON parse error at line {line_no}: {error}")
            continue

        if has_value(record.get("name")):
            stats["name_non_empty"] += 1
        if has_value(record.get("desc")):
            stats["desc_non_empty"] += 1
        if has_value(record.get("symptom")):
            stats["symptom_non_empty"] += 1
        if has_value(record.get("check")):
            stats["check_non_empty"] += 1
        if has_value(record.get("recommand_drug")):
            stats["recommand_drug_non_empty"] += 1

        if is_valid_record(record):
            stats["valid_records"] += 1
            if len(samples) < 3:
                samples.append(record)

    print(f"Total records: {stats['total_records']}")
    print(f"Valid records: {stats['valid_records']}")
    print(f"Name non-empty: {stats['name_non_empty']}")
    print(f"Desc non-empty: {stats['desc_non_empty']}")
    print(f"Symptom non-empty: {stats['symptom_non_empty']}")
    print(f"Check non-empty: {stats['check_non_empty']}")
    print(f"Recommand drug non-empty: {stats['recommand_drug_non_empty']}")
    print(f"JSON parse errors: {parse_errors}")

    print("\nFirst 3 valid samples:")
    for index, sample in enumerate(samples, start=1):
        category = "、".join(normalize_items(sample.get("category")))
        symptoms = "、".join(normalize_items(sample.get("symptom"))[:5])
        print(f"{index}. name: {clean_text(sample.get('name'))}")
        print(f"   category: {category}")
        print(f"   symptom_top5: {symptoms}")


if __name__ == "__main__":
    main()
