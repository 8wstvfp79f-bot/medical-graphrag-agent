from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_CONFIG_PATH = PROJECT_ROOT / "config" / "medical_data_quality.json"
DEFAULT_REJECTION_AUDIT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "rejected_medical_items.jsonl"
)

MEDICAL_LIST_FIELDS = (
    "category",
    "symptom",
    "check",
    "cure_department",
    "cure_way",
    "common_drug",
    "recommand_drug",
    "acompany",
)
PLACEHOLDERS = frozenset(
    {
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
)
_CONTENT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_TRIM_CHARS = " \t\r\n,，.。;；:：、|/\\()（）[]【】{}<>《》-_=+*#@!！?？'\""


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("\u3000", " ").split())


def normalize_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items: list[str] = []
        for item in value:
            items.extend(normalize_items(item))
        return items
    if isinstance(value, set):
        items = []
        for item in value:
            items.extend(normalize_items(item))
        return items
    text = clean_text(value)
    return [text] if text else []


@dataclass(frozen=True)
class RejectedMedicalItem:
    disease_name: str
    field: str
    value: str
    reason: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class MedicalQualityConfigError(ValueError):
    """Raised when the versioned medical data-quality config is invalid."""


class MedicalQualityPolicy:
    """Apply one auditable field policy to documents, graph facts, and answers."""

    def __init__(
        self,
        *,
        rejected_exact: Mapping[str, Mapping[str, str]] | None = None,
        aliases: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.rejected_exact = {
            clean_text(field): {
                clean_text(value): clean_text(reason) or "configured_exact_rejection"
                for value, reason in values.items()
                if clean_text(value)
            }
            for field, values in (rejected_exact or {}).items()
            if clean_text(field)
        }
        self.aliases = {
            clean_text(field): {
                clean_text(value): clean_text(canonical)
                for value, canonical in values.items()
                if clean_text(value) and clean_text(canonical)
            }
            for field, values in (aliases or {}).items()
            if clean_text(field)
        }

    @classmethod
    def from_path(cls, path: Path = DEFAULT_QUALITY_CONFIG_PATH) -> MedicalQualityPolicy:
        if not path.exists():
            raise FileNotFoundError(f"Medical quality config not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MedicalQualityConfigError(
                f"Invalid JSON in medical quality config {path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise MedicalQualityConfigError("Medical quality config must be an object")
        if payload.get("schema_version") != 1:
            raise MedicalQualityConfigError(
                "Medical quality config schema_version must be 1"
            )
        rejected = payload.get("rejected_exact", {})
        aliases = payload.get("aliases", {})
        if not isinstance(rejected, Mapping) or not isinstance(aliases, Mapping):
            raise MedicalQualityConfigError(
                "rejected_exact and aliases must be objects"
            )
        for section_name, section in (
            ("rejected_exact", rejected),
            ("aliases", aliases),
        ):
            if not all(isinstance(values, Mapping) for values in section.values()):
                raise MedicalQualityConfigError(
                    f"Every {section_name} field must contain an object"
                )
        return cls(rejected_exact=rejected, aliases=aliases)

    def filter_items(
        self,
        field: str,
        value: object,
        *,
        disease_name: str = "",
        source: str = "",
    ) -> tuple[list[str], list[RejectedMedicalItem]]:
        field = clean_text(field)
        accepted: list[str] = []
        accepted_keys: set[str] = set()
        rejected: list[RejectedMedicalItem] = []
        exact_rejections = self.rejected_exact.get(field, {})
        aliases = self.aliases.get(field, {})

        def reject(item: str, reason: str) -> None:
            rejected.append(
                RejectedMedicalItem(
                    disease_name=clean_text(disease_name),
                    field=field,
                    value=item,
                    reason=reason,
                    source=clean_text(source),
                )
            )

        for raw_item in normalize_items(value):
            item = clean_text(raw_item)
            if not item:
                continue
            configured_reason = exact_rejections.get(item)
            if configured_reason:
                reject(item, configured_reason)
                continue
            compact = item.strip(_TRIM_CHARS)
            if not compact:
                reject(item, "punctuation_only")
                continue
            if compact.casefold() in PLACEHOLDERS:
                reject(item, "placeholder_value")
                continue
            if not _CONTENT_RE.search(compact):
                reject(item, "no_meaningful_characters")
                continue

            canonical = aliases.get(item, item)
            key = canonical.casefold()
            if key in accepted_keys:
                reject(item, "duplicate_after_normalization")
                continue
            accepted_keys.add(key)
            accepted.append(canonical)
        return accepted, rejected

    def remove_rejected_terms(self, field: str, value: object) -> str:
        """Remove configured exact rejects from an unstructured fallback snippet."""

        cleaned = clean_text(value)
        for rejected_value in self.rejected_exact.get(clean_text(field), {}):
            cleaned = cleaned.replace(rejected_value, "")
        cleaned = re.sub(r"\s*、\s*、+\s*", "、", cleaned)
        cleaned = re.sub(r"([,，;；])\s*[,，;；]+", r"\1", cleaned)
        return cleaned.strip(" 、,，;；")


@lru_cache(maxsize=1)
def load_default_quality_policy() -> MedicalQualityPolicy:
    return MedicalQualityPolicy.from_path(DEFAULT_QUALITY_CONFIG_PATH)


def sanitize_medical_record(
    record: Mapping[str, object],
    *,
    policy: MedicalQualityPolicy | None = None,
    source: str,
) -> tuple[dict[str, object], list[RejectedMedicalItem]]:
    active_policy = policy or load_default_quality_policy()
    sanitized = dict(record)
    disease_name = clean_text(record.get("name"))
    rejected: list[RejectedMedicalItem] = []
    for field in MEDICAL_LIST_FIELDS:
        accepted, field_rejected = active_policy.filter_items(
            field,
            record.get(field),
            disease_name=disease_name,
            source=source,
        )
        sanitized[field] = accepted
        rejected.extend(field_rejected)
    return sanitized, rejected


def write_rejection_audit(
    path: Path,
    items: Sequence[RejectedMedicalItem],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[tuple[str, str, str, str, str], RejectedMedicalItem] = {}
    for item in items:
        key = (
            item.disease_name,
            item.field,
            item.value,
            item.reason,
            item.source,
        )
        unique[key] = item
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.disease_name,
            item.field,
            item.value,
            item.reason,
        ),
    )
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in ordered:
            output.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return len(ordered)
