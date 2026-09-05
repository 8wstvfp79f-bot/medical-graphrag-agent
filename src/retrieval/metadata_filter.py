from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


REQUIRED_METADATA_FIELDS = ("disease_name", "category", "department", "source")
MULTI_VALUE_FIELDS = frozenset({"category", "department"})

MetadataFilterInput = Mapping[str, str | Sequence[str]]
NormalizedMetadataFilter = dict[str, tuple[str, ...]]


class MetadataFilterError(ValueError):
    """Raised when a caller sends unsupported or empty metadata filters."""


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("\u3000", " ").split())


def _values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean(value)
        return [cleaned] if cleaned else []
    if isinstance(value, Sequence):
        result: list[str] = []
        for item in value:
            result.extend(_values(item))
        return list(dict.fromkeys(result))
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def normalize_chunk_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    metadata = metadata or {}
    disease_values = _values(metadata.get("disease_name"))
    source_values = _values(metadata.get("source"))
    return {
        "disease_name": disease_values[0] if disease_values else "",
        "category": _values(metadata.get("category")),
        "department": _values(metadata.get("department")),
        "source": source_values[0] if source_values else "",
    }


def normalize_metadata_filter(
    metadata_filter: MetadataFilterInput | None,
) -> NormalizedMetadataFilter:
    if not metadata_filter:
        return {}

    unknown_fields = set(metadata_filter) - set(REQUIRED_METADATA_FIELDS)
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise MetadataFilterError(f"Unsupported metadata filter fields: {names}")

    normalized: NormalizedMetadataFilter = {}
    for field, value in metadata_filter.items():
        values = tuple(_values(value))
        if not values:
            raise MetadataFilterError(f"Metadata filter '{field}' must not be empty")
        normalized[field] = values
    return normalized


def metadata_matches(
    metadata: Mapping[str, object] | None,
    metadata_filter: MetadataFilterInput | NormalizedMetadataFilter | None,
) -> bool:
    normalized_metadata = normalize_chunk_metadata(metadata)
    normalized_filter = normalize_metadata_filter(metadata_filter)

    for field, expected_values in normalized_filter.items():
        actual_values = _values(normalized_metadata[field])
        actual_keys = {value.casefold() for value in actual_values}
        expected_keys = {value.casefold() for value in expected_values}
        if actual_keys.isdisjoint(expected_keys):
            return False
    return True


def matched_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Return the auditable metadata fields alongside every retrieval hit."""
    return normalize_chunk_metadata(metadata)


def build_milvus_filter_expression(
    metadata_filter: MetadataFilterInput | NormalizedMetadataFilter | None,
    *,
    metadata_field: str = "metadata",
) -> str:
    """Translate the public filter mapping into a Milvus JSON-field expression."""
    normalized = normalize_metadata_filter(metadata_filter)
    expressions: list[str] = []
    for field, values in normalized.items():
        encoded_values = json.dumps(list(values), ensure_ascii=False)
        json_path = f'{metadata_field}["{field}"]'
        if field in MULTI_VALUE_FIELDS:
            expressions.append(f"json_contains_any({json_path}, {encoded_values})")
        else:
            expressions.append(f"{json_path} in {encoded_values}")
    return " and ".join(expressions)
