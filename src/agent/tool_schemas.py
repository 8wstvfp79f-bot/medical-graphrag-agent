from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.retrieval.graph_schema import CORE_RELATIONS, normalize_relations
from src.retrieval.metadata_filter import (
    REQUIRED_METADATA_FIELDS,
    normalize_metadata_filter,
)


MAX_TOOL_LIMIT = 50
SUPPORTED_TOOL_NAMES = frozenset(
    {"vector_search", "graph_search", "hybrid_search"}
)


def _nullable_filter_property() -> dict[str, object]:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ]
    }


METADATA_FILTER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        field: _nullable_filter_property() for field in REQUIRED_METADATA_FIELDS
    },
    "required": list(REQUIRED_METADATA_FIELDS),
    "additionalProperties": False,
}


MEDICAL_TOOL_SCHEMAS: tuple[dict[str, object], ...] = (
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Search semantically related medical text chunks in Milvus. Use it "
                "for causes, prevention, transmission, definitions, or other intents "
                "that are not represented by the core graph relations."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_LIMIT,
                    },
                    "metadata_filter": METADATA_FILTER_SCHEMA,
                },
                "required": ["query", "top_k", "metadata_filter"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_search",
            "description": (
                "Expand one exact disease entity through allowlisted core medical "
                "relations in Neo4j."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "disease_name": {"type": "string", "minLength": 1},
                    "relations": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(CORE_RELATIONS)},
                        "minItems": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_LIMIT,
                    },
                },
                "required": ["query", "disease_name", "relations", "limit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "Retrieve both Milvus text evidence and Neo4j graph evidence for "
                "symptoms, checks, treatments, departments, or complications."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "disease_name": {"type": "string", "minLength": 1},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_LIMIT,
                    },
                    "vector_top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_LIMIT,
                    },
                    "graph_top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_LIMIT,
                    },
                    "metadata_filter": METADATA_FILTER_SCHEMA,
                    "relations": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(CORE_RELATIONS)},
                        "minItems": 1,
                    },
                    "allow_partial": {"type": "boolean"},
                },
                "required": [
                    "query",
                    "disease_name",
                    "top_k",
                    "vector_top_k",
                    "graph_top_k",
                    "metadata_filter",
                    "relations",
                    "allow_partial",
                ],
                "additionalProperties": False,
            },
        },
    },
)


class ToolCallValidationError(ValueError):
    """Raised before an untrusted Function Calling payload reaches a backend."""


@dataclass(frozen=True)
class ValidatedToolCall:
    name: str
    arguments: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "arguments": self.arguments}


_ALLOWED_ARGUMENTS = {
    "vector_search": frozenset({"query", "top_k", "metadata_filter"}),
    "graph_search": frozenset({"query", "disease_name", "relations", "limit"}),
    "hybrid_search": frozenset(
        {
            "query",
            "disease_name",
            "top_k",
            "vector_top_k",
            "graph_top_k",
            "metadata_filter",
            "relations",
            "allow_partial",
        }
    ),
}


def empty_metadata_filter(*, disease_name: str | None = None) -> dict[str, object]:
    """Return the full nullable object required by strict Function Calling."""

    return {
        "disease_name": disease_name,
        "category": None,
        "department": None,
        "source": None,
    }


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ToolCallValidationError(f"Tool argument '{field}' must be a string")
    cleaned = " ".join(value.replace("\u3000", " ").split())
    if not cleaned:
        raise ToolCallValidationError(f"Tool argument '{field}' must not be empty")
    return cleaned


def _bounded_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolCallValidationError(f"Tool argument '{field}' must be an integer")
    if value < 1 or value > MAX_TOOL_LIMIT:
        raise ToolCallValidationError(
            f"Tool argument '{field}' must be between 1 and {MAX_TOOL_LIMIT}"
        )
    return value


def _metadata_filter(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ToolCallValidationError(
            "Tool argument 'metadata_filter' must be an object"
        )
    missing = set(REQUIRED_METADATA_FIELDS) - set(value)
    if missing:
        raise ToolCallValidationError(
            "Missing metadata_filter fields required by the strict schema: "
            f"{', '.join(sorted(missing))}"
        )
    compact = {key: item for key, item in value.items() if item is not None}
    try:
        normalized = normalize_metadata_filter(compact)
    except ValueError as exc:
        raise ToolCallValidationError(str(exc)) from exc
    result: dict[str, object] = {field: None for field in REQUIRED_METADATA_FIELDS}
    result.update({field: list(values) for field, values in normalized.items()})
    return result


def _relations(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ToolCallValidationError("Tool argument 'relations' must be an array")
    try:
        return list(normalize_relations(list(value)))
    except ValueError as exc:
        raise ToolCallValidationError(str(exc)) from exc


def _validate_required_and_unknown(
    name: str,
    arguments: Mapping[str, object],
) -> None:
    allowed = _ALLOWED_ARGUMENTS[name]
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolCallValidationError(
            f"Unsupported arguments for {name}: {', '.join(sorted(unknown))}"
        )
    missing = allowed - set(arguments)
    if missing:
        raise ToolCallValidationError(
            f"Missing arguments for {name}: {', '.join(sorted(missing))}"
        )


def validate_tool_call(
    name: object,
    arguments: object,
) -> ValidatedToolCall:
    """Validate and normalize an LLM- or rule-produced tool call.

    This is the trust boundary: only allowlisted tools, fields, relations, metadata
    keys, and bounded result sizes can reach Milvus or Neo4j.
    """

    if not isinstance(name, str) or name not in SUPPORTED_TOOL_NAMES:
        raise ToolCallValidationError(
            "Unsupported tool name. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_TOOL_NAMES))}"
        )
    if not isinstance(arguments, Mapping):
        raise ToolCallValidationError("Tool arguments must be an object")
    _validate_required_and_unknown(name, arguments)

    normalized: dict[str, object] = {
        "query": _clean_text(arguments["query"], "query")
    }
    if name == "vector_search":
        normalized["top_k"] = _bounded_int(arguments["top_k"], "top_k")
        normalized_filter = _metadata_filter(arguments["metadata_filter"])
        configured_diseases = normalized_filter.get("disease_name")
        if not isinstance(configured_diseases, list) or len(configured_diseases) != 1:
            raise ToolCallValidationError(
                "vector_search requires exactly one metadata_filter.disease_name"
            )
        normalized["metadata_filter"] = normalized_filter
        return ValidatedToolCall(name=name, arguments=normalized)

    disease_name = _clean_text(arguments["disease_name"], "disease_name")
    normalized["disease_name"] = disease_name
    normalized["relations"] = _relations(arguments["relations"])
    if name == "graph_search":
        normalized["limit"] = _bounded_int(arguments["limit"], "limit")
        return ValidatedToolCall(name=name, arguments=normalized)

    normalized["top_k"] = _bounded_int(arguments["top_k"], "top_k")
    normalized["vector_top_k"] = _bounded_int(
        arguments["vector_top_k"], "vector_top_k"
    )
    normalized["graph_top_k"] = _bounded_int(
        arguments["graph_top_k"], "graph_top_k"
    )
    normalized_filter = _metadata_filter(arguments["metadata_filter"])
    configured_diseases = normalized_filter.get("disease_name")
    if not isinstance(configured_diseases, list):
        configured_diseases = []
    if {value.casefold() for value in configured_diseases} != {
        disease_name.casefold()
    }:
        raise ToolCallValidationError(
            "hybrid_search metadata_filter.disease_name must match disease_name"
        )
    normalized["metadata_filter"] = normalized_filter
    allow_partial = arguments["allow_partial"]
    if not isinstance(allow_partial, bool):
        raise ToolCallValidationError(
            "Tool argument 'allow_partial' must be a boolean"
        )
    normalized["allow_partial"] = allow_partial
    return ValidatedToolCall(name=name, arguments=normalized)
