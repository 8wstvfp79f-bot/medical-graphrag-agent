from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetadataFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disease_name: str | list[str] | None = None
    category: str | list[str] | None = None
    department: str | list[str] | None = None
    source: str | list[str] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    disease_name: str | None = Field(default=None, max_length=200)
    metadata_filter: MetadataFilterRequest | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    vector_top_k: int = Field(default=6, ge=1, le=50)
    graph_top_k: int = Field(default=6, ge=1, le=50)
    allow_partial: bool = False
    stream: bool = False

    @field_validator("query", "disease_name")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.replace("\u3000", " ").split())
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    request_id: str
    answer: str
    answer_generation: dict[str, Any]
    sources: list[str]
    citations: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    graph_evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    query_plan: dict[str, Any]
    retrieval_stats: dict[str, Any] | None
    context_budget: dict[str, Any] | None
