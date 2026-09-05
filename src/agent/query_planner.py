from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from src.agent.tool_schemas import empty_metadata_filter, validate_tool_call
from src.retrieval.graph_schema import CORE_RELATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.jsonl"


class QueryPlanningError(ValueError):
    """Raised when the local query planner cannot build a safe plan."""


@dataclass(frozen=True)
class QueryPlan:
    planner: str
    status: str
    query: str
    intents: tuple[str, ...]
    disease_candidates: tuple[str, ...]
    disease_name: str | None
    selected_tool: str | None
    tool_call: dict[str, object] | None
    reason: str
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["intents"] = list(self.intents)
        result["disease_candidates"] = list(self.disease_candidates)
        return result


class DiseaseCatalog:
    """Exact disease-name catalog built from the project's normalized documents."""

    def __init__(self, disease_names: list[str] | tuple[str, ...]) -> None:
        cleaned = {
            " ".join(str(name).replace("\u3000", " ").split())
            for name in disease_names
            if str(name).strip()
        }
        if not cleaned:
            raise QueryPlanningError("Disease catalog must contain at least one name")
        self.disease_names = tuple(sorted(cleaned, key=lambda name: (-len(name), name)))
        self._exact = {name.casefold(): name for name in self.disease_names}

    @classmethod
    def from_documents(cls, path: Path = DEFAULT_DOCUMENTS_PATH) -> DiseaseCatalog:
        if not path.exists():
            raise FileNotFoundError(f"Documents file not found: {path}")
        names: list[str] = []
        with path.open("r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QueryPlanningError(
                        f"Invalid JSON at {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise QueryPlanningError(
                        f"Expected an object at {path}:{line_no}"
                    )
                name = str(record.get("disease_name", "")).strip()
                if name:
                    names.append(name)
        return cls(names)

    def exact_name(self, value: str) -> str | None:
        cleaned = " ".join(str(value).replace("\u3000", " ").split())
        return self._exact.get(cleaned.casefold())

    def extract(self, query: str) -> tuple[str, ...]:
        cleaned_query = " ".join(query.replace("\u3000", " ").split()).casefold()
        matched = [
            name for name in self.disease_names if name.casefold() in cleaned_query
        ]
        maximal = [
            name
            for name in matched
            if not any(
                name.casefold() != other.casefold()
                and name.casefold() in other.casefold()
                for other in matched
            )
        ]
        return tuple(
            sorted(
                maximal,
                key=lambda name: (cleaned_query.index(name.casefold()), -len(name)),
            )
        )


RELATION_INTENT_KEYWORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("symptom", "has_symptom", ("症状", "表现", "征兆", "体征")),
    ("diagnosis", "diagnosed_by", ("检查", "诊断", "确诊", "化验", "检测")),
    ("treatment", "treated_by", ("治疗", "怎么治", "用药", "药物", "吃什么药")),
    ("department", "belongs_to", ("科室", "挂什么科", "看什么科", "去哪科")),
    ("complication", "has_complication", ("并发症", "并发", "后遗症")),
)

TEXT_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cause", ("病因", "原因", "为什么得", "怎么得")),
    ("prevention", ("预防", "防止", "避免")),
    ("transmission", ("传播", "传染", "传染性")),
    ("overview", ("是什么", "介绍", "概述")),
)


class DeterministicMedicalQueryPlanner:
    """Turn a Chinese medical question into one validated retrieval tool call.

    This planner is intentionally deterministic and offline. The exported JSON
    schemas can later be sent to an LLM, while the same validation boundary and
    executor remain unchanged.
    """

    planner_name = "deterministic-medical-planner-v1"

    def __init__(
        self,
        catalog: DiseaseCatalog,
        *,
        top_k: int = 10,
        vector_top_k: int = 6,
        graph_top_k: int = 6,
        allow_partial: bool = False,
    ) -> None:
        self.catalog = catalog
        self.top_k = top_k
        self.vector_top_k = vector_top_k
        self.graph_top_k = graph_top_k
        self.allow_partial = allow_partial

    def _recognize_intents(self, query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        intents: list[str] = []
        relations_by_intent: dict[str, str] = {}
        for intent, relation, keywords in RELATION_INTENT_KEYWORDS:
            if any(keyword in query for keyword in keywords):
                intents.append(intent)
                relations_by_intent[intent] = relation
        for intent, keywords in TEXT_INTENT_KEYWORDS:
            if any(keyword in query for keyword in keywords) and intent not in intents:
                intents.append(intent)
        if not intents:
            intents.append("overview")

        requested = set(relations_by_intent.values())
        relations = tuple(relation for relation in CORE_RELATIONS if relation in requested)
        return tuple(intents), relations

    def plan(
        self,
        query: str,
        *,
        disease_name: str | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        top_k: int | None = None,
        vector_top_k: int | None = None,
        graph_top_k: int | None = None,
        allow_partial: bool | None = None,
    ) -> QueryPlan:
        cleaned_query = " ".join(str(query).replace("\u3000", " ").split())
        if not cleaned_query:
            raise QueryPlanningError("query must not be empty")

        if disease_name is not None:
            exact = self.catalog.exact_name(disease_name)
            candidates = (exact,) if exact else ()
            if exact is None:
                return QueryPlan(
                    planner=self.planner_name,
                    status="needs_clarification",
                    query=cleaned_query,
                    intents=(),
                    disease_candidates=(),
                    disease_name=None,
                    selected_tool=None,
                    tool_call=None,
                    reason="The supplied disease name is not in documents.jsonl.",
                    clarification_question=(
                        f"当前知识库中找不到“{disease_name}”，请确认疾病全名。"
                    ),
                )
        else:
            candidates = self.catalog.extract(cleaned_query)

        if not candidates:
            return QueryPlan(
                planner=self.planner_name,
                status="needs_clarification",
                query=cleaned_query,
                intents=(),
                disease_candidates=(),
                disease_name=None,
                selected_tool=None,
                tool_call=None,
                reason="No exact disease entity was found in the question.",
                clarification_question="你想咨询哪一种疾病？请在问题中写出疾病全名。",
            )
        if len(candidates) > 1:
            names = "、".join(candidates)
            return QueryPlan(
                planner=self.planner_name,
                status="needs_clarification",
                query=cleaned_query,
                intents=(),
                disease_candidates=candidates,
                disease_name=None,
                selected_tool=None,
                tool_call=None,
                reason="More than one independent disease entity was found.",
                clarification_question=f"问题中提到了{names}，你具体想查询哪一种？",
            )

        selected_disease = candidates[0]
        intents, relations = self._recognize_intents(cleaned_query)
        configured_filter = empty_metadata_filter(disease_name=selected_disease)
        if metadata_filter:
            configured_filter.update(metadata_filter)
            configured_filter["disease_name"] = selected_disease
        resolved_top_k = self.top_k if top_k is None else top_k
        resolved_vector_top_k = (
            self.vector_top_k if vector_top_k is None else vector_top_k
        )
        resolved_graph_top_k = (
            self.graph_top_k if graph_top_k is None else graph_top_k
        )
        resolved_allow_partial = (
            self.allow_partial if allow_partial is None else allow_partial
        )
        if relations:
            tool_name = "hybrid_search"
            arguments: dict[str, object] = {
                "query": cleaned_query,
                "disease_name": selected_disease,
                "top_k": resolved_top_k,
                "vector_top_k": resolved_vector_top_k,
                "graph_top_k": resolved_graph_top_k,
                "metadata_filter": configured_filter,
                "relations": list(relations),
                "allow_partial": resolved_allow_partial,
            }
            reason = (
                "The intent maps to one or more core Neo4j relations, so both "
                "semantic text and structured graph evidence are requested."
            )
        else:
            tool_name = "vector_search"
            arguments = {
                "query": cleaned_query,
                "top_k": resolved_top_k,
                "metadata_filter": configured_filter,
            }
            reason = (
                "The intent is not represented by a core graph relation, so the "
                "planner uses metadata-filtered semantic text retrieval."
            )

        validated = validate_tool_call(tool_name, arguments)
        return QueryPlan(
            planner=self.planner_name,
            status="ready",
            query=cleaned_query,
            intents=intents,
            disease_candidates=candidates,
            disease_name=selected_disease,
            selected_tool=tool_name,
            tool_call=validated.to_dict(),
            reason=reason,
        )
