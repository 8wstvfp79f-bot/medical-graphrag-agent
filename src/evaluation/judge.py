from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from src.evaluation.schemas import EvaluationCase
from src.llm.client import LLMClientError, OpenAICompatibleChatClient


_JUDGE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "completeness": {"type": "number", "minimum": 0, "maximum": 1},
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "faithfulness",
        "relevance",
        "completeness",
        "unsupported_claims",
        "rationale",
    ],
    "additionalProperties": False,
}
_JUDGE_EVIDENCE_FIELDS = (
    "citation_id",
    "evidence_type",
    "disease_name",
    "content",
    "relation",
    "entity_name",
)


def _compact_evidence(evidence: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Keep only fields that can affect semantic judging.

    Retrieval ranks, fusion scores, metadata copies, and provenance remain in the
    evaluation artifact but must not consume the local model's context window.
    """

    compact: list[dict[str, object]] = []
    for item in evidence:
        selected = {
            field: item.get(field)
            for field in _JUDGE_EVIDENCE_FIELDS
            if item.get(field) not in (None, "", [], {})
        }
        if selected:
            compact.append(selected)
    return compact


@dataclass(frozen=True)
class JudgeRequest:
    case_id: str
    retrieval_mode: str
    question: str
    disease_name: str
    gold_entities: list[str]
    answer: str
    evidence: list[dict[str, object]]
    rubric: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> "JudgeRequest":
        if not isinstance(value, Mapping):
            raise ValueError("Judge request must be an object")
        evidence = value.get("evidence")
        rubric = value.get("rubric")
        gold_entities = value.get("gold_entities")
        if not isinstance(evidence, list) or not all(
            isinstance(item, Mapping) for item in evidence
        ):
            raise ValueError("Judge request evidence must be an object array")
        if not isinstance(rubric, Mapping):
            raise ValueError("Judge request rubric must be an object")
        if not isinstance(gold_entities, list):
            raise ValueError("Judge request gold_entities must be an array")
        fields = {
            name: str(value.get(name, "")).strip()
            for name in (
                "case_id",
                "retrieval_mode",
                "question",
                "disease_name",
                "answer",
            )
        }
        if not all(fields.values()):
            raise ValueError("Judge request is missing required string fields")
        return cls(
            **fields,
            gold_entities=[str(item).strip() for item in gold_entities if str(item).strip()],
            evidence=[dict(item) for item in evidence],
            rubric={str(key): str(item) for key, item in rubric.items()},
        )


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: float
    relevance: float
    completeness: float
    unsupported_claims: list[str]
    rationale: str

    @classmethod
    def from_mapping(cls, value: object) -> "JudgeResult":
        if not isinstance(value, Mapping):
            raise LLMClientError("Judge result must be a JSON object")

        def score(name: str) -> float:
            try:
                result = float(value.get(name))
            except (TypeError, ValueError) as exc:
                raise LLMClientError(f"Judge result {name} must be numeric") from exc
            if result < 0 or result > 1:
                raise LLMClientError(f"Judge result {name} must be between 0 and 1")
            return round(result, 6)

        unsupported = value.get("unsupported_claims", [])
        if not isinstance(unsupported, list):
            raise LLMClientError("unsupported_claims must be an array")
        return cls(
            faithfulness=score("faithfulness"),
            relevance=score("relevance"),
            completeness=score("completeness"),
            unsupported_claims=[
                str(item).strip() for item in unsupported if str(item).strip()
            ],
            rationale=str(value.get("rationale", "")).strip(),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LLMJudge(Protocol):
    async def judge(self, request: JudgeRequest) -> Mapping[str, object]:
        """Return semantic faithfulness, relevance, and completeness judgments."""


class OpenAICompatibleLLMJudge:
    """Evidence-only semantic judge backed by a real configured chat model."""

    prompt_version = "medical-evidence-judge-v1"

    def __init__(
        self,
        client: OpenAICompatibleChatClient,
        *,
        max_tokens: int = 900,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.client = client
        self.max_tokens = max_tokens

    async def judge(self, request: JudgeRequest) -> Mapping[str, object]:
        payload = {
            "question": request.question,
            "disease_name": request.disease_name,
            "reference_entities": request.gold_entities,
            "answer": request.answer,
            "evidence": _compact_evidence(request.evidence),
            "rubric": request.rubric,
            "score_range": "0.0 to 1.0",
            "output_schema": {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
                "unsupported_claims": ["unsupported factual claim"],
                "rationale": "brief evidence-based explanation",
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严格的医疗RAG评测员。只依据给定证据判断回答，"
                    "不要用外部医学知识补全。逐项识别证据无法支持的事实性陈述。"
                    "三个分数必须是0到1之间的数字。只输出符合schema的JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        first_error: Exception | None = None
        first_result: Mapping[str, object] | None = None
        try:
            first_result = await self.client.complete_json(
                messages,
                max_tokens=self.max_tokens,
                schema=_JUDGE_RESPONSE_SCHEMA,
                schema_name="medical_evidence_judgment",
            )
            return JudgeResult.from_mapping(first_result).to_dict()
        except LLMClientError as exc:
            first_error = exc

        repair_payload = {
            **payload,
            "invalid_previous_result": first_result,
            "validation_error": str(first_error),
            "repair_requirements": [
                "重新完成评审，不沿用格式错误的字段",
                "faithfulness、relevance、completeness必须是0到1之间的数字",
                "unsupported_claims必须是字符串数组",
                "所有schema字段都必须存在",
            ],
        }
        repaired = await self.client.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是医疗RAG评分结果修复器。只依据给定问题、回答和证据"
                        "重新评分；不要加入外部知识。只输出符合schema的JSON对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(repair_payload, ensure_ascii=False),
                },
            ],
            max_tokens=self.max_tokens,
            schema=_JUDGE_RESPONSE_SCHEMA,
            schema_name="repaired_medical_evidence_judgment",
        )
        return JudgeResult.from_mapping(repaired).to_dict()


def build_judge_request(
    case: EvaluationCase,
    *,
    retrieval_mode: str,
    answer: str,
    evidence: Sequence[Mapping[str, object]],
) -> JudgeRequest:
    return JudgeRequest(
        case_id=case.case_id,
        retrieval_mode=retrieval_mode,
        question=case.question,
        disease_name=case.disease_name,
        gold_entities=list(case.gold_entities),
        answer=answer,
        evidence=[dict(item) for item in evidence],
        rubric={
            "faithfulness": "回答中的医学事实是否都能由所给证据支持？",
            "relevance": "回答是否直接解决问题，而非提供无关信息？",
            "completeness": "回答是否覆盖参考实体中的关键信息？",
        },
    )
