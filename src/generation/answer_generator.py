from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from src.data.medical_quality import MedicalQualityPolicy, load_default_quality_policy
from src.llm.client import LLMClientError, OpenAICompatibleChatClient


DISCLAIMER = "以上内容来自当前知识库检索证据，仅供健康信息参考，不能替代医生诊断。"
RELATION_LABELS = {
    "has_symptom": "相关症状",
    "diagnosed_by": "相关检查",
    "treated_by": "相关治疗或药品",
    "belongs_to": "相关就诊科室",
    "has_complication": "相关并发症",
}
RELATION_FIELDS = {
    "has_symptom": "symptom",
    "diagnosed_by": "check",
    "treated_by": "common_drug",
    "belongs_to": "cure_department",
    "has_complication": "acompany",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_CITATION_ID = re.compile(r"\[(E\d+)\]")
_CITATION_BRACKETS = re.compile(r"[\[【]([^\]】]+)[\]】]")
_CITATION_TOKEN = re.compile(r"E\d+", flags=re.IGNORECASE)
_TOPIC_RULES = (
    ("symptom", ("症状", "表现"), ("症状", "临床表现"), ("症状", "表现", "特征")),
    ("diagnosis", ("检查", "诊断"), ("检查",), ("检查", "诊断", "检验")),
    ("treatment", ("治疗", "用药", "药物"), ("治疗方式", "常用药品", "推荐药品"), ("治疗", "药", "手术")),
    ("department", ("科室", "挂号"), ("治疗科室", "科室"), ("科室", "门诊")),
    ("complication", ("并发症",), ("并发症",), ("并发症", "合并")),
    ("cause", ("病因", "原因"), ("病因",), ("病因", "发病原因", "病原")),
    ("prevention", ("预防", "传播"), ("预防", "传播途径"), ("预防", "传播")),
)
_ALL_FIELD_LABELS = tuple(
    dict.fromkeys(label for _, _, labels, _ in _TOPIC_RULES for label in labels)
)
_LABEL_TO_MEDICAL_FIELD = {
    "症状": "symptom",
    "临床表现": "symptom",
    "检查": "check",
    "治疗方式": "cure_way",
    "常用药品": "common_drug",
    "推荐药品": "recommand_drug",
    "治疗科室": "cure_department",
    "科室": "cure_department",
    "并发症": "acompany",
}
_INTENT_TO_MEDICAL_FIELDS = {
    "symptom": ("symptom",),
    "diagnosis": ("check",),
    "treatment": ("cure_way", "common_drug", "recommand_drug"),
    "department": ("cure_department",),
    "complication": ("acompany",),
}


class AnswerGenerationError(ValueError):
    """Raised when an evidence package cannot be converted into an answer."""


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    model: str
    mode: str
    citations: list[dict[str, object]]
    sources: list[str]
    used_evidence_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AnswerGenerator(Protocol):
    async def generate(
        self,
        query: str,
        context_package: Mapping[str, object],
        *,
        intents: Sequence[str] | None = None,
    ) -> GeneratedAnswer:
        """Generate one citation-aware answer from a bounded context package."""


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\u3000", " ").split())


def _normalize_answer_citations(answer: str) -> tuple[str, list[str]]:
    """Normalize common local-model citation variants to ``[E1]`` tokens."""

    def replace(match: re.Match[str]) -> str:
        tokens = [item.upper() for item in _CITATION_TOKEN.findall(match.group(1))]
        return "".join(f"[{item}]" for item in dict.fromkeys(tokens)) or match.group(0)

    normalized = _CITATION_BRACKETS.sub(replace, answer)
    return normalized, list(dict.fromkeys(_CITATION_ID.findall(normalized)))


def _citation_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"E(\d+)", value)
    return (int(match.group(1)), value) if match else (sys.maxsize, value)


def _grounded_statement_schema(valid_ids: set[str]) -> dict[str, object]:
    """Build a schema that only permits citation IDs present in this context."""

    ordered_ids = sorted(valid_ids, key=_citation_sort_key)
    return {
        "type": "object",
        "properties": {
            "statements": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "citation_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": ordered_ids},
                        },
                    },
                    "required": ["text", "citation_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["statements"],
        "additionalProperties": False,
    }


def _normalize_citation_id(value: object) -> str:
    cleaned = _clean(value).upper().strip("[]【】")
    return cleaned if re.fullmatch(r"E\d+", cleaned) else ""


def _strip_inline_citations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "" if _CITATION_TOKEN.search(match.group(1)) else match.group(0)

    return _clean(_CITATION_BRACKETS.sub(replace, text))


def _render_grounded_statements(
    response: Mapping[str, object], valid_ids: set[str]
) -> tuple[str, list[str]]:
    """Render citations in application code instead of trusting model punctuation."""

    raw_statements = response.get("statements")
    if not isinstance(raw_statements, Sequence) or isinstance(
        raw_statements, (str, bytes)
    ):
        raise LLMClientError("Structured answer is missing the statements array")

    lines: list[str] = []
    cited_ids: list[str] = []
    for item in raw_statements:
        if not isinstance(item, Mapping):
            raise LLMClientError("Structured answer contains a non-object statement")
        text = _strip_inline_citations(_clean(item.get("text")))
        raw_ids = item.get("citation_ids")
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise LLMClientError("Structured statement is missing citation_ids")
        statement_ids = list(
            dict.fromkeys(
                citation_id
                for citation_id in (_normalize_citation_id(value) for value in raw_ids)
                if citation_id
            )
        )
        invalid_ids = [item for item in statement_ids if item not in valid_ids]
        if not text or not statement_ids or invalid_ids:
            raise LLMClientError(
                "Structured statement is empty, uncited, or contains an unknown citation"
            )
        citations = "".join(f"[{citation_id}]" for citation_id in statement_ids)
        lines.append(f"{len(lines) + 1}. {text.rstrip('。')} {citations}。")
        cited_ids.extend(statement_ids)

    if not lines:
        raise LLMClientError("Structured answer contains no usable statements")
    return "根据当前知识库检索到的证据：\n\n" + "\n".join(lines), list(
        dict.fromkeys(cited_ids)
    )


def _snippet(
    query: str,
    content: str,
    *,
    max_chars: int,
    intents: Sequence[str] | None = None,
    quality_policy: MedicalQualityPolicy,
) -> str:
    content = _clean(content)
    normalized_intents = {_clean(intent) for intent in (intents or []) if _clean(intent)}
    active_rules = [
        rule
        for rule in _TOPIC_RULES
        if rule[0] in normalized_intents or any(term in query for term in rule[1])
    ]
    for _, _, field_labels, _ in active_rules:
        for field_label in field_labels:
            next_labels = "|".join(
                re.escape(label)
                for label in _ALL_FIELD_LABELS
                if label != field_label
            )
            match = re.search(
                rf"{re.escape(field_label)}\s*[:：]\s*(.+?)(?=\s*(?:{next_labels})\s*[:：]|$)",
                content,
                flags=re.DOTALL,
            )
            if match:
                field_value = _clean(match.group(1))
                medical_field = _LABEL_TO_MEDICAL_FIELD.get(field_label)
                if medical_field:
                    accepted, _ = quality_policy.filter_items(
                        medical_field,
                        re.split(r"\s*、\s*", field_value),
                        source="answer_generation",
                    )
                    if not accepted:
                        continue
                    field_value = "、".join(accepted)
                selected = f"{field_label}：{field_value}"
                if len(selected) <= max_chars:
                    return selected
                return f"{selected[:max_chars].rstrip('，,；;：:')}……"

    for intent, _, _, _ in active_rules:
        for medical_field in _INTENT_TO_MEDICAL_FIELDS.get(intent, ()):
            content = quality_policy.remove_rejected_terms(medical_field, content)

    sentences = [
        _clean(sentence)
        for sentence in _SENTENCE_SPLIT.split(content)
        if _clean(sentence)
    ]
    active_markers = [
        marker for _, _, _, markers in active_rules for marker in markers
    ]
    preferred = [
        sentence
        for sentence in sentences
        if active_markers and any(marker in sentence for marker in active_markers)
    ]
    candidates = preferred or [sentence for sentence in sentences if len(sentence) >= 10]
    selected = (candidates or sentences or [_clean(content)])[0]
    if len(selected) <= max_chars:
        return selected
    return f"{selected[:max_chars].rstrip('，,；;：:')}……"


def _select_evidence(
    evidence: list[dict[str, object]], max_items: int
) -> list[dict[str, object]]:
    """Keep retrieval order while preventing one evidence type monopolizing output."""

    selected = list(evidence[:max_items])
    if max_items < 2 or len(evidence) <= max_items:
        return selected
    available_types = {_clean(item.get("evidence_type")) for item in evidence}
    selected_types = {_clean(item.get("evidence_type")) for item in selected}
    if available_types.intersection({"graph", "hybrid"}) and not selected_types.intersection(
        {"graph", "hybrid"}
    ):
        graph_item = next(
            item
            for item in evidence
            if _clean(item.get("evidence_type")) in {"graph", "hybrid"}
        )
        selected[-1] = graph_item
    return selected


class ExtractiveCitationAnswerGenerator:
    """Create a usable answer without inventing facts or requiring an API key.

    It quotes short evidence summaries and graph facts instead of paraphrasing with
    a generative model. A later LLM backend can implement the same protocol.
    """

    model_name = "extractive-citation-v1"

    def __init__(
        self,
        *,
        max_evidence_items: int = 3,
        max_chars: int = 220,
        quality_policy: MedicalQualityPolicy | None = None,
    ) -> None:
        if max_evidence_items <= 0 or max_chars <= 0:
            raise AnswerGenerationError("Answer limits must be greater than zero")
        self.max_evidence_items = max_evidence_items
        self.max_chars = max_chars
        self.quality_policy = quality_policy or load_default_quality_policy()

    async def generate(
        self,
        query: str,
        context_package: Mapping[str, object],
        *,
        intents: Sequence[str] | None = None,
    ) -> GeneratedAnswer:
        query = _clean(query)
        if not query:
            raise AnswerGenerationError("query must not be empty")
        raw_evidence = context_package.get("evidence", [])
        raw_citations = context_package.get("citations", [])
        if not isinstance(raw_evidence, Sequence) or isinstance(
            raw_evidence, (str, bytes)
        ):
            raise AnswerGenerationError("context evidence must be an array")
        if not isinstance(raw_citations, Sequence) or isinstance(
            raw_citations, (str, bytes)
        ):
            raise AnswerGenerationError("context citations must be an array")

        available_evidence = [
            dict(item) for item in raw_evidence if isinstance(item, Mapping)
        ]
        evidence = _select_evidence(available_evidence, self.max_evidence_items)
        citation_by_id = {
            _clean(item.get("citation_id")): dict(item)
            for item in raw_citations
            if isinstance(item, Mapping) and _clean(item.get("citation_id"))
        }
        if not evidence:
            return GeneratedAnswer(
                answer=f"当前知识库没有找到足够证据回答这个问题。\n\n{DISCLAIMER}",
                model=self.model_name,
                mode="extractive_fallback",
                citations=[],
                sources=[],
                used_evidence_ids=[],
            )

        statements: list[str] = []
        citations: list[dict[str, object]] = []
        sources: list[str] = []
        evidence_ids: list[str] = []
        for index, item in enumerate(evidence, start=1):
            citation_id = _clean(item.get("citation_id")) or f"E{index}"
            evidence_type = _clean(item.get("evidence_type"))
            disease_name = _clean(item.get("disease_name")) or "该疾病"
            relation = _clean(item.get("relation"))
            entity_name = _clean(item.get("entity_name"))
            if evidence_type in {"graph", "hybrid"} and relation and entity_name:
                medical_field = RELATION_FIELDS.get(relation)
                if medical_field:
                    accepted, _ = self.quality_policy.filter_items(
                        medical_field,
                        [entity_name],
                        disease_name=disease_name,
                        source="answer_generation",
                    )
                    if not accepted:
                        continue
                    entity_name = accepted[0]
                label = RELATION_LABELS.get(relation, "相关医学信息")
                statement = f"{disease_name}的{label}包括“{entity_name}” [{citation_id}]。"
            else:
                snippet = _snippet(
                    query,
                    _clean(item.get("content")),
                    max_chars=self.max_chars,
                    intents=intents,
                    quality_policy=self.quality_policy,
                )
                statement = f"资料显示：{snippet} [{citation_id}]"
            statements.append(f"{len(statements) + 1}. {statement}")

            citation = citation_by_id.get(citation_id)
            if citation and citation not in citations:
                citations.append(citation)
            item_sources = item.get("sources", [item.get("source")])
            if isinstance(item_sources, Sequence) and not isinstance(
                item_sources, (str, bytes)
            ):
                for source in item_sources:
                    cleaned_source = _clean(source)
                    if cleaned_source and cleaned_source not in sources:
                        sources.append(cleaned_source)
            evidence_id = _clean(item.get("evidence_id"))
            if evidence_id:
                evidence_ids.append(evidence_id)

        if not statements:
            return GeneratedAnswer(
                answer=f"当前知识库没有找到足够证据回答这个问题。\n\n{DISCLAIMER}",
                model=self.model_name,
                mode="extractive_fallback",
                citations=[],
                sources=[],
                used_evidence_ids=[],
            )

        answer = "根据当前知识库检索到的证据：\n\n" + "\n".join(statements)
        answer += f"\n\n{DISCLAIMER}"
        return GeneratedAnswer(
            answer=answer,
            model=self.model_name,
            mode="extractive_fallback",
            citations=citations,
            sources=sources,
            used_evidence_ids=evidence_ids,
        )


class GroundedLLMAnswerGenerator:
    """Generate from bounded evidence through a configured chat model.

    Citation identifiers are validated against the Context Budget package before
    the answer is returned.  An extractive backend may be supplied as an explicit
    operational fallback; the response mode exposes which path actually ran.
    """

    def __init__(
        self,
        client: OpenAICompatibleChatClient,
        *,
        fallback: AnswerGenerator | None = None,
        max_tokens: int = 900,
    ) -> None:
        if max_tokens <= 0:
            raise AnswerGenerationError("max_tokens must be greater than zero")
        self.client = client
        self.fallback = fallback
        self.max_tokens = max_tokens

    async def _fallback(
        self,
        query: str,
        context_package: Mapping[str, object],
        intents: Sequence[str] | None,
        error: Exception,
    ) -> GeneratedAnswer:
        if self.fallback is None:
            raise AnswerGenerationError(
                f"Grounded LLM generation failed: {type(error).__name__}: {error}"
            ) from error
        return await self.fallback.generate(query, context_package, intents=intents)

    async def generate(
        self,
        query: str,
        context_package: Mapping[str, object],
        *,
        intents: Sequence[str] | None = None,
    ) -> GeneratedAnswer:
        query = _clean(query)
        context = _clean(context_package.get("context"))
        raw_citations = context_package.get("citations", [])
        raw_evidence = context_package.get("evidence", [])
        if not query:
            raise AnswerGenerationError("query must not be empty")
        if not context:
            if self.fallback is not None:
                return await self.fallback.generate(
                    query, context_package, intents=intents
                )
            raise AnswerGenerationError("context must not be empty")
        citations = (
            [
                dict(item)
                for item in raw_citations
                if isinstance(item, Mapping) and _clean(item.get("citation_id"))
            ]
            if isinstance(raw_citations, Sequence)
            and not isinstance(raw_citations, (str, bytes))
            else []
        )
        evidence = (
            [dict(item) for item in raw_evidence if isinstance(item, Mapping)]
            if isinstance(raw_evidence, Sequence)
            and not isinstance(raw_evidence, (str, bytes))
            else []
        )
        valid_ids = {_clean(item.get("citation_id")) for item in citations}
        if not valid_ids:
            return await self._fallback(
                query,
                context_package,
                intents,
                LLMClientError("Context contains no valid citation IDs"),
            )
        response_schema = _grounded_statement_schema(valid_ids)
        prompt = {
            "question": query,
            "intents": list(intents or []),
            "evidence_context": context,
            "requirements": [
                "只使用证据内容，不使用外部医学知识",
                "把答案拆成若干条独立事实，每条事实分别填写text和citation_ids",
                "citation_ids只能从本次证据中的有效编号选择",
                "text中不要手写[E1]，引用格式由应用程序统一渲染",
                "证据不足时明确说明，不猜测诊断",
                "输出中文并简洁回答；免责声明由应用程序追加",
            ],
            "output_schema": {
                "statements": [
                    {"text": "证据支持的一条事实", "citation_ids": ["E1"]}
                ]
            },
        }
        try:
            response = await self.client.complete_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是医疗知识库问答助手。严格基于给定证据回答，"
                            "不得补充证据之外的医学结论。只输出JSON对象。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False),
                    },
                ],
                max_tokens=self.max_tokens,
                schema=response_schema,
                schema_name="grounded_medical_answer",
            )
            try:
                answer, cited_ids = _render_grounded_statements(response, valid_ids)
            except LLMClientError as validation_error:
                repair_prompt = {
                    "question": query,
                    "draft_response": response,
                    "evidence_context": context,
                    "valid_citation_ids": sorted(
                        valid_ids,
                        key=_citation_sort_key,
                    ),
                    "problem": str(validation_error),
                    "requirements": [
                        "保留证据支持的内容，不新增任何医学事实",
                        "每一项都必须有非空text和至少一个citation_ids",
                        "citation_ids只能使用valid_citation_ids中的编号",
                        "text中不要手写引用标记",
                        "只输出JSON对象",
                    ],
                    "output_schema": {
                        "statements": [
                            {"text": "修复后的事实", "citation_ids": ["E1"]}
                        ]
                    },
                }
                response = await self.client.complete_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是回答格式修复器。只能依据给定证据修复引用，"
                                "不得加入新事实。只输出JSON对象。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(repair_prompt, ensure_ascii=False),
                        },
                    ],
                    max_tokens=self.max_tokens,
                    schema=response_schema,
                    schema_name="repaired_medical_answer",
                )
                answer, cited_ids = _render_grounded_statements(response, valid_ids)
        except Exception as exc:
            return await self._fallback(query, context_package, intents, exc)

        selected_citations = [
            item for item in citations if _clean(item.get("citation_id")) in cited_ids
        ]
        selected_evidence = [
            item for item in evidence if _clean(item.get("citation_id")) in cited_ids
        ]
        sources: list[str] = []
        evidence_ids: list[str] = []
        for item in selected_evidence:
            item_sources = item.get("sources", [item.get("source")])
            if isinstance(item_sources, Sequence) and not isinstance(
                item_sources, (str, bytes)
            ):
                for source in item_sources:
                    cleaned = _clean(source)
                    if cleaned and cleaned not in sources:
                        sources.append(cleaned)
            evidence_id = _clean(item.get("evidence_id"))
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        if DISCLAIMER not in answer:
            answer = f"{answer}\n\n{DISCLAIMER}"
        return GeneratedAnswer(
            answer=answer,
            model=self.client.model,
            mode="llm_grounded",
            citations=selected_citations,
            sources=sources,
            used_evidence_ids=evidence_ids,
        )
