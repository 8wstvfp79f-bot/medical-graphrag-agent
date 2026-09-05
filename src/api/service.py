from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence

from src.agent.orchestrator import GraphRAGAgent
from src.agent.query_planner import QueryPlan
from src.generation.answer_generator import AnswerGenerator


class MedicalChatService:
    """Async application service shared by JSON and SSE API responses."""

    def __init__(
        self,
        agent: GraphRAGAgent,
        answer_generator: AnswerGenerator,
        *,
        max_concurrent_requests: int = 1,
    ) -> None:
        if max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be greater than zero")
        self.agent = agent
        self.answer_generator = answer_generator
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

    def _plan(self, request: Mapping[str, object]) -> QueryPlan:
        metadata_filter = request.get("metadata_filter")
        return self.agent.planner.plan(
            str(request.get("query", "")),
            disease_name=request.get("disease_name"),
            metadata_filter=(
                metadata_filter if isinstance(metadata_filter, Mapping) else None
            ),
            top_k=int(request.get("top_k", 10)),
            vector_top_k=int(request.get("vector_top_k", 6)),
            graph_top_k=int(request.get("graph_top_k", 6)),
            allow_partial=bool(request.get("allow_partial", False)),
        )

    async def _execute(self, plan: QueryPlan, request_id: str) -> dict[str, object]:
        async with self._semaphore:
            result = await self.agent.arun_plan(plan)
        result["request_id"] = request_id
        return result

    async def _response(
        self,
        query: str,
        agent_result: Mapping[str, object],
    ) -> dict[str, object]:
        plan_value = agent_result.get("query_plan")
        plan = dict(plan_value) if isinstance(plan_value, Mapping) else {}
        if agent_result.get("status") != "evidence_ready":
            clarification = str(
                plan.get("clarification_question")
                or "请补充更明确的疾病名称后再查询。"
            )
            return {
                "status": agent_result.get("status", "needs_clarification"),
                "request_id": agent_result["request_id"],
                "answer": clarification,
                "answer_generation": {
                    "mode": "clarification",
                    "model": None,
                },
                "sources": [],
                "citations": [],
                "evidence": [],
                "retrieved_chunks": [],
                "graph_evidence": [],
                "tool_calls": [],
                "query_plan": plan,
                "retrieval_stats": None,
                "context_budget": None,
            }

        post_value = agent_result.get("post_retrieval")
        post = dict(post_value) if isinstance(post_value, Mapping) else {}
        context_value = post.get("context")
        context = dict(context_value) if isinstance(context_value, Mapping) else {}
        intent_values = plan.get("intents", [])
        intents = (
            [str(intent) for intent in intent_values]
            if isinstance(intent_values, Sequence)
            and not isinstance(intent_values, (str, bytes))
            else []
        )
        generated = await self.answer_generator.generate(
            query,
            context,
            intents=intents,
        )

        retrieval_value = agent_result.get("retrieval")
        retrieval = (
            dict(retrieval_value) if isinstance(retrieval_value, Mapping) else {}
        )
        raw_retrieval_evidence = retrieval.get("evidence", [])
        retrieval_evidence = (
            [dict(item) for item in raw_retrieval_evidence if isinstance(item, Mapping)]
            if isinstance(raw_retrieval_evidence, Sequence)
            else []
        )
        selected_value = context.get("evidence", [])
        selected = (
            [dict(item) for item in selected_value if isinstance(item, Mapping)]
            if isinstance(selected_value, Sequence)
            else []
        )
        return {
            "status": "completed",
            "request_id": agent_result["request_id"],
            "answer": generated.answer,
            "answer_generation": {
                "mode": generated.mode,
                "model": generated.model,
                "used_evidence_ids": generated.used_evidence_ids,
            },
            "sources": generated.sources,
            "citations": generated.citations,
            "evidence": selected,
            "retrieved_chunks": [
                item
                for item in retrieval_evidence
                if item.get("evidence_type") in {"vector", "hybrid"}
            ],
            "graph_evidence": [
                item
                for item in retrieval_evidence
                if item.get("evidence_type") in {"graph", "hybrid"}
            ],
            "tool_calls": list(agent_result.get("tool_calls", [])),
            "query_plan": plan,
            "retrieval_stats": retrieval.get("stats"),
            "context_budget": context.get("budget"),
        }

    async def chat(self, request: Mapping[str, object]) -> dict[str, object]:
        request_id = f"req_{uuid.uuid4().hex}"
        plan = self._plan(request)
        result = await self._execute(plan, request_id)
        return await self._response(str(request["query"]), result)

    async def stream_chat(
        self, request: Mapping[str, object]
    ) -> AsyncIterator[dict[str, object]]:
        request_id = f"req_{uuid.uuid4().hex}"
        try:
            plan = self._plan(request)
            yield {
                "event": "plan",
                "data": {"request_id": request_id, "query_plan": plan.to_dict()},
            }
            if plan.status == "ready" and plan.tool_call:
                yield {
                    "event": "tool_start",
                    "data": {"request_id": request_id, **plan.tool_call},
                }
            result = await self._execute(plan, request_id)
            if result.get("tool_calls"):
                yield {
                    "event": "tool_result",
                    "data": {
                        "request_id": request_id,
                        "tool_calls": result["tool_calls"],
                    },
                }
            response = await self._response(str(request["query"]), result)
            for citation in response["citations"]:
                yield {
                    "event": "citation",
                    "data": {"request_id": request_id, **citation},
                }
            answer = str(response["answer"])
            for start in range(0, len(answer), 48):
                yield {
                    "event": "token",
                    "data": {
                        "request_id": request_id,
                        "text": answer[start : start + 48],
                    },
                }
                await asyncio.sleep(0)
            yield {"event": "done", "data": response}
        except Exception as exc:
            yield {
                "event": "error",
                "data": {
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
