from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src.agent.query_planner import QueryPlan
from src.agent.tool_schemas import empty_metadata_filter, validate_tool_call
from src.api.runtime import LocalMedicalRuntime
from src.evaluation.judge import JudgeRequest, build_judge_request
from src.evaluation.metrics import evaluate_answer, evaluate_retrieval
from src.evaluation.schemas import EvaluationCase, load_evaluation_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "medical_qa_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "evaluation" / "phase9_metrics.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "evaluation_report.md"


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _forced_plan(
    base: QueryPlan,
    case: EvaluationCase,
    *,
    mode: str,
    top_k: int,
    vector_top_k: int,
    graph_top_k: int,
) -> QueryPlan:
    if base.status != "ready":
        raise RuntimeError(
            f"Planner could not resolve evaluation case {case.case_id}: {base.reason}"
        )
    metadata_filter = empty_metadata_filter(disease_name=case.disease_name)
    if mode == "vector":
        validated = validate_tool_call(
            "vector_search",
            {
                "query": case.question,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            },
        )
        reason = "Evaluation baseline: metadata-filtered Milvus retrieval only."
    elif mode == "hybrid":
        validated = validate_tool_call(
            "hybrid_search",
            {
                "query": case.question,
                "disease_name": case.disease_name,
                "top_k": top_k,
                "vector_top_k": vector_top_k,
                "graph_top_k": graph_top_k,
                "metadata_filter": metadata_filter,
                "relations": [case.relation],
                "allow_partial": False,
            },
        )
        reason = "Evaluation candidate: Milvus plus the intent-matched Neo4j relation."
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    return replace(
        base,
        intents=(case.intent,),
        selected_tool=validated.name,
        tool_call=validated.to_dict(),
        reason=reason,
    )


async def _evaluate_mode(
    service: object,
    case: EvaluationCase,
    *,
    mode: str,
    hit_k: int,
    top_k: int,
    vector_top_k: int,
    graph_top_k: int,
) -> tuple[dict[str, object], JudgeRequest]:
    agent = getattr(service, "agent")
    answer_generator = getattr(service, "answer_generator")
    base = agent.planner.plan(case.question, disease_name=case.disease_name)
    plan = _forced_plan(
        base,
        case,
        mode=mode,
        top_k=top_k,
        vector_top_k=vector_top_k,
        graph_top_k=graph_top_k,
    )
    started = time.perf_counter()
    agent_result = await agent.arun_plan(plan)
    if agent_result.get("status") != "evidence_ready":
        raise RuntimeError(
            f"Case {case.case_id}/{mode} did not produce evidence: "
            f"{agent_result.get('status')}"
        )

    retrieval = _mapping(agent_result.get("retrieval"))
    raw_evidence = _mapping_list(retrieval.get("evidence"))
    post = _mapping(agent_result.get("post_retrieval"))
    reranked_evidence = _mapping_list(post.get("reranked_evidence"))
    context = _mapping(post.get("context"))
    context_evidence = _mapping_list(context.get("evidence"))
    generated = await answer_generator.generate(
        case.question,
        context,
        intents=[case.intent],
    )
    total_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    tool_calls = _mapping_list(agent_result.get("tool_calls"))
    tool_elapsed_ms = (
        float(tool_calls[0].get("elapsed_ms", 0.0)) if tool_calls else 0.0
    )

    mode_result: dict[str, object] = {
        "status": "completed",
        "selected_tool": plan.selected_tool,
        "raw_retrieval": evaluate_retrieval(case, raw_evidence, k=hit_k),
        "reranked_retrieval": evaluate_retrieval(
            case, reranked_evidence, k=hit_k
        ),
        "answer_metrics": evaluate_answer(
            case,
            generated.answer,
            generated.citations,
            context_evidence,
        ),
        "answer": generated.answer,
        "answer_generation": {
            "model": generated.model,
            "mode": generated.mode,
            "used_evidence_ids": generated.used_evidence_ids,
        },
        "citations": generated.citations,
        "top_evidence": [
            {
                key: item.get(key)
                for key in (
                    "evidence_id",
                    "evidence_type",
                    "disease_name",
                    "relation",
                    "entity_name",
                    "retrieval_rank",
                    "rerank_rank",
                    "rerank_score",
                )
            }
            for item in reranked_evidence[:hit_k]
        ],
        "retrieval_stats": retrieval.get("stats"),
        "context_budget": context.get("budget"),
        "latency_ms": {
            "tool_and_post_retrieval": tool_elapsed_ms,
            "end_to_end_with_answer": total_elapsed_ms,
        },
    }
    judge_request = build_judge_request(
        case,
        retrieval_mode=mode,
        answer=generated.answer,
        evidence=context_evidence,
    )
    return mode_result, judge_request


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _aggregate_mode(cases: Sequence[Mapping[str, object]], mode: str) -> dict[str, object]:
    completed = [
        _mapping(_mapping(case.get("modes")).get(mode))
        for case in cases
        if _mapping(_mapping(case.get("modes")).get(mode)).get("status")
        == "completed"
    ]
    raw = [_mapping(item.get("raw_retrieval")) for item in completed]
    reranked = [_mapping(item.get("reranked_retrieval")) for item in completed]
    answers = [_mapping(item.get("answer_metrics")) for item in completed]
    latencies = [_mapping(item.get("latency_ms")) for item in completed]
    return {
        "completed_cases": len(completed),
        "raw_hit_at_3": _mean([float(item.get("hit_at_k", False)) for item in raw]),
        "raw_mrr": _mean([float(item.get("reciprocal_rank", 0.0)) for item in raw]),
        "raw_gold_recall_at_3": _mean(
            [float(item.get("gold_recall_at_k", 0.0)) for item in raw]
        ),
        "reranked_hit_at_3": _mean(
            [float(item.get("hit_at_k", False)) for item in reranked]
        ),
        "reranked_mrr": _mean(
            [float(item.get("reciprocal_rank", 0.0)) for item in reranked]
        ),
        "reranked_gold_recall_at_3": _mean(
            [float(item.get("gold_recall_at_k", 0.0)) for item in reranked]
        ),
        "metadata_accuracy_at_3": _mean(
            [float(item.get("metadata_accuracy_at_k", 0.0)) for item in reranked]
        ),
        "graph_relation_hit_at_3": _mean(
            [float(item.get("graph_relation_hit_at_k", False)) for item in reranked]
        ),
        "citation_validity_rate": _mean(
            [float(item.get("citation_validity", False)) for item in answers]
        ),
        "statement_citation_coverage": _mean(
            [float(item.get("statement_citation_coverage", 0.0)) for item in answers]
        ),
        "unsupported_claim_rate": _mean(
            [float(item.get("unsupported_claim_rate", 0.0)) for item in answers]
        ),
        "answer_gold_entity_coverage": _mean(
            [float(item.get("gold_entity_coverage", 0.0)) for item in answers]
        ),
        "forbidden_output_case_rate": _mean(
            [float(bool(item.get("forbidden_hits"))) for item in answers]
        ),
        "average_tool_latency_ms": _mean(
            [float(item.get("tool_and_post_retrieval", 0.0)) for item in latencies]
        ),
        "average_end_to_end_latency_ms": _mean(
            [float(item.get("end_to_end_with_answer", 0.0)) for item in latencies]
        ),
    }


def _summary(case_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    vector = _aggregate_mode(case_results, "vector")
    hybrid = _aggregate_mode(case_results, "hybrid")
    delta_fields = (
        "raw_hit_at_3",
        "reranked_hit_at_3",
        "reranked_gold_recall_at_3",
        "answer_gold_entity_coverage",
        "unsupported_claim_rate",
    )
    delta = {
        field: round(float(hybrid[field]) - float(vector[field]), 6)
        for field in delta_fields
    }
    return {"vector_only": vector, "hybrid": hybrid, "hybrid_minus_vector": delta}


def _percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _render_markdown(report: Mapping[str, object]) -> str:
    metadata = _mapping(report.get("metadata"))
    summary = _mapping(report.get("summary"))
    vector = _mapping(summary.get("vector_only"))
    hybrid = _mapping(summary.get("hybrid"))
    judge = _mapping(metadata.get("llm_as_a_judge"))
    cases = _mapping_list(report.get("cases"))
    lines = [
        "# 第 9 阶段：检索与回答评估报告",
        "",
        f"- 生成时间（UTC）：{metadata.get('generated_at')}",
        f"- 固定测试集：`{metadata.get('dataset')}`（{metadata.get('case_count')} 题）",
        f"- Milvus 集合：`{metadata.get('milvus_collection')}`",
        "- 测试集来源：从本项目清洗后的核心关系三元组抽取，尚未经过独立临床专家标注。",
        (
            "- 大模型裁判（LLM-as-a-Judge）："
            f"{judge.get('status', 'not_configured')}；已导出"
            f"{judge.get('exported_requests', 0)}条待评请求。没有真实裁判结果时不报告语义幻觉率。"
        ),
        "",
        "## 核心结果",
        "",
        "| 指标 | 纯向量检索 | 混合检索（Milvus + Neo4j） |",
        "|---|---:|---:|",
    ]
    rows = (
        ("原始 Top-3 命中率", "raw_hit_at_3"),
        ("重排后 Top-3 命中率", "reranked_hit_at_3"),
        ("重排后参考实体召回率@3", "reranked_gold_recall_at_3"),
        ("元数据正确率@3", "metadata_accuracy_at_3"),
        ("图关系命中率@3", "graph_relation_hit_at_3"),
        ("引用有效率", "citation_validity_rate"),
        ("回答参考实体覆盖率", "answer_gold_entity_coverage"),
        ("无证据陈述率（规则代理）", "unsupported_claim_rate"),
        ("脏词输出题目占比", "forbidden_output_case_rate"),
    )
    for label, field in rows:
        lines.append(f"| {label} | {_percent(vector.get(field, 0))} | {_percent(hybrid.get(field, 0))} |")
    lines.extend(
        [
            "",
            "## 平均链路耗时",
            "",
            "| 指标 | 纯向量检索 | 混合检索（Milvus + Neo4j） |",
            "|---|---:|---:|",
            f"| 工具与检索后处理 | {float(vector.get('average_tool_latency_ms', 0)):.1f} ms | {float(hybrid.get('average_tool_latency_ms', 0)):.1f} ms |",
            f"| 检索到回答生成完成 | {float(vector.get('average_end_to_end_latency_ms', 0)):.1f} ms | {float(hybrid.get('average_end_to_end_latency_ms', 0)):.1f} ms |",
            "",
            "Top-3 命中定义：前三条证据至少包含一个固定参考实体，或包含“正确疾病 + 正确核心关系”的 Neo4j 事实。参考实体召回率@3只计算固定参考实体子集，因此图谱返回同关系下其他有效实体时可能命中成功，但召回率不增加。",
            "",
            "结果解读：扩展集上混合检索提高了 Top-3 命中率、图关系覆盖率和回答参考实体覆盖率；其参考实体召回率@3低于纯向量检索，说明图谱也会补入同一关系下正确但不在固定参考子集中的实体。后续仍需独立专家标注和融合权重实验。",
            "",
            "`无证据陈述率`只检查编号陈述是否带有效引用，以及陈述能否直接回查到引用证据；它是可复现的规则代理，不等于真正的语义幻觉率。",
            "",
            "## 每题 Top-3 结果",
            "",
            "| 用例 | 意图 | 纯向量：原始/重排 | 混合检索：原始/重排 | 混合检索回答覆盖率 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for case in cases:
        modes = _mapping(case.get("modes"))
        vector_result = _mapping(modes.get("vector"))
        hybrid_result = _mapping(modes.get("hybrid"))
        vector_raw = _mapping(vector_result.get("raw_retrieval"))
        vector_rerank = _mapping(vector_result.get("reranked_retrieval"))
        hybrid_raw = _mapping(hybrid_result.get("raw_retrieval"))
        hybrid_rerank = _mapping(hybrid_result.get("reranked_retrieval"))
        hybrid_answer = _mapping(hybrid_result.get("answer_metrics"))
        lines.append(
            f"| {case.get('case_id')} | {case.get('intent')} | "
            f"{int(bool(vector_raw.get('hit_at_k')))}/{int(bool(vector_rerank.get('hit_at_k')))} | "
            f"{int(bool(hybrid_raw.get('hit_at_k')))}/{int(bool(hybrid_rerank.get('hit_at_k')))} | "
            f"{_percent(hybrid_answer.get('gold_entity_coverage', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 如何复现",
            "",
            "确保 Milvus、Neo4j 已启动，并在项目环境中执行：",
            "",
            "```powershell",
            "python -m src.evaluation.runner",
            "```",
            "",
            "完整机器可读结果位于 `results/evaluation/phase9_metrics.json`；待外部裁判模型评审的输入位于同目录的 `phase9_metrics.judge_requests.jsonl`。",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, object]:
    if args.hit_k != 3:
        raise ValueError("Phase 9 report currently requires --hit-k 3")
    if args.top_k < args.hit_k:
        raise ValueError("--top-k must be greater than or equal to --hit-k")
    cases = load_evaluation_cases(args.dataset)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise ValueError(f"Unknown --case-id: {args.case_id}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        cases = cases[: args.limit]

    runtime = LocalMedicalRuntime()
    service = runtime.start()
    case_results: list[dict[str, object]] = []
    judge_requests: list[JudgeRequest] = []
    try:
        for index, case in enumerate(cases, start=1):
            modes: dict[str, object] = {}
            for mode in ("vector", "hybrid"):
                try:
                    result, judge_request = await _evaluate_mode(
                        service,
                        case,
                        mode=mode,
                        hit_k=args.hit_k,
                        top_k=args.top_k,
                        vector_top_k=args.vector_top_k,
                        graph_top_k=args.graph_top_k,
                    )
                    modes[mode] = result
                    judge_requests.append(judge_request)
                except Exception as exc:
                    modes[mode] = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    if args.fail_fast:
                        raise
            case_results.append({**case.to_dict(), "modes": modes})
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(cases)}",
                        "case_id": case.case_id,
                        "vector": _mapping(modes.get("vector")).get("status"),
                        "hybrid": _mapping(modes.get("hybrid")).get("status"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        runtime.close()

    generated_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, object] = {
        "metadata": {
            "phase": 9,
            "generated_at": generated_at,
            "dataset": _display_path(args.dataset),
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "case_count": len(cases),
            "hit_k": args.hit_k,
            "top_k": args.top_k,
            "vector_top_k": args.vector_top_k,
            "graph_top_k": args.graph_top_k,
            "milvus_collection": os.getenv("MILVUS_COLLECTION"),
            "embedding_model": os.getenv("BGE_MODEL_PATH"),
            "reranker_model": os.getenv("BGE_RERANKER_MODEL_PATH"),
            "test_set_provenance": (
                "Derived from cleaned project core-relation triples; "
                "not independently clinician-annotated"
            ),
            "llm_as_a_judge": {
                "status": "not_configured",
                "exported_requests": len(judge_requests),
            },
        },
        "summary": _summary(case_results),
        "cases": case_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    judge_path = args.output.with_name(f"{args.output.stem}.judge_requests.jsonl")
    judge_path.write_text(
        "\n".join(
            json.dumps(request.to_dict(), ensure_ascii=False)
            for request in judge_requests
        )
        + ("\n" if judge_requests else ""),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Vector-only and GraphRAG retrieval on a fixed QA set"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-id",
        help="Run one exact evaluation case, for example case_007_symptom",
    )
    parser.add_argument("--hit-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-top-k", type=int, default=6)
    parser.add_argument("--graph-top-k", type=int, default=6)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    report = asyncio.run(run(parse_args()))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
