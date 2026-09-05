from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.judge import JudgeRequest, OpenAICompatibleLLMJudge
from src.llm.client import OpenAICompatibleChatClient
from src.retrieval.neo4j_client import load_project_environment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "results" / "evaluation" / "phase9_metrics.judge_requests.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "evaluation" / "phase9_metrics.judge_results.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "results" / "evaluation" / "phase9_judge_summary.json"
)
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "llm_judge_report.md"


def load_judge_requests(path: Path) -> list[JudgeRequest]:
    if not path.is_file():
        raise FileNotFoundError(f"Judge request file not found: {path}")
    requests: list[JudgeRequest] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                request = JudgeRequest.from_mapping(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid judge request at {path}:{line_no}: {exc}") from exc
            key = (request.case_id, request.retrieval_mode)
            if key in seen:
                raise ValueError(f"Duplicate judge request: {key}")
            seen.add(key)
            requests.append(request)
    if not requests:
        raise ValueError("Judge request file must not be empty")
    return requests


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def aggregate_judge_results(
    results: Sequence[Mapping[str, object]],
    *,
    faithfulness_threshold: float,
) -> dict[str, object]:
    if faithfulness_threshold < 0 or faithfulness_threshold > 1:
        raise ValueError("faithfulness_threshold must be between 0 and 1")
    modes: dict[str, dict[str, object]] = {}
    for mode in ("vector", "hybrid"):
        completed = [
            item
            for item in results
            if item.get("status") == "completed"
            and item.get("retrieval_mode") == mode
        ]
        judge_values = [
            dict(item.get("judge"))
            for item in completed
            if isinstance(item.get("judge"), Mapping)
        ]
        hallucinated = [
            item
            for item in completed
            if bool(item.get("hallucinated"))
        ]
        modes[mode] = {
            "completed": len(completed),
            "failed": sum(
                item.get("status") == "failed"
                and item.get("retrieval_mode") == mode
                for item in results
            ),
            "average_faithfulness": _mean(
                [float(item.get("faithfulness", 0)) for item in judge_values]
            ),
            "average_relevance": _mean(
                [float(item.get("relevance", 0)) for item in judge_values]
            ),
            "average_completeness": _mean(
                [float(item.get("completeness", 0)) for item in judge_values]
            ),
            "semantic_hallucination_case_rate": (
                round(len(hallucinated) / len(completed), 6) if completed else None
            ),
        }
    vector_rate = modes["vector"]["semantic_hallucination_case_rate"]
    hybrid_rate = modes["hybrid"]["semantic_hallucination_case_rate"]
    relative_reduction: float | None = None
    absolute_reduction: float | None = None
    if isinstance(vector_rate, (int, float)) and isinstance(
        hybrid_rate, (int, float)
    ):
        absolute_reduction = round(float(vector_rate) - float(hybrid_rate), 6)
        if float(vector_rate) > 0:
            relative_reduction = round(
                (float(vector_rate) - float(hybrid_rate)) / float(vector_rate), 6
            )
    return {
        "faithfulness_threshold": faithfulness_threshold,
        "vector_only": modes["vector"],
        "hybrid": modes["hybrid"],
        "hallucination_absolute_reduction": absolute_reduction,
        "hallucination_relative_reduction": relative_reduction,
        "metric_definition": (
            "A case is marked hallucinated when the judge lists one or more "
            "unsupported claims or faithfulness is below the configured threshold."
        ),
    }


def _percent(value: object) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.1f}%"


def render_markdown(report: Mapping[str, object]) -> str:
    summary = dict(report.get("summary", {}))
    vector = dict(summary.get("vector_only", {}))
    hybrid = dict(summary.get("hybrid", {}))
    metadata = dict(report.get("metadata", {}))
    lines = [
        "# LLM-as-a-Judge 语义评测报告",
        "",
        f"- 生成时间（UTC）：{metadata.get('generated_at')}",
        f"- Judge模型：`{metadata.get('model')}`",
        f"- Prompt版本：`{metadata.get('prompt_version')}`",
        f"- 完成/失败：{metadata.get('completed_requests')}/{metadata.get('failed_requests')}",
        f"- Faithfulness阈值：{summary.get('faithfulness_threshold')}",
        "- 评测口径：存在Unsupported Claim，或Faithfulness低于阈值，即记为语义幻觉题目。",
        "",
        "| 指标 | Vector-only | Hybrid |",
        "|---|---:|---:|",
        f"| 平均Faithfulness | {_percent(vector.get('average_faithfulness'))} | {_percent(hybrid.get('average_faithfulness'))} |",
        f"| 平均Relevance | {_percent(vector.get('average_relevance'))} | {_percent(hybrid.get('average_relevance'))} |",
        f"| 平均Completeness | {_percent(vector.get('average_completeness'))} | {_percent(hybrid.get('average_completeness'))} |",
        f"| 语义幻觉题目占比 | {_percent(vector.get('semantic_hallucination_case_rate'))} | {_percent(hybrid.get('semantic_hallucination_case_rate'))} |",
        "",
        f"- 幻觉率绝对下降：{_percent(summary.get('hallucination_absolute_reduction'))}",
        f"- 幻觉率相对下降：{_percent(summary.get('hallucination_relative_reduction'))}",
        "",
        "本报告是模型评审结果，不是临床安全认证。当前固定集来自项目清洗数据，尚未经过独立临床专家标注；若生成与Judge使用同一模型，还应注明可能存在自评偏差。",
        "",
    ]
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, object]:
    requests = load_judge_requests(args.input)
    if args.case_id:
        requests = [item for item in requests if item.case_id == args.case_id]
    if args.mode:
        requests = [item for item in requests if item.retrieval_mode == args.mode]
    if (args.case_id or args.mode) and not requests:
        raise ValueError("No Judge requests matched --case-id/--mode")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        requests = requests[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_by_key: dict[tuple[str, str], dict[str, object]] = {}
    if args.resume and args.output.is_file():
        with args.output.open(encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, Mapping) or item.get("status") != "completed":
                    continue
                key = (str(item.get("case_id")), str(item.get("retrieval_mode")))
                completed_by_key[key] = dict(item)
    else:
        args.output.write_text("", encoding="utf-8")
    request_keys = {(item.case_id, item.retrieval_mode) for item in requests}
    completed_by_key = {
        key: item for key, item in completed_by_key.items() if key in request_keys
    }
    pending_requests = [
        item
        for item in requests
        if (item.case_id, item.retrieval_mode) not in completed_by_key
    ]
    client = OpenAICompatibleChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=os.getenv("JUDGE_LLM_API_KEY") or os.getenv("LLM_API_KEY"),
        timeout_seconds=args.timeout,
        max_retries=args.retries,
    )
    judge = OpenAICompatibleLLMJudge(client)
    semaphore = asyncio.Semaphore(args.concurrency)
    output_lock = asyncio.Lock()
    already_completed = len(completed_by_key)

    async def evaluate(index: int, request: JudgeRequest) -> dict[str, object]:
        started = time.perf_counter()
        async with semaphore:
            try:
                result = dict(await judge.judge(request))
                unsupported = result.get("unsupported_claims", [])
                hallucinated = (
                    float(result.get("faithfulness", 0))
                    < args.faithfulness_threshold
                    or bool(unsupported)
                )
                output = {
                    "case_id": request.case_id,
                    "retrieval_mode": request.retrieval_mode,
                    "status": "completed",
                    "model": args.model,
                    "prompt_version": judge.prompt_version,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "hallucinated": hallucinated,
                    "judge": result,
                }
            except Exception as exc:
                output = {
                    "case_id": request.case_id,
                    "retrieval_mode": request.retrieval_mode,
                    "status": "failed",
                    "model": args.model,
                    "prompt_version": judge.prompt_version,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            async with output_lock:
                with args.output.open("a", encoding="utf-8") as sink:
                    sink.write(json.dumps(output, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "progress": f"{already_completed + index}/{len(requests)}",
                        "case_id": request.case_id,
                        "mode": request.retrieval_mode,
                        "status": output["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return output

    new_results = list(
        await asyncio.gather(
            *(
                evaluate(index, request)
                for index, request in enumerate(pending_requests, 1)
            )
        )
    )
    results_by_key = dict(completed_by_key)
    results_by_key.update(
        {
            (str(item.get("case_id")), str(item.get("retrieval_mode"))): item
            for item in new_results
        }
    )
    results = [
        results_by_key[(request.case_id, request.retrieval_mode)]
        for request in requests
        if (request.case_id, request.retrieval_mode) in results_by_key
    ]
    summary = aggregate_judge_results(
        results, faithfulness_threshold=args.faithfulness_threshold
    )
    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "prompt_version": judge.prompt_version,
            "input": str(args.input),
            "request_count": len(requests),
            "completed_requests": sum(item["status"] == "completed" for item in results),
            "failed_requests": sum(item["status"] == "failed" for item in results),
        },
        "summary": summary,
    }
    args.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n",
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real configured LLM judge over exported RAG answers"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--base-url",
        default=os.getenv("JUDGE_LLM_BASE_URL") or os.getenv("LLM_BASE_URL"),
    )
    parser.add_argument(
        "--model", default=os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", help="Judge one exact evaluation case")
    parser.add_argument("--mode", choices=("vector", "hybrid"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=float(os.getenv("JUDGE_LLM_FAITHFULNESS_THRESHOLD", "0.8")),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit after writing reports if any request failed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed rows already present in --output",
    )
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error(
            "Configure JUDGE_LLM_BASE_URL/LLM_BASE_URL and "
            "JUDGE_LLM_MODEL/LLM_MODEL"
        )
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than zero")
    return args


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    load_project_environment()
    args = parse_args()
    report = asyncio.run(run(args))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.fail_on_error and report["metadata"]["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
