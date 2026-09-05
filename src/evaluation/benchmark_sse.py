from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "performance" / "sse_ttfb.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "performance_report.md"


def analyze_sse_lines(
    lines: Sequence[tuple[float, str]],
    *,
    headers_ms: float,
    complete_ms: float,
) -> dict[str, object]:
    first_byte_ms: float | None = None
    first_event_ms: float | None = None
    first_answer_token_ms: float | None = None
    done_event_ms: float | None = None
    event_names: list[str] = []
    for elapsed_ms, line in lines:
        stripped = line.strip()
        if stripped and first_byte_ms is None:
            first_byte_ms = elapsed_ms
        if not stripped.startswith("event:"):
            continue
        name = stripped.removeprefix("event:").strip()
        event_names.append(name)
        if first_event_ms is None:
            first_event_ms = elapsed_ms
        if name == "token" and first_answer_token_ms is None:
            first_answer_token_ms = elapsed_ms
        if name == "done":
            done_event_ms = elapsed_ms
    status = "completed" if "done" in event_names else "failed"
    if "error" in event_names:
        status = "failed"
    return {
        "status": status,
        "headers_ms": round(headers_ms, 3),
        "first_byte_ms": round(first_byte_ms, 3) if first_byte_ms is not None else None,
        "first_sse_event_ms": (
            round(first_event_ms, 3) if first_event_ms is not None else None
        ),
        "first_answer_token_ms": (
            round(first_answer_token_ms, 3)
            if first_answer_token_ms is not None
            else None
        ),
        "done_event_ms": round(done_event_ms, 3) if done_event_ms is not None else None,
        "complete_ms": round(complete_ms, 3),
        "events": event_names,
    }


def measure_sse_request(
    url: str,
    payload: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    body = dict(payload)
    body["stream"] = True
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            headers_ms = (time.perf_counter() - started) * 1000
            status_code = getattr(response, "status", 200)
            lines: list[tuple[float, str]] = []
            while True:
                raw = response.readline()
                if not raw:
                    break
                elapsed_ms = (time.perf_counter() - started) * 1000
                lines.append((elapsed_ms, raw.decode("utf-8", errors="replace")))
            complete_ms = (time.perf_counter() - started) * 1000
        result = analyze_sse_lines(
            lines, headers_ms=headers_ms, complete_ms=complete_ms
        )
        result["http_status"] = status_code
        return result
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "complete_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def summarize_samples(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    completed = [item for item in samples if item.get("status") == "completed"]
    metrics: dict[str, object] = {}
    for field in (
        "headers_ms",
        "first_byte_ms",
        "first_sse_event_ms",
        "first_answer_token_ms",
        "done_event_ms",
        "complete_ms",
    ):
        values = [
            float(item[field])
            for item in completed
            if isinstance(item.get(field), (int, float))
        ]
        metrics[field] = {
            "count": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "min": round(min(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
        }
    return {
        "requests": len(samples),
        "completed": len(completed),
        "failed": len(samples) - len(completed),
        "metrics_ms": metrics,
        "definitions": {
            "first_byte_ms": "TTFB: first non-empty SSE response line",
            "first_sse_event_ms": "first named SSE event, normally plan",
            "first_answer_token_ms": "first user-visible answer token after retrieval and generation",
        },
    }


def _display(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.1f} ms"


def render_markdown(report: Mapping[str, object]) -> str:
    metadata = dict(report.get("metadata", {}))
    summary = dict(report.get("summary", {}))
    metrics = dict(summary.get("metrics_ms", {}))
    lines = [
        "# SSE 性能基准报告",
        "",
        f"- 生成时间（UTC）：{metadata.get('generated_at')}",
        f"- 接口：`{metadata.get('url')}`",
        f"- 正式请求/预热：{summary.get('requests')}/{metadata.get('warmups')}",
        f"- 成功/失败：{summary.get('completed')}/{summary.get('failed')}",
        "",
        "| 指标 | P50 | P95 | 定义 |",
        "|---|---:|---:|---|",
    ]
    labels = (
        ("first_byte_ms", "TTFB", "收到首个非空SSE响应行"),
        ("first_sse_event_ms", "首SSE事件", "通常是plan，不代表答案已经生成"),
        ("first_answer_token_ms", "首回答Token", "检索、重排和生成后首段答案"),
        ("complete_ms", "完整响应", "收到完整SSE响应"),
    )
    for field, label, definition in labels:
        value = dict(metrics.get(field, {}))
        lines.append(
            f"| {label} | {_display(value.get('p50'))} | "
            f"{_display(value.get('p95'))} | {definition} |"
        )
    lines.extend(
        [
            "",
            "简历中的TTFB必须引用本报告的实测P50/P95，并与“首回答Token”明确区分。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.requests <= 0 or args.warmups < 0 or args.timeout <= 0:
        raise ValueError("requests/timeout must be positive and warmups non-negative")
    payload = {
        "query": args.query,
        "disease_name": args.disease_name,
        "top_k": args.top_k,
        "vector_top_k": args.vector_top_k,
        "graph_top_k": args.graph_top_k,
    }
    for index in range(args.warmups):
        sample = measure_sse_request(
            args.url, payload, timeout_seconds=args.timeout
        )
        if sample.get("status") != "completed":
            raise RuntimeError(f"Warmup {index + 1} failed: {sample}")
    samples: list[dict[str, object]] = []
    for index in range(args.requests):
        sample = measure_sse_request(
            args.url, payload, timeout_seconds=args.timeout
        )
        sample["request_index"] = index + 1
        samples.append(sample)
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{args.requests}",
                    "status": sample.get("status"),
                    "ttfb_ms": sample.get("first_byte_ms"),
                    "first_answer_token_ms": sample.get("first_answer_token_ms"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "url": args.url,
            "query": args.query,
            "disease_name": args.disease_name,
            "warmups": args.warmups,
        },
        "summary": summarize_samples(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure real SSE TTFB and first answer token latency"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--query", default="百日咳有哪些典型症状？")
    parser.add_argument("--disease-name", default="百日咳")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-top-k", type=int, default=6)
    parser.add_argument("--graph-top-k", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    report = run(parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
