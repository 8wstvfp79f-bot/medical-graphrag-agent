from __future__ import annotations

import unittest

from src.evaluation.benchmark_sse import analyze_sse_lines, summarize_samples


class SSEBenchmarkTests(unittest.TestCase):
    def test_named_events_distinguish_ttfb_from_first_answer_token(self) -> None:
        sample = analyze_sse_lines(
            [
                (12.0, "event: plan\n"),
                (12.1, "data: {}\n"),
                (250.0, "event: tool_result\n"),
                (320.0, "event: token\n"),
                (400.0, "event: done\n"),
            ],
            headers_ms=8.0,
            complete_ms=401.0,
        )

        self.assertEqual(sample["status"], "completed")
        self.assertEqual(sample["first_byte_ms"], 12.0)
        self.assertEqual(sample["first_answer_token_ms"], 320.0)

    def test_summary_reports_p50_and_p95(self) -> None:
        samples = [
            {
                "status": "completed",
                "headers_ms": 5,
                "first_byte_ms": value,
                "first_sse_event_ms": value,
                "first_answer_token_ms": value + 100,
                "done_event_ms": value + 200,
                "complete_ms": value + 201,
            }
            for value in (10, 20, 30)
        ]
        summary = summarize_samples(samples)

        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["metrics_ms"]["first_byte_ms"]["p50"], 20)
        self.assertEqual(summary["metrics_ms"]["first_byte_ms"]["p95"], 29)


if __name__ == "__main__":
    unittest.main()
