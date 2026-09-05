from __future__ import annotations

import unittest

from src.evaluation.judge_runner import aggregate_judge_results


class JudgeRunnerTests(unittest.TestCase):
    def test_hallucination_reduction_is_computed_from_paired_modes(self) -> None:
        results = [
            {
                "retrieval_mode": "vector",
                "status": "completed",
                "hallucinated": True,
                "judge": {
                    "faithfulness": 0.6,
                    "relevance": 0.9,
                    "completeness": 0.7,
                },
            },
            {
                "retrieval_mode": "vector",
                "status": "completed",
                "hallucinated": False,
                "judge": {
                    "faithfulness": 1,
                    "relevance": 1,
                    "completeness": 1,
                },
            },
            {
                "retrieval_mode": "hybrid",
                "status": "completed",
                "hallucinated": False,
                "judge": {
                    "faithfulness": 1,
                    "relevance": 1,
                    "completeness": 1,
                },
            },
            {
                "retrieval_mode": "hybrid",
                "status": "completed",
                "hallucinated": False,
                "judge": {
                    "faithfulness": 0.9,
                    "relevance": 1,
                    "completeness": 0.8,
                },
            },
        ]

        summary = aggregate_judge_results(
            results, faithfulness_threshold=0.8
        )

        self.assertEqual(
            summary["vector_only"]["semantic_hallucination_case_rate"], 0.5
        )
        self.assertEqual(
            summary["hybrid"]["semantic_hallucination_case_rate"], 0.0
        )
        self.assertEqual(summary["hallucination_relative_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()
