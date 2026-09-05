import json
import tempfile
import unittest
from pathlib import Path

from src.agent.query_planner import QueryPlan
from src.evaluation.metrics import evaluate_answer, evaluate_retrieval
from src.evaluation.runner import DEFAULT_DATASET, _forced_plan
from src.evaluation.schemas import (
    EvaluationCase,
    EvaluationDatasetError,
    load_evaluation_cases,
)


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="pertussis_symptom",
        question="百日咳有哪些典型症状？",
        disease_name="百日咳",
        intent="symptom",
        relation="has_symptom",
        gold_entities=("痉挛性咳嗽", "胸闷"),
        forbidden_entities=("闫鹏辉",),
    )


class EvaluationTests(unittest.TestCase):
    def test_fixed_dataset_contains_250_valid_cases(self) -> None:
        cases = load_evaluation_cases(DEFAULT_DATASET)
        self.assertEqual(len(cases), 250)
        self.assertEqual(len({case.case_id for case in cases}), 250)
        self.assertEqual(
            {case.intent for case in cases},
            {"symptom", "diagnosis", "treatment", "department", "complication"},
        )

    def test_dataset_rejects_duplicate_case_ids(self) -> None:
        row = {
            "case_id": "duplicate",
            "question": "百日咳有哪些症状？",
            "disease_name": "百日咳",
            "intent": "symptom",
            "relation": "has_symptom",
            "gold_entities": ["咳嗽"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False)
                + "\n"
                + json.dumps(row, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationDatasetError, "Duplicate case_id"):
                load_evaluation_cases(path)

    def test_retrieval_metrics_measure_hit_rank_metadata_and_dirty_terms(self) -> None:
        evidence = [
            {
                "content": "百日咳的痉挛性咳嗽",
                "disease_name": "百日咳",
                "relation": "has_symptom",
                "entity_name": "痉挛性咳嗽",
            },
            {"content": "其他内容", "disease_name": "百日咳"},
        ]
        metrics = evaluate_retrieval(_case(), evidence, k=3)
        self.assertIs(metrics["hit_at_k"], True)
        self.assertEqual(metrics["first_relevant_rank"], 1)
        self.assertEqual(metrics["reciprocal_rank"], 1.0)
        self.assertEqual(metrics["gold_recall_at_k"], 0.5)
        self.assertEqual(metrics["metadata_accuracy_at_k"], 1.0)
        self.assertIs(metrics["graph_relation_hit_at_k"], True)
        self.assertEqual(metrics["forbidden_hits"], [])

    def test_correct_graph_relation_counts_as_relevant_outside_reference_subset(
        self,
    ) -> None:
        metrics = evaluate_retrieval(
            _case(),
            [
                {
                    "content": "百日咳 -[has_symptom]-> 发热",
                    "disease_name": "百日咳",
                    "relation": "has_symptom",
                    "entity_name": "发热",
                }
            ],
            k=3,
        )
        self.assertIs(metrics["hit_at_k"], True)
        self.assertIs(metrics["graph_relation_hit_at_k"], True)
        self.assertEqual(metrics["gold_recall_at_k"], 0.0)

    def test_answer_metrics_validate_grounded_graph_statement(self) -> None:
        answer = (
            "1. 百日咳的相关症状包括“痉挛性咳嗽” [E1]。\n\n"
            "以上内容来自当前知识库检索证据，仅供健康信息参考。"
        )
        citations = [{"citation_id": "E1", "evidence_id": "graph-1"}]
        evidence = [
            {
                "citation_id": "E1",
                "evidence_id": "graph-1",
                "evidence_type": "graph",
                "disease_name": "百日咳",
                "relation": "has_symptom",
                "entity_name": "痉挛性咳嗽",
                "content": "百日咳 -[has_symptom]-> 痉挛性咳嗽",
            }
        ]
        metrics = evaluate_answer(_case(), answer, citations, evidence)
        self.assertIs(metrics["citation_validity"], True)
        self.assertEqual(metrics["statement_citation_coverage"], 1.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.0)
        self.assertEqual(metrics["gold_entity_coverage"], 0.5)

    def test_answer_metrics_reject_missing_and_unsupported_citations(self) -> None:
        metrics = evaluate_answer(
            _case(),
            "1. 百日咳一定会自行痊愈 [E9]。\n2. 无需就医。",
            [{"citation_id": "E1", "evidence_id": "vector-1"}],
            [
                {
                    "citation_id": "E1",
                    "evidence_id": "vector-1",
                    "evidence_type": "vector",
                    "content": "百日咳症状包括痉挛性咳嗽",
                }
            ],
        )
        self.assertIs(metrics["citation_validity"], False)
        self.assertEqual(metrics["invalid_citation_ids"], ["E9"])
        self.assertEqual(metrics["statement_citation_coverage"], 0.5)
        self.assertEqual(metrics["unsupported_claim_rate"], 1.0)

    def test_forced_plans_keep_question_and_change_retrieval_strategy(self) -> None:
        case = _case()
        base = QueryPlan(
            planner="test",
            status="ready",
            query=case.question,
            intents=(case.intent,),
            disease_candidates=(case.disease_name,),
            disease_name=case.disease_name,
            selected_tool="hybrid_search",
            tool_call={},
            reason="test",
        )
        vector = _forced_plan(
            base,
            case,
            mode="vector",
            top_k=10,
            vector_top_k=6,
            graph_top_k=6,
        )
        hybrid = _forced_plan(
            base,
            case,
            mode="hybrid",
            top_k=10,
            vector_top_k=6,
            graph_top_k=6,
        )
        self.assertEqual(vector.query, case.question)
        self.assertEqual(hybrid.query, case.question)
        self.assertEqual(vector.selected_tool, "vector_search")
        self.assertEqual(hybrid.selected_tool, "hybrid_search")
        self.assertEqual(
            hybrid.tool_call["arguments"]["relations"], ["has_symptom"]
        )
        self.assertEqual(
            vector.tool_call["arguments"]["metadata_filter"]["disease_name"],
            ["百日咳"],
        )


if __name__ == "__main__":
    unittest.main()
