from __future__ import annotations

import unittest

from src.agent.query_planner import (
    DeterministicMedicalQueryPlanner,
    DiseaseCatalog,
)


class QueryPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = DiseaseCatalog(["百日咳", "副百日咳", "流感"])
        self.planner = DeterministicMedicalQueryPlanner(self.catalog)

    def test_symptom_intent_routes_to_hybrid(self) -> None:
        plan = self.planner.plan("百日咳有哪些典型症状？")

        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.disease_name, "百日咳")
        self.assertEqual(plan.intents, ("symptom",))
        self.assertEqual(plan.selected_tool, "hybrid_search")
        self.assertEqual(plan.tool_call["arguments"]["relations"], ["has_symptom"])

    def test_multiple_structured_intents_map_to_stable_relations(self) -> None:
        plan = self.planner.plan("百日咳有什么症状，需要做哪些检查和治疗？")

        self.assertEqual(plan.intents, ("symptom", "diagnosis", "treatment"))
        self.assertEqual(
            plan.tool_call["arguments"]["relations"],
            ["has_symptom", "diagnosed_by", "treated_by"],
        )

    def test_cause_intent_routes_to_vector_only(self) -> None:
        plan = self.planner.plan("百日咳的病因是什么？")

        self.assertEqual(plan.intents, ("cause", "overview"))
        self.assertEqual(plan.selected_tool, "vector_search")

    def test_longest_disease_name_wins_over_substring(self) -> None:
        plan = self.planner.plan("副百日咳有哪些症状？")

        self.assertEqual(plan.disease_candidates, ("副百日咳",))
        self.assertEqual(plan.disease_name, "副百日咳")

    def test_no_disease_or_multiple_diseases_asks_for_clarification(self) -> None:
        missing = self.planner.plan("有哪些典型症状？")
        multiple = self.planner.plan("比较百日咳和流感的症状")

        self.assertEqual(missing.status, "needs_clarification")
        self.assertEqual(missing.tool_call, None)
        self.assertIn("疾病全名", missing.clarification_question)
        self.assertEqual(multiple.status, "needs_clarification")
        self.assertEqual(multiple.disease_candidates, ("百日咳", "流感"))

    def test_exact_override_is_validated_against_catalog(self) -> None:
        valid = self.planner.plan("有哪些症状？", disease_name="百日咳")
        unknown = self.planner.plan("有哪些症状？", disease_name="不存在的疾病")

        self.assertEqual(valid.status, "ready")
        self.assertEqual(valid.disease_name, "百日咳")
        self.assertEqual(unknown.status, "needs_clarification")


if __name__ == "__main__":
    unittest.main()
