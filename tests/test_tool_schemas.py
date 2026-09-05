from __future__ import annotations

import unittest

from src.agent.tool_schemas import (
    MEDICAL_TOOL_SCHEMAS,
    ToolCallValidationError,
    empty_metadata_filter,
    validate_tool_call,
)


class ToolSchemaTests(unittest.TestCase):
    def test_all_tool_schemas_are_strict_and_closed(self) -> None:
        self.assertEqual(len(MEDICAL_TOOL_SCHEMAS), 3)
        for schema in MEDICAL_TOOL_SCHEMAS:
            function = schema["function"]
            parameters = function["parameters"]
            self.assertTrue(function["strict"])
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(
                set(parameters["properties"]), set(parameters["required"])
            )

    def test_valid_hybrid_call_is_normalized(self) -> None:
        call = validate_tool_call(
            "hybrid_search",
            {
                "query": "  百日咳有哪些症状？ ",
                "disease_name": "百日咳",
                "top_k": 10,
                "vector_top_k": 6,
                "graph_top_k": 6,
                "metadata_filter": empty_metadata_filter(disease_name="百日咳"),
                "relations": ["has_symptom", "has_symptom"],
                "allow_partial": False,
            },
        )

        self.assertEqual(call.name, "hybrid_search")
        self.assertEqual(call.arguments["query"], "百日咳有哪些症状？")
        self.assertEqual(call.arguments["relations"], ["has_symptom"])
        self.assertEqual(
            call.arguments["metadata_filter"]["disease_name"], ["百日咳"]
        )
        self.assertIsNone(call.arguments["metadata_filter"]["department"])

    def test_unknown_tool_and_argument_are_rejected(self) -> None:
        with self.assertRaises(ToolCallValidationError):
            validate_tool_call("delete_collection", {})
        arguments = {
            "query": "百日咳病因",
            "top_k": 5,
            "metadata_filter": empty_metadata_filter(disease_name="百日咳"),
            "drop": True,
        }
        with self.assertRaisesRegex(ToolCallValidationError, "Unsupported arguments"):
            validate_tool_call("vector_search", arguments)

    def test_invalid_relation_and_mismatched_disease_are_rejected(self) -> None:
        base = {
            "query": "百日咳有哪些症状",
            "disease_name": "百日咳",
            "top_k": 10,
            "vector_top_k": 6,
            "graph_top_k": 6,
            "metadata_filter": empty_metadata_filter(disease_name="流感"),
            "relations": ["has_symptom"],
            "allow_partial": False,
        }
        with self.assertRaisesRegex(ToolCallValidationError, "must match"):
            validate_tool_call("hybrid_search", base)
        base["metadata_filter"] = empty_metadata_filter(disease_name="百日咳")
        base["relations"] = ["made_up_relation"]
        with self.assertRaisesRegex(ToolCallValidationError, "Unsupported graph relation"):
            validate_tool_call("hybrid_search", base)

    def test_vector_search_requires_exactly_one_disease(self) -> None:
        arguments = {
            "query": "有哪些症状",
            "top_k": 5,
            "metadata_filter": empty_metadata_filter(),
        }
        with self.assertRaisesRegex(ToolCallValidationError, "exactly one"):
            validate_tool_call("vector_search", arguments)


if __name__ == "__main__":
    unittest.main()
