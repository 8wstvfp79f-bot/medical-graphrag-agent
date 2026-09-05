from __future__ import annotations

import asyncio
import unittest

from src.generation.answer_generator import (
    DISCLAIMER,
    AnswerGenerationError,
    ExtractiveCitationAnswerGenerator,
    GroundedLLMAnswerGenerator,
)
from src.llm.client import OpenAICompatibleChatClient


class AnswerGeneratorTests(unittest.TestCase):
    def test_grounded_llm_generator_keeps_only_valid_citations(self) -> None:
        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"statements":[{"text":"百日咳可出现痉挛性咳嗽。",'
                                '"citation_ids":["E1"]}]}'
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleChatClient(
            base_url="http://local.test/v1",
            model="generation-test",
            transport=transport,
        )
        context = {
            "context": "[E1]\n百日咳可出现痉挛性咳嗽。",
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "content": "百日咳可出现痉挛性咳嗽。",
                    "sources": ["medical.json"],
                }
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "sources": ["medical.json"],
                }
            ],
        }

        result = asyncio.run(
            GroundedLLMAnswerGenerator(client).generate(
                "百日咳有什么症状？", context, intents=["symptom"]
            )
        )

        self.assertEqual(result.mode, "llm_grounded")
        self.assertEqual(result.model, "generation-test")
        self.assertEqual(result.used_evidence_ids, ["ev-1"])
        self.assertIn("[E1]", result.answer)

    def test_grounded_llm_generator_repairs_missing_citations_once(self) -> None:
        responses = iter(
            [
                '{"answer":"百日咳可出现痉挛性咳嗽。","used_citation_ids":[]}',
                (
                    '{"statements":[{"text":"百日咳可出现痉挛性咳嗽。",'
                    '"citation_ids":["E1"]}]}'
                ),
            ]
        )
        calls = 0

        def transport(url, headers, payload, timeout):
            nonlocal calls
            del url, headers, payload, timeout
            calls += 1
            return {"choices": [{"message": {"content": next(responses)}}]}

        client = OpenAICompatibleChatClient(
            base_url="http://local.test/v1",
            model="generation-test",
            transport=transport,
        )
        context = {
            "context": "[E1]\n百日咳可出现痉挛性咳嗽。",
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "content": "百日咳可出现痉挛性咳嗽。",
                    "sources": ["medical.json"],
                }
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "sources": ["medical.json"],
                }
            ],
        }

        result = asyncio.run(
            GroundedLLMAnswerGenerator(client).generate(
                "百日咳有什么症状？", context, intents=["symptom"]
            )
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result.mode, "llm_grounded")
        self.assertIn("[E1]", result.answer)

    def test_grounded_llm_generator_normalizes_local_model_citation_brackets(self) -> None:
        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"statements":[{"text":"相关症状包括咳嗽【e1，E2】。",'
                                '"citation_ids":["[E1]","E2"]}]}'
                            )
                        }
                    }
                ]
            }

        client = OpenAICompatibleChatClient(
            base_url="http://local.test/v1",
            model="generation-test",
            transport=transport,
        )
        evidence = [
            {
                "citation_id": citation_id,
                "evidence_id": f"ev-{index}",
                "content": "症状证据。",
                "sources": ["medical.json"],
            }
            for index, citation_id in enumerate(("E1", "E2"), start=1)
        ]
        context = {
            "context": "[E1]\n症状证据一。\n[E2]\n症状证据二。",
            "evidence": evidence,
            "citations": [
                {
                    "citation_id": item["citation_id"],
                    "evidence_id": item["evidence_id"],
                    "sources": item["sources"],
                }
                for item in evidence
            ],
        }

        result = asyncio.run(
            GroundedLLMAnswerGenerator(client).generate("有哪些症状？", context)
        )

        self.assertIn("[E1][E2]", result.answer)
        self.assertEqual(result.used_evidence_ids, ["ev-1", "ev-2"])

    def test_generates_traceable_answer_from_vector_and_graph_evidence(self) -> None:
        generator = ExtractiveCitationAnswerGenerator(max_evidence_items=2)
        context = {
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-vector",
                    "evidence_type": "vector",
                    "disease_name": "百日咳",
                    "content": "百日咳的典型症状包括阵发性痉挛性咳嗽。",
                    "sources": ["medical.json"],
                },
                {
                    "citation_id": "E2",
                    "evidence_id": "ev-graph",
                    "evidence_type": "graph",
                    "disease_name": "百日咳",
                    "relation": "has_symptom",
                    "entity_name": "痉挛性咳嗽",
                    "content": "百日咳 -[has_symptom]-> 痉挛性咳嗽",
                    "sources": ["triples.csv"],
                },
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-vector",
                    "evidence_type": "vector",
                    "sources": ["medical.json"],
                },
                {
                    "citation_id": "E2",
                    "evidence_id": "ev-graph",
                    "evidence_type": "graph",
                    "sources": ["triples.csv"],
                },
            ],
        }

        result = asyncio.run(generator.generate("百日咳有什么症状？", context))

        self.assertIn("[E1]", result.answer)
        self.assertIn("[E2]", result.answer)
        self.assertIn("痉挛性咳嗽", result.answer)
        self.assertIn(DISCLAIMER, result.answer)
        self.assertEqual(result.sources, ["medical.json", "triples.csv"])
        self.assertEqual(result.used_evidence_ids, ["ev-vector", "ev-graph"])

    def test_no_evidence_returns_honest_fallback(self) -> None:
        result = asyncio.run(
            ExtractiveCitationAnswerGenerator().generate(
                "百日咳有什么症状？", {"evidence": [], "citations": []}
            )
        )

        self.assertIn("没有找到足够证据", result.answer)
        self.assertEqual(result.citations, [])

    def test_graph_evidence_is_not_lost_when_vectors_fill_answer_limit(self) -> None:
        evidence = [
            {
                "citation_id": f"E{index}",
                "evidence_id": f"ev-{index}",
                "evidence_type": "vector",
                "disease_name": "百日咳",
                "content": "疾病介绍文本。",
                "sources": ["medical.json"],
            }
            for index in range(1, 4)
        ]
        evidence.append(
            {
                "citation_id": "E4",
                "evidence_id": "ev-4",
                "evidence_type": "graph",
                "disease_name": "百日咳",
                "relation": "has_symptom",
                "entity_name": "鸡鸣样吸气吼声",
                "content": "百日咳 -[has_symptom]-> 鸡鸣样吸气吼声",
                "sources": ["triples.csv"],
            }
        )
        citations = [
            {
                "citation_id": item["citation_id"],
                "evidence_id": item["evidence_id"],
                "evidence_type": item["evidence_type"],
                "sources": item["sources"],
            }
            for item in evidence
        ]

        result = asyncio.run(
            ExtractiveCitationAnswerGenerator(max_evidence_items=3).generate(
                "百日咳有什么症状？",
                {"evidence": evidence, "citations": citations},
            )
        )

        self.assertIn("鸡鸣样吸气吼声", result.answer)
        self.assertIn("[E4]", result.answer)
        self.assertNotIn("[E3]", result.answer)

    def test_structured_symptom_field_is_extracted_from_long_chunk(self) -> None:
        context = {
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "evidence_type": "vector",
                    "disease_name": "百日咳",
                    "content": (
                        "预防：保持通风。症状：痉挛性咳嗽、低热、鸡鸣样吸气声 "
                        "检查：血常规"
                    ),
                    "sources": ["medical.json"],
                }
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-1",
                    "evidence_type": "vector",
                    "sources": ["medical.json"],
                }
            ],
        }

        result = asyncio.run(
            ExtractiveCitationAnswerGenerator().generate(
                "百日咳有哪些症状？", context
            )
        )

        self.assertIn("症状：痉挛性咳嗽、低热、鸡鸣样吸气声", result.answer)
        self.assertNotIn("保持通风", result.answer)

    def test_structured_field_filters_rejected_items_and_duplicates(self) -> None:
        context = {
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-vector-cleanup",
                    "evidence_type": "vector",
                    "disease_name": "百日咳",
                    "content": (
                        "症状：痉挛性咳嗽、闫鹏辉、痉挛性咳嗽、低热 "
                        "检查：血常规"
                    ),
                    "sources": ["medical.json"],
                },
                {
                    "citation_id": "E2",
                    "evidence_id": "ev-graph-rejected",
                    "evidence_type": "graph",
                    "disease_name": "百日咳",
                    "relation": "has_symptom",
                    "entity_name": "闫鹏辉",
                    "content": "百日咳 -[has_symptom]-> 闫鹏辉",
                    "sources": ["triples.csv"],
                },
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-vector-cleanup",
                    "evidence_type": "vector",
                    "sources": ["medical.json"],
                },
                {
                    "citation_id": "E2",
                    "evidence_id": "ev-graph-rejected",
                    "evidence_type": "graph",
                    "sources": ["triples.csv"],
                },
            ],
        }

        result = asyncio.run(
            ExtractiveCitationAnswerGenerator(max_evidence_items=2).generate(
                "百日咳相关信息",
                context,
                intents=["symptom"],
            )
        )

        self.assertNotIn("闫鹏辉", result.answer)
        self.assertEqual(result.answer.count("痉挛性咳嗽"), 1)
        self.assertIn("低热", result.answer)
        self.assertIn("[E1]", result.answer)
        self.assertNotIn("[E2]", result.answer)
        self.assertEqual(result.used_evidence_ids, ["ev-vector-cleanup"])

    def test_unstructured_fallback_scrubs_confirmed_rejected_term(self) -> None:
        context = {
            "evidence": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-unstructured",
                    "evidence_type": "vector",
                    "disease_name": "百日咳",
                    "content": "典型表现包括痉挛性咳嗽、闫鹏辉、低热。",
                    "sources": ["stale-collection"],
                }
            ],
            "citations": [
                {
                    "citation_id": "E1",
                    "evidence_id": "ev-unstructured",
                    "evidence_type": "vector",
                    "sources": ["stale-collection"],
                }
            ],
        }

        result = asyncio.run(
            ExtractiveCitationAnswerGenerator().generate(
                "百日咳相关信息",
                context,
                intents=["symptom"],
            )
        )

        self.assertIn("痉挛性咳嗽", result.answer)
        self.assertIn("低热", result.answer)
        self.assertNotIn("闫鹏辉", result.answer)

    def test_blank_query_and_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(AnswerGenerationError):
            ExtractiveCitationAnswerGenerator(max_evidence_items=0)
        with self.assertRaises(AnswerGenerationError):
            asyncio.run(
                ExtractiveCitationAnswerGenerator().generate(
                    "  ", {"evidence": [], "citations": []}
                )
            )


if __name__ == "__main__":
    unittest.main()
