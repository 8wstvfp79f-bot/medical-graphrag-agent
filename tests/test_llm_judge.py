from __future__ import annotations

import asyncio
import json
import unittest

from src.evaluation.judge import (
    JudgeRequest,
    JudgeResult,
    OpenAICompatibleLLMJudge,
)
from src.llm.client import LLMClientError, OpenAICompatibleChatClient


class LLMJudgeTests(unittest.TestCase):
    def test_real_judge_adapter_validates_structured_scores(self) -> None:
        captured_payload = None

        def transport(url, headers, payload, timeout):
            nonlocal captured_payload
            del url, headers, timeout
            captured_payload = payload
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"faithfulness":0.9,"relevance":1,'
                                '"completeness":0.8,"unsupported_claims":[],'
                                '"rationale":"evidence supports the answer"}'
                            )
                        }
                    }
                ]
            }

        request = JudgeRequest(
            case_id="case-1",
            retrieval_mode="hybrid",
            question="百日咳有哪些症状？",
            disease_name="百日咳",
            gold_entities=["咳嗽"],
            answer="百日咳可出现咳嗽[E1]。",
            evidence=[
                {
                    "citation_id": "E1",
                    "content": "百日咳可出现咳嗽",
                    "retrieval_score": 0.9,
                    "provenance": [{"source": "not-needed-by-judge"}],
                }
            ],
            rubric={"faithfulness": "是否有证据"},
        )
        client = OpenAICompatibleChatClient(
            base_url="http://judge.test",
            model="judge-model",
            transport=transport,
        )
        result = asyncio.run(OpenAICompatibleLLMJudge(client).judge(request))

        self.assertEqual(result["faithfulness"], 0.9)
        self.assertEqual(result["unsupported_claims"], [])
        user_message = captured_payload["messages"][1]["content"]
        sent_evidence = json.loads(user_message)["evidence"][0]
        self.assertEqual(sent_evidence["content"], "百日咳可出现咳嗽")
        self.assertNotIn("retrieval_score", sent_evidence)
        self.assertNotIn("provenance", sent_evidence)

    def test_judge_result_rejects_out_of_range_scores(self) -> None:
        with self.assertRaises(LLMClientError):
            JudgeResult.from_mapping(
                {
                    "faithfulness": 1.1,
                    "relevance": 1,
                    "completeness": 1,
                    "unsupported_claims": [],
                }
            )

    def test_judge_retries_once_when_first_structured_score_is_invalid(self) -> None:
        responses = iter(
            [
                (
                    '{"faithfulness":1.2,"relevance":1,"completeness":1,'
                    '"unsupported_claims":[],"rationale":"invalid range"}'
                ),
                (
                    '{"faithfulness":0.9,"relevance":1,"completeness":0.8,'
                    '"unsupported_claims":[],"rationale":"repaired"}'
                ),
            ]
        )
        calls = 0

        def transport(url, headers, payload, timeout):
            nonlocal calls
            del url, headers, payload, timeout
            calls += 1
            return {"choices": [{"message": {"content": next(responses)}}]}

        request = JudgeRequest(
            case_id="case-1",
            retrieval_mode="hybrid",
            question="百日咳有哪些症状？",
            disease_name="百日咳",
            gold_entities=["咳嗽"],
            answer="百日咳可出现咳嗽[E1]。",
            evidence=[{"citation_id": "E1", "content": "百日咳可出现咳嗽"}],
            rubric={"faithfulness": "是否有证据"},
        )
        client = OpenAICompatibleChatClient(
            base_url="http://judge.test",
            model="judge-model",
            transport=transport,
        )

        result = asyncio.run(OpenAICompatibleLLMJudge(client).judge(request))

        self.assertEqual(calls, 2)
        self.assertEqual(result["faithfulness"], 0.9)
        self.assertEqual(result["rationale"], "repaired")


if __name__ == "__main__":
    unittest.main()
