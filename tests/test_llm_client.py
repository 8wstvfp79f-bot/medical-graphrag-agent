from __future__ import annotations

import asyncio
import unittest

from src.llm.client import (
    LLMClientError,
    OpenAICompatibleChatClient,
    parse_json_object,
)


class LLMClientTests(unittest.TestCase):
    def test_client_sends_model_and_parses_message(self) -> None:
        captured: dict[str, object] = {}

        def transport(url, headers, payload, timeout):
            captured.update(
                {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
            )
            return {"choices": [{"message": {"content": '{"score": 1}'}}]}

        client = OpenAICompatibleChatClient(
            base_url="http://localhost:1234/v1",
            model="local-model",
            api_key="secret",
            transport=transport,
        )
        result = asyncio.run(
            client.complete_json([{"role": "user", "content": "judge"}])
        )

        self.assertEqual(result, {"score": 1})
        self.assertEqual(captured["url"], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "local-model")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")

    def test_json_parser_accepts_fenced_output(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_lm_studio_json_object_error_falls_back_to_text_mode(self) -> None:
        response_formats: list[object] = []

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            response_formats.append(payload.get("response_format"))
            if payload.get("response_format") == {"type": "json_object"}:
                raise LLMClientError(
                    "LLM endpoint returned HTTP 400: "
                    "'response_format.type' must be 'json_schema' or 'text'"
                )
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        client = OpenAICompatibleChatClient(
            base_url="http://localhost:1234",
            model="local-model",
            max_retries=0,
            transport=transport,
        )

        result = asyncio.run(
            client.complete_json([{"role": "user", "content": "judge"}])
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            response_formats,
            [{"type": "json_object"}, {"type": "text"}],
        )

    def test_json_schema_is_sent_as_structured_output(self) -> None:
        captured: dict[str, object] = {}

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            captured.update(payload)
            return {"choices": [{"message": {"content": '{"answer":"ok"}'}}]}

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        client = OpenAICompatibleChatClient(
            base_url="http://localhost:1234",
            model="local-model",
            transport=transport,
        )

        result = asyncio.run(
            client.complete_json(
                [{"role": "user", "content": "answer"}],
                schema=schema,
                schema_name="answer_response",
            )
        )

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(
            captured["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
