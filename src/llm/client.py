from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence


class LLMClientError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint returns an invalid response."""


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, object], float], Mapping[str, object]
]


def _chat_completions_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("base_url must not be empty")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions" if value.endswith("/v1") else f"{value}/v1/chat/completions"


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    timeout: float,
) -> Mapping[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMClientError(
            f"LLM endpoint returned HTTP {exc.code}: {body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMClientError(f"Could not reach LLM endpoint: {exc.reason}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMClientError("LLM endpoint did not return valid JSON") from exc
    if not isinstance(value, Mapping):
        raise LLMClientError("LLM endpoint response must be a JSON object")
    return value


def _message_content(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise LLMClientError("LLM response is missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LLMClientError("LLM response choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise LLMClientError("LLM response is missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(str(block["text"]))
        if parts:
            return "".join(parts).strip()
    raise LLMClientError("LLM response message has no text content")


def parse_json_object(value: str) -> dict[str, object]:
    text = value.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise LLMClientError("LLM response does not contain a JSON object")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMClientError("Could not parse JSON object from LLM response") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON response must be an object")
    return parsed


def _requires_text_response_format(error: Exception) -> bool:
    """Detect LM Studio servers that reject OpenAI's json_object mode."""

    message = str(error).lower()
    return (
        "response_format.type" in message
        and "json_schema" in message
        and "text" in message
    )


class OpenAICompatibleChatClient:
    """Minimal async client for local or hosted OpenAI-compatible endpoints.

    The API key is only placed in the Authorization header and is never included
    in returned errors, logs, or saved evaluation artifacts.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("timeout_seconds must be positive and max_retries non-negative")
        self.url = _chat_completions_url(base_url)
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.transport = transport or _default_transport

    def _complete_sync(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: Mapping[str, object] | None = None,
        schema_name: str = "structured_response",
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": dict(json_schema),
                    },
                }
                if json_schema is not None
                else {"type": "json_object"}
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        attempt = 0
        used_text_json_fallback = False
        while True:
            try:
                response = self.transport(
                    self.url, headers, payload, self.timeout_seconds
                )
                return _message_content(response)
            except Exception as exc:
                if (
                    json_mode
                    and not used_text_json_fallback
                    and _requires_text_response_format(exc)
                ):
                    # LM Studio 0.4 accepts json_schema or text, but not the
                    # older json_object value.  The prompts already require a
                    # JSON-only answer and complete_json validates/parses it.
                    payload["response_format"] = {"type": "text"}
                    used_text_json_fallback = True
                    continue
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))
                attempt += 1
        if isinstance(last_error, LLMClientError):
            raise last_error
        raise LLMClientError(f"LLM request failed: {type(last_error).__name__}") from last_error

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            json_schema=None,
        )

    async def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        schema: Mapping[str, object] | None = None,
        schema_name: str = "structured_response",
    ) -> dict[str, object]:
        text = await asyncio.to_thread(
            self._complete_sync,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            json_schema=schema,
            schema_name=schema_name,
        )
        return parse_json_object(text)
