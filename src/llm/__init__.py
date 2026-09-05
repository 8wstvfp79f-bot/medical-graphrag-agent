"""Small provider-neutral helpers for OpenAI-compatible chat endpoints."""

from src.llm.client import LLMClientError, OpenAICompatibleChatClient

__all__ = ["LLMClientError", "OpenAICompatibleChatClient"]
