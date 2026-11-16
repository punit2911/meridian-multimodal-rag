"""Anthropic Messages API client (tool use) over plain HTTPS."""
from __future__ import annotations

from typing import Optional

import requests

from ..config import LLMSettings
from .base import LLMClient, LLMError, LLMReply, Message, ToolCall, ToolSpec, with_retries


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, cfg: LLMSettings):
        if not cfg.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.cfg = cfg
        self.model = cfg.effective_model
        self.url = cfg.anthropic_base_url.rstrip("/") + "/v1/messages"

    @staticmethod
    def _wire(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "assistant":
                blocks: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments} for t in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # parallel results must share one user turn
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def chat(self, system: str, messages: list[Message], tools: Optional[list[ToolSpec]] = None, *, max_tokens: int = 1200, temperature: float = 0.0) -> LLMReply:
        body: dict = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages": self._wire(messages)}
        if tools:
            body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        headers = {"x-api-key": self.cfg.anthropic_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        resp = with_retries(lambda: requests.post(self.url, json=body, headers=headers, timeout=self.cfg.timeout_s), self.cfg.max_retries)
        if resp.status_code != 200:
            raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in data.get("content", []) if b.get("type") == "tool_use"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"malformed Anthropic reply: {exc}") from exc
        return LLMReply(text=text.strip(), tool_calls=calls, usage=data.get("usage", {}))
