"""OpenAI-compatible chat-completions client (OpenAI, Azure-style gateways, vLLM, Ollama, LM Studio...)."""
from __future__ import annotations

import json
from typing import Optional

import requests

from ..config import LLMSettings
from .base import LLMClient, LLMError, LLMReply, Message, ToolCall, ToolSpec, with_retries


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, cfg: LLMSettings):
        base = cfg.openai_base_url.rstrip("/")
        local = any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not cfg.openai_api_key and not local:
            raise LLMError("OPENAI_API_KEY is not set")
        self.cfg = cfg
        self.model = cfg.effective_model
        self.url = base + "/chat/completions"

    @staticmethod
    def _wire(system: str, messages: list[Message]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "assistant":
                d: dict = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": json.dumps(t.arguments)}} for t in m.tool_calls]
                out.append(d)
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    def chat(self, system: str, messages: list[Message], tools: Optional[list[ToolSpec]] = None, *, max_tokens: int = 1200, temperature: float = 0.0) -> LLMReply:
        body: dict = {"model": self.model, "messages": self._wire(system, messages), "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools]
        headers = {"content-type": "application/json"}
        if self.cfg.openai_api_key:
            headers["authorization"] = f"Bearer {self.cfg.openai_api_key}"
        resp = with_retries(lambda: requests.post(self.url, json=body, headers=headers, timeout=self.cfg.timeout_s), self.cfg.max_retries)
        if resp.status_code != 200:
            raise LLMError(f"OpenAI-compatible API error {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            msg = data["choices"][0]["message"]
            calls = []
            for t in msg.get("tool_calls") or []:
                try:
                    args = json.loads(t["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append(ToolCall(t["id"], t["function"]["name"], args if isinstance(args, dict) else {}))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed chat-completions reply: {exc}") from exc
        return LLMReply(text=(msg.get("content") or "").strip(), tool_calls=calls, usage=data.get("usage", {}))
