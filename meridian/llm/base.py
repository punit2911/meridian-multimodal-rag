"""Provider-neutral chat interface with tool calling. Implementations only translate wire formats; all agent logic lives in agent/llm_agent.py."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


class LLMError(RuntimeError):
    """Any failure talking to a model provider (network, auth, malformed reply). The agent falls back to the offline composer."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # for assistant messages
    tool_call_id: str = ""  # for tool messages
    tool_name: str = ""


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLMClient:
    name = "base"
    model = ""

    def chat(self, system: str, messages: list[Message], tools: Optional[list[ToolSpec]] = None, *, max_tokens: int = 1200,
             temperature: float = 0.0) -> LLMReply:
        raise NotImplementedError


def with_retries(fn, retries: int, retry_statuses=(408, 409, 429, 500, 502, 503, 504, 529)) -> Any:
    """Exponential backoff on transient HTTP failures. `fn` returns a requests.Response."""
    import requests

    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            resp = fn()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
        else:
            if resp.status_code not in retry_statuses:
                return resp
            last = LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if attempt < retries - 1:
            time.sleep(min(8.0, 0.8 * (2 ** attempt)))
    raise LLMError(f"request failed after {retries} attempt(s): {last}")
