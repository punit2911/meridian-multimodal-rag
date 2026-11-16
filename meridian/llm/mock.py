"""Scripted in-process LLM used by the tests and the offline demo of the tool-calling path (no network)."""
from __future__ import annotations

import re
from typing import Callable, Optional

from .base import LLMClient, LLMReply, Message, ToolCall, ToolSpec


class MockLLM(LLMClient):
    """First round: calls search_documents once. Then answers with the first evidence sentence that shares the most words with the question,
    citing its passage number. `responder(question, evidence) -> str` can override the answer (to simulate hallucinations)."""

    name = "mock"
    model = "scripted"

    def __init__(self, responder: Optional[Callable[[str, str], str]] = None, call_tool: bool = True, fail: bool = False):
        self.responder, self.call_tool, self.fail = responder, call_tool, fail
        self.calls = 0

    def chat(self, system, messages, tools: Optional[list[ToolSpec]] = None, *, max_tokens: int = 1200, temperature: float = 0.0) -> LLMReply:
        from .base import LLMError
        self.calls += 1
        if self.fail:
            raise LLMError("scripted failure")
        first = messages[0].content
        question = re.search(r"Question: (.*)", first).group(1)
        if self.call_tool and tools and not any(m.role == "tool" for m in messages):
            return LLMReply(tool_calls=[ToolCall("t1", "search_documents", {"query": question})])
        evidence = "\n".join(m.content for m in messages if m.role in ("user", "tool"))
        if self.responder:
            return LLMReply(text=self.responder(question, evidence))
        qw = set(re.findall(r"[a-z0-9\-]+", question.lower()))
        best, best_s = "", 0
        for m in re.finditer(r"(\[\d+\])[^\n]*\n?([^\n]+)", evidence):
            sent_words = set(re.findall(r"[a-z0-9\-]+", m.group(2).lower()))
            sc = len(qw & sent_words)
            if sc > best_s:
                best, best_s = f"{m.group(2).strip()} {m.group(1)}", sc
        return LLMReply(text=best or "INSUFFICIENT_EVIDENCE")
