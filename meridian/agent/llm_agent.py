"""LLM tool-calling agent.

The deterministic pipeline (analysis → hybrid retrieval → CRAG grading → evidence assembly → structured tools) still runs first and decides
*whether* an answer is possible. The LLM then writes the answer from that numbered evidence and may call two tools to look further
(`search_documents`, `lookup_spec`). Whatever it writes passes through the same claim-level verifier as the offline composer;
if the model is unreachable, returns nothing or fails verification, the offline composer's answer is used instead.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

from ..config import Settings
from ..kb import KnowledgeBase
from ..llm.base import LLMClient, LLMError, Message, ToolSpec
from ..retrieval.analyzer import analyze
from ..retrieval.pipeline import HybridRetriever
from ..utils.text import truncate
from .agent import Orchestrator
from .compose import Composed, ComposeContext, compose
from .tools import lookup_spec

MAX_TOOL_ROUNDS = 3

SYSTEM = """You are Meridian, a maintenance-documentation assistant for industrial equipment.
Answer ONLY from the numbered evidence passages supplied by the user turn or returned by your tools.
Rules:
- Cite every factual sentence with the passage number in square brackets, e.g. [2]. Never cite a number you were not given.
- Copy numbers, units, part numbers, alarm codes and dates exactly as written in the evidence.
- If two passages disagree, prefer the one marked as a bulletin / later effective date and say that it supersedes the older value.
- If the user's premise is wrong (for example a wrong torque value), say so and give the documented value.
- Include any WARNING or lock-out notice that applies to a procedure.
- If the evidence does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
- Be concise: at most 6 sentences or a short numbered list for procedures. No preamble."""

TOOLS = [
    ToolSpec("search_documents", "Search the maintenance knowledge base. Returns numbered passages. Use when the evidence is incomplete.",
             {"type": "object", "properties": {"query": {"type": "string"}, "model": {"type": "string", "description": "optional product model such as HX-200"}},
              "required": ["query"]}),
    ToolSpec("lookup_spec", "Look up a specification value (torque, pressure, temperature, flow...) for a model with conflict resolution between manual and bulletins.",
             {"type": "object", "properties": {"question": {"type": "string"}, "model": {"type": "string"}}, "required": ["question"]}),
]


class LLMOrchestrator(Orchestrator):
    mode = "llm_tools"

    def __init__(self, kb: KnowledgeBase, settings: Settings, client: LLMClient, retriever: Optional[HybridRetriever] = None):
        super().__init__(kb, settings, retriever)
        self.client = client

    # ------------------------------------------------------------------------------------------ evidence block
    def _evidence_text(self, ctx: ComposeContext) -> str:
        kb, book = self.kb, ctx.book
        lines: list[str] = []
        for it in ctx.evidence:
            tag = book.cite(it.chunk, via=it.via, score=it.score)
            d = kb.doc(it.chunk.doc_id)
            meta = f"{d.title} | {d.doc_type} | effective {d.effective_date or 'n/a'} | page {it.chunk.page}" if d else ""
            lines.append(f"{tag} ({meta})\n{truncate(it.chunk.text, 900)}")
        extra: list[str] = []
        for f in ctx.spec.selected:
            extra.append(f"- Spec fact: {f.subject} / {f.attribute} = {f.display} ({f.model}; {f.doc_title}; effective {f.effective_date or 'n/a'}) {book.cite(kb.chunk(f.chunk_id))}")
        for c in ctx.spec.conflicts:
            extra.append(f"- Conflict resolved: {c.winner} says {c.winner_value}; it supersedes {c.loser} ({c.loser_value}). Rule: {c.rule}.")
        if ctx.premise:
            extra.append(f"- Premise check: {ctx.premise.verdict}")
        for r in ctx.chart_reads:
            extra.append(f"- Chart read (digitised, approximate): at {r.x_label} the value is {r.display()} {book.cite(r.chunk)}")
        if ctx.analytics and ctx.analytics.ok:
            extra.append(f"- Computed over the log ({ctx.analytics.code}): {ctx.analytics.text}")
        return "\n\n".join(lines) + ("\n\nStructured findings:\n" + "\n".join(extra) if extra else "")

    # ------------------------------------------------------------------------------------------ tools
    def _run_tool(self, name: str, args: dict, ctx: ComposeContext) -> str:
        kb, book = self.kb, ctx.book
        if name == "search_documents":
            q = str(args.get("query", "")).strip()
            if not q:
                return "Error: empty query."
            if args.get("model"):
                q = f"{args['model']} {q}"
            a = analyze(q, set(kb.known_models()))
            res = self.retriever.retrieve(a)
            if not res.candidates:
                return "No passages found."
            out = []
            for c in res.candidates[:4]:
                out.append(f"{book.cite(c.chunk, via='tool', score=c.final)} {truncate(c.chunk.text, 600)}")
            return "\n\n".join(out)
        if name == "lookup_spec":
            q = str(args.get("question", "")).strip()
            if args.get("model"):
                q = f"{args['model']} {q}"
            a = analyze(q, set(kb.known_models()))
            sp = lookup_spec(kb, a)
            if not sp.selected:
                return "No matching specification found."
            lines = [f"{f.subject} / {f.attribute} = {f.display} ({f.model}; {f.doc_title}; {f.effective_date or 'n/a'}) {book.cite(kb.chunk(f.chunk_id))}" for f in sp.selected]
            lines += [f"Conflict: {c.winner} ({c.winner_value}) supersedes {c.loser} ({c.loser_value})" for c in sp.conflicts]
            return "\n".join(lines)
        return f"Error: unknown tool {name}."

    # ------------------------------------------------------------------------------------------ compose hook
    def _compose(self, ctx: ComposeContext, question: str, emit: Callable) -> Composed:
        ctx.book.reset()  # the LLM prompt gets its own numbering; the offline composer re-cites on fallback
        t = time.perf_counter()
        try:
            user = f"Question: {question}\n\nEvidence passages:\n\n{self._evidence_text(ctx)}"
            msgs = [Message("user", user)]
            text = ""
            for rnd in range(MAX_TOOL_ROUNDS + 1):
                reply = self.client.chat(SYSTEM, msgs, TOOLS if rnd < MAX_TOOL_ROUNDS else None, max_tokens=self.s.llm.max_tokens)
                if not reply.tool_calls:
                    text = reply.text
                    break
                msgs.append(Message("assistant", reply.text, reply.tool_calls))
                for tc in reply.tool_calls:
                    result = self._run_tool(tc.name, tc.arguments, ctx)
                    emit("tool", f"LLM called {tc.name}", truncate(json.dumps(tc.arguments), 160), {"result": truncate(result, 400)})
                    msgs.append(Message("tool", result, tool_call_id=tc.id, tool_name=tc.name))
            ms = (time.perf_counter() - t) * 1000
            if text.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
                emit("compose", "LLM judged the evidence insufficient", "", {}, ms, "warn")
                ctx.book.reset()
                return Composed()
            if text.strip():
                emit("compose", f"LLM ({self.client.name}/{self.client.model}) wrote a cited answer", f"{len(ctx.book.order)} evidence item(s)", {"kind": "llm"}, ms)
                return Composed(text=text.strip(), kind="llm")
        except LLMError as exc:
            emit("compose", "LLM unavailable, used the offline composer", str(exc)[:200], {}, (time.perf_counter() - t) * 1000, "warn")
        ctx.book.reset()
        return compose(ctx)
