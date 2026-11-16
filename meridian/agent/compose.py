"""Deterministic, evidence-extractive answer composer + citation book.

Every sentence is assembled from a retrieved row / step list / chart reading / computed result and carries a citation, so the
offline mode cannot invent content. The composer is also the *fallback* of the LLM mode when an LLM answer fails verification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..kb import KnowledgeBase
from ..retrieval.analyzer import Intent, QueryAnalysis
from ..retrieval.context import EvidenceItem
from ..schemas import Chunk, Citation, ConflictNote, Fact, Modality
from ..utils import units as U
from ..utils.text import content_tokens, split_sentences, truncate
from ..verify.verifier import EvidenceDoc
from .tools import AnalyticsResult, ChartRead, PremiseResult, SpecResult, convert_fact, requested_unit

# Fixed vocabulary of the composer's own sentence templates. The verifier does not demand evidence for these connective words
# (numbers, identifiers and the content words of a claim are still checked).
_TEMPLATE_PHRASES = (
    "Note lists but supersedes it with Applicability Safety Procedure from section Referenced value Probable cause Remedy means alarm "
    "Yes No that value is outdated now not Your question does not name a product so here is the answer for each one The for on of "
    "According to Closest material Labels visible in the figure page At the read about Top values Computed There are matching records "
    "in The most frequent with tied average total over is by"
)
from ..utils.text import content_tokens as _ct  # noqa: E402

TEMPLATE_TERMS = frozenset(_ct(_TEMPLATE_PHRASES))

_STEP_LINE = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*)$")


# ====================================================================================================== citation book
class CiteBook:
    """Assigns [n] in order of first use and remembers the verifier text for each evidence item."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.order: list[str] = []
        self.chunks: dict[str, Chunk] = {}
        self.via: dict[str, str] = {}
        self.score: dict[str, float] = {}
        self.extra: dict[str, list[str]] = {}

    def reset(self) -> None:
        self.order.clear(); self.chunks.clear(); self.via.clear(); self.score.clear(); self.extra.clear()

    def cite(self, chunk: Chunk, via: str = "retrieval", extra: str = "", score: float = 0.0) -> str:
        if chunk.id not in self.chunks:
            self.order.append(chunk.id)
            self.chunks[chunk.id] = chunk
            self.via[chunk.id] = via
            self.score[chunk.id] = score
        if extra:
            self.extra.setdefault(chunk.id, []).append(extra)
        return f"[{self.index_of(chunk.id)}]"

    def index_of(self, chunk_id: str) -> int:
        return self.order.index(chunk_id) + 1

    def evidence_docs(self) -> list[EvidenceDoc]:
        docs = []
        for i, cid in enumerate(self.order, 1):
            ch = self.chunks[cid]
            d = self.kb.doc(ch.doc_id)
            prov = (f"Source: {d.title}; models {', '.join(d.models) or 'n/a'}; effective {d.effective_date or 'n/a'}; revision {d.revision or 'n/a'}; page {ch.page}."
                    if d else "")
            text = ch.text + ("\n" + prov if prov else "") + ("\n" + "\n".join(self.extra[cid]) if cid in self.extra else "")
            docs.append(EvidenceDoc(index=i, text=text, doc_id=ch.doc_id))
        return docs

    def citations(self) -> list[Citation]:
        out = []
        for i, cid in enumerate(self.order, 1):
            ch = self.chunks[cid]
            d = self.kb.doc(ch.doc_id)
            out.append(Citation(
                index=i, chunk_id=ch.id, doc_id=ch.doc_id, doc_title=d.title if d else ch.doc_id, page=ch.page, modality=ch.modality.value,
                section=ch.section, snippet=truncate(ch.content, 260), bbox=ch.bbox.as_list() if ch.bbox else None,
                page_size=[ch.bbox.page_width, ch.bbox.page_height] if ch.bbox else None, via=self.via.get(cid, "retrieval"),
                score=round(self.score.get(cid, 0.0), 4), has_image=bool(ch.image_path)))
        return out


@dataclass
class ComposeContext:
    kb: KnowledgeBase
    analysis: QueryAnalysis
    evidence: list[EvidenceItem]
    spec: SpecResult
    premise: Optional[PremiseResult]
    chart_reads: list[ChartRead]
    analytics: Optional[AnalyticsResult]
    book: CiteBook
    min_support: float = 0.5


@dataclass
class Composed:
    text: str = ""
    kind: str = ""  # analytics | chart | premise | spec | troubleshoot | procedure | figure | extract | none
    notes: list[str] = field(default_factory=list)


# ====================================================================================================== helpers
def _chunk_by_id(ctx: ComposeContext, cid: str) -> Optional[Chunk]:
    for it in ctx.evidence:
        if it.chunk.id == cid:
            return it.chunk
    return ctx.kb.chunk(cid)


def _via(ctx: ComposeContext, cid: str) -> str:
    for it in ctx.evidence:
        if it.chunk.id == cid:
            return it.via
    return "tool"


def _doc_title(ctx: ComposeContext, doc_id: str) -> str:
    d = ctx.kb.doc(doc_id)
    return d.title if d else doc_id


def _cells(ch: Chunk) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in ch.meta.get("cells", [])}


def _fact_sentence(ctx: ComposeContext, f: Fact, ch: Chunk) -> str:
    model = f"the {f.model}" if f.model else "this product"
    subj = f.subject.lower()
    if f.attribute in ("value", ""):
        s = f"The {subj} of {model} is {f.display}"
    else:
        s = f"The {f.attribute} for {subj} on {model} is {f.display}"
    conv = convert_fact(f, requested_unit(ctx.analysis.question))
    if conv:
        s += f" ({conv})"
    extras = [f"{k}: {v}" for k, v in ch.meta.get("cells", []) if not re.fullmatch(r"[\d.,]+\s*\S*", str(v)) and str(k).lower() not in ("value",)]
    if extras and f.attribute not in ("value", ""):
        s += " (" + "; ".join(extras[:2]) + ")"
    return s


def _conflict_lines(ctx: ComposeContext, notes: list[ConflictNote]) -> list[str]:
    lines = []
    for n in notes[:2]:
        win = _chunk_by_id(ctx, n.winner_chunk)
        lose = _chunk_by_id(ctx, n.loser_chunk)
        if not win or not lose:
            continue
        wd, ld = ctx.kb.doc(win.doc_id), ctx.kb.doc(lose.doc_id)
        wc = ctx.book.cite(win, via="conflict")
        lc = ctx.book.cite(lose, via="conflict")
        date = f" ({wd.effective_date})" if wd and wd.effective_date else ""
        line = (f"Note: {ld.title if ld else n.loser} lists {n.loser_value} {lc}, but {wd.title if wd else n.winner}{date} "
                f"supersedes it with {n.winner_value} {wc}.")
        applies = _find_applies(ctx, win.doc_id)
        if applies is not None:
            ac, text = applies
            line += f" Applicability: {text} {ac}"
        lines.append(line)
    return lines


def _find_applies(ctx: ComposeContext, doc_id: str) -> Optional[tuple[str, str]]:
    for ch in ctx.kb.doc_chunks(doc_id):
        m = re.search(r"Applies to:\s*(.+)", ch.content)
        if m:
            cite = ctx.book.cite(ch, via="conflict")
            return cite, m.group(1).strip().rstrip(".") + "."
    return None


# ====================================================================================================== branches
def _compose_analytics(ctx: ComposeContext) -> Composed:
    r = ctx.analytics
    assert r is not None and r.ok
    schema = next((c for c in ctx.kb.chunks if c.meta.get("dataset") and c.meta.get("kind") == "schema" and _doc_title(ctx, c.doc_id) == r.dataset), None)
    if schema is None:
        return Composed()
    cite = ctx.book.cite(schema, via="calculator", extra=f"Computed result: {r.text} Method: {r.code}. Rows used: {r.rows_used} of {r.rows_total}.")
    lines = [f"{r.text} {cite}"]
    if r.op == "argmax" and r.table:
        top = ", ".join(f"{list(t.values())[0]} ({t['count']})" for t in r.table[:5])
        ctx.book.cite(schema, via="calculator", extra=f"Top values: {top}.")
        lines.append(f"Top values: {top} {cite}")
    return Composed("\n".join(lines), "analytics", [f"Computed with: {r.code}"])


def _compose_chart(ctx: ComposeContext) -> Composed:
    lines = []
    for cr in ctx.chart_reads[:2]:
        ch = cr.chunk
        chart = ch.meta.get("chart", {})
        xt = chart.get("x_title", "")
        m = re.search(r"\(([^)]+)\)", xt)
        xunit = f" {m.group(1)}" if m else ""
        how = "interpolated between digitised points" if cr.interpolated else "read at a digitised point"
        sent = (f"At {cr.x_label}{xunit}, the {cr.y_title.lower() or 'value'} read from Figure {cr.figure_no} is about {cr.display()}")
        extra = f"Chart reading: at {cr.x_label}{xunit} the {cr.y_title.lower()} is {cr.y:.1f} {cr.y_unit}. Figure {cr.figure_no}."
        cite = ctx.book.cite(ch, via="chart_reader", extra=extra, score=cr.reliability)
        lines.append(f"{sent} {cite}")
        ctx_note = f"Chart values are digitised from the image ({how}); treat them as approximate (confidence {cr.reliability:.0%})."
    return Composed("\n".join(lines), "chart", [ctx_note] if lines else [])


def _compose_premise(ctx: ComposeContext) -> Composed:
    p = ctx.premise
    assert p is not None and p.fact is not None and p.claimed is not None
    f = p.fact
    ch = _chunk_by_id(ctx, f.chunk_id)
    if ch is None:
        return Composed()
    cite = ctx.book.cite(ch, via=_via(ctx, ch.id) if _via(ctx, ch.id) != "tool" else "retrieval")
    claimed = U.format_quantity(p.claimed.value, p.claimed.unit)
    subj = f.subject.lower()
    model = f" of the {f.model}" if f.model else ""
    what = f"{'the ' + f.attribute + ' for ' if f.attribute not in ('value', '') else 'the '}{subj}{model}"
    if p.verdict == "confirmed":
        lines = [f"Yes: {what} is {f.display} {cite}."]
    elif p.verdict == "superseded":
        lines = [f"No, that value is outdated: {what} is now {f.display} {cite}, not {claimed}."]
    else:
        lines = [f"No: {what} is {f.display} {cite}, not {claimed}."]
    lines += _conflict_lines(ctx, ctx.spec.conflicts)
    return Composed("\n".join(lines), "premise")


def _compose_spec(ctx: ComposeContext) -> Composed:
    a = ctx.analysis
    sel = ctx.spec.selected
    if not sel:
        return Composed()
    lines: list[str] = []
    multi_model = len({f.model for f in sel}) > 1
    if a.needs_model and multi_model:
        lines.append("Your question does not name a product, so here is the answer for each one:")
    for f in sel:
        ch = _chunk_by_id(ctx, f.chunk_id)
        if ch is None:
            continue
        cite = ctx.book.cite(ch, via=_via(ctx, ch.id) if _via(ctx, ch.id) != "tool" else "retrieval", score=0.0)
        s = _fact_sentence(ctx, f, ch)
        lines.append(f"{'- ' if multi_model else ''}{s} {cite}.")
    lines += _conflict_lines(ctx, ctx.spec.conflicts)
    return Composed("\n".join(lines), "spec")


def _alarm_rows(ctx: ComposeContext) -> list[tuple[Chunk, dict[str, str], float]]:
    a = ctx.analysis
    qt = set(a.tokens)
    rows = []
    for it in ctx.evidence:
        ch = it.chunk
        if ch.modality != Modality.TABLE_ROW:
            continue
        cells = _cells(ch)
        if not ({"probable cause", "remedy"} & set(cells)):
            continue
        code = str(ch.meta.get("subject", "")).upper()
        if a.codes and code not in a.codes:
            continue
        text_toks = set(content_tokens(" ".join(cells.values())))
        overlap = len(qt & text_toks) / max(len(qt), 1)
        rows.append((ch, cells, overlap + it.score * 0.01))
    if a.codes:
        return sorted(rows, key=lambda r: -r[2])
    rows = [r for r in rows if r[2] >= 0.12]
    return sorted(rows, key=lambda r: -r[2])[:2]


def _compose_troubleshoot(ctx: ComposeContext) -> Composed:
    rows = _alarm_rows(ctx)
    if not rows:
        return Composed()
    lines = []
    if len(rows) > 1 and len({r[0].doc_id for r in rows}) > 1:
        lines.append("The same alarm code appears in several manuals. Check which product you have:")
    for ch, cells, _ in rows[:3]:
        cite = ctx.book.cite(ch, via=_via(ctx, ch.id) if _via(ctx, ch.id) != "tool" else "retrieval")
        d = ctx.kb.doc(ch.doc_id)
        model = d.models[0] if d and d.models else ""
        code = ch.meta.get("subject", "")
        desc = cells.get("description", "").strip()
        cause = cells.get("probable cause", "").strip()
        remedy = cells.get("remedy", "").strip()
        s = f"{model + ': ' if model else ''}alarm {code} means {desc.lower()}."
        if cause:
            s += f" Probable cause: {cause}."
        if remedy:
            s += f" Remedy: {remedy}."
        lines.append(f"{'- ' if len(rows) > 1 else ''}{s} {cite}")
    for it in ctx.evidence:
        if it.chunk.modality == Modality.WARNING and it.via == "warning":
            lines.append(f"Safety: {it.chunk.content} {ctx.book.cite(it.chunk, via='warning')}")
            break
    return Composed("\n".join(lines), "troubleshoot")


def _compose_procedure(ctx: ComposeContext) -> Composed:
    steps = [it for it in ctx.evidence if it.chunk.modality == Modality.TEXT and it.via == "retrieval" and
             (it.chunk.meta.get("block") == "steps" or any(_STEP_LINE.match(l) for l in it.chunk.content.split("\n")))]
    if not steps:
        return Composed()
    primary = steps[0].chunk
    group = [it.chunk for it in steps if it.chunk.doc_id == primary.doc_id and it.chunk.section_no == primary.section_no]
    group.sort(key=lambda c: ctx.kb.ix.by_id.get(c.id, 0))
    lines: list[str] = []
    warns = [it for it in ctx.evidence if it.chunk.modality == Modality.WARNING and it.chunk.doc_id == primary.doc_id]
    warns.sort(key=lambda it: (0 if it.chunk.section_no == primary.section_no else 1, 0 if re.search(r"lock\s*out", it.chunk.content, re.I) else 1))
    for w in warns[:3]:
        lines.append(f"Safety: {w.chunk.content} {ctx.book.cite(w.chunk, via='warning')}")
    d = ctx.kb.doc(primary.doc_id)
    head = f"Procedure from {d.title if d else primary.doc_id}, section {primary.section or primary.section_no}:"
    lines.append(head)
    for ch in group:
        cite = ctx.book.cite(ch, via="retrieval")
        for raw in ch.content.split("\n"):
            raw = raw.strip()
            if raw:
                lines.append(f"{raw} {cite}")
    refs = [it for it in ctx.evidence if it.via == "crossref" and it.chunk.modality == Modality.TABLE_ROW]
    for it in refs[:3]:
        ch = it.chunk
        cite = ctx.book.cite(ch, via="crossref")
        cells = ch.meta.get("cells", [])
        shown = "; ".join(f"{k} {v}" for k, v in cells)
        lines.append(f"Referenced value ({ch.meta.get('caption', 'table')}): {ch.meta.get('subject', '')}: {shown} {cite}")
    return Composed("\n".join(lines), "procedure")


def _compose_figure(ctx: ComposeContext) -> Composed:
    figs = [it for it in ctx.evidence if it.chunk.modality in (Modality.FIGURE, Modality.CHART, Modality.IMAGE)]
    if not figs:
        return Composed()
    lines = []
    for it in figs[:2]:
        ch = it.chunk
        d = ctx.kb.doc(ch.doc_id)
        cite = ctx.book.cite(ch, via="retrieval", score=it.score)
        cap = ch.meta.get("caption") or ch.content.split("\n")[0]
        s = f"{cap.rstrip('.')} is on page {ch.page} of {d.title if d else ch.doc_id}"
        ocr = (ch.meta.get("ocr") or "").strip()
        if ocr and ch.modality != Modality.IMAGE:
            s += f". Labels visible in the figure: {truncate(ocr, 150)}"
        lines.append(f"{s} {cite}")
    return Composed("\n".join(lines), "figure")


def _compose_extract(ctx: ComposeContext) -> Composed:
    """Fallback: best 1-2 sentences per top evidence chunk by query-term coverage."""
    models = {m.lower() for m in ctx.analysis.models}
    qt = [t for t in ctx.analysis.tokens if not t.startswith("e-") and t not in models]
    if not qt or not ctx.evidence:
        return Composed()
    scored: list[tuple[float, str, EvidenceItem]] = []
    for it in ctx.evidence[:4]:
        if it.chunk.modality in (Modality.WARNING,):
            continue
        for s in split_sentences(it.chunk.content):
            toks = set(content_tokens(s))
            cov = sum(1 for t in qt if t in toks) / len(qt)
            scored.append((cov, s, it))
    scored.sort(key=lambda x: -x[0])
    out, used = [], set()
    for cov, s, it in scored:
        if cov < ctx.min_support or len(out) >= 2 or s in used:
            continue
        used.add(s)
        cite = ctx.book.cite(it.chunk, via=it.via)
        d = ctx.kb.doc(it.chunk.doc_id)
        out.append(f"According to {d.title if d else it.chunk.doc_id}: {s.rstrip('.')} {cite}.")
    return Composed("\n".join(out), "extract")


# ====================================================================================================== dispatch
def compose(ctx: ComposeContext) -> Composed:
    a = ctx.analysis
    if ctx.analytics is not None and ctx.analytics.ok and a.intent == Intent.ANALYTICS:
        return _compose_analytics(ctx)
    if ctx.chart_reads:
        c = _compose_chart(ctx)
        if c.text:
            return c
    if ctx.premise is not None and ctx.premise.verdict in ("confirmed", "refuted", "superseded"):
        c = _compose_premise(ctx)
        if c.text:
            return c
    order = {
        Intent.TROUBLESHOOT: [_compose_troubleshoot, _compose_spec, _compose_procedure],
        Intent.PROCEDURE: [_compose_procedure, _compose_troubleshoot],
        Intent.FIGURE: [_compose_figure],
        Intent.SPEC: [_compose_spec],
        Intent.GENERAL: [_compose_spec, _compose_procedure],
        Intent.ANALYTICS: [_compose_spec],
    }[a.intent]
    for fn in order:
        c = fn(ctx)
        if c.text:
            return c
    return _compose_extract(ctx)
