"""The agent orchestrator (policy agent).

A deterministic controller with explicit decisions at every step, all recorded in a trace the UI renders:

  perceive (photo -> text/codes/model)  ->  analyze  ->  route
  -> retrieve -> grade (CRAG) -> [corrective retry: relax filters / keyword query] -> abstain if still unsupported
  -> expand (cross-references, safety warnings) -> tools (spec lookup + conflict resolution, chart reader, log analytics, premise check)
  -> compose (evidence-extractive, cited) -> verify claims -> repair / abstain -> calibrated confidence
"""
from __future__ import annotations

import json
import re
import math
import time
from pathlib import Path
from typing import Callable, Optional

from ..config import Settings
from ..kb import KnowledgeBase
from ..retrieval.analyzer import Intent, QueryAnalysis, analyze
from ..retrieval.context import ContextConfig, EvidenceItem, select_evidence
from ..retrieval.grader import Grade, grade
from ..retrieval.perception import Perception, perceive
from ..retrieval.pipeline import HybridRetriever, RetrievalConfig, RetrievalResult
from ..schemas import AgentResult, ClaimVerdict, Modality, TraceEvent
from ..utils.text import split_sentences, tokenize, truncate
from ..verify.verifier import Verifier, faithfulness, repair
from .compose import TEMPLATE_TERMS, CiteBook, Composed, ComposeContext, compose
from .tools import (AnalyticsResult, ChartRead, PremiseResult, SpecResult, check_premise, lookup_spec, read_chart_values, run_analytics)

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "resources" / "calibration.json"
MAX_QUESTION_CHARS = 2000
DEFAULT_IMAGE_QUESTION = "What does this show and what should I do about it?"


GENERIC_TERMS = {"much", "mani", "need", "requir", "what", "should", "must", "pump", "unit", "machin"}


class AgentError(ValueError):
    """User-correctable input problem (empty question, unreadable image)."""


def load_calibration() -> Optional[dict]:
    try:
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def calibrate(raw: float, params: Optional[dict]) -> float:
    if not params:
        return max(0.0, min(1.0, raw))
    z = params["a"] * raw + params["b"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class Orchestrator:
    mode = "policy"

    def __init__(self, kb: KnowledgeBase, settings: Settings, retriever: Optional[HybridRetriever] = None):
        self.kb = kb
        self.s = settings
        self.retriever = retriever or HybridRetriever(kb)
        self.calibration = load_calibration()

    # ------------------------------------------------------------------------------------------ public
    def answer(self, question: str, image: Optional[bytes] = None, *, retrieval_cfg: Optional[RetrievalConfig] = None,
               context_cfg: Optional[ContextConfig] = None, on_event: Optional[Callable[[TraceEvent], None]] = None) -> AgentResult:
        t_start = time.perf_counter()
        question = (question or "").strip()
        if not question and image:
            question = DEFAULT_IMAGE_QUESTION
        if not question:
            raise AgentError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise AgentError(f"The question is too long ({len(question)} characters, maximum {MAX_QUESTION_CHARS}).")
        trace: list[TraceEvent] = []
        timings: dict[str, float] = {}
        warnings: list[str] = []

        def emit(step: str, title: str, detail: str = "", data: Optional[dict] = None, ms: float = 0.0, status: str = "ok") -> None:
            ev = TraceEvent(step=step, title=title, detail=detail, data=data or {}, ms=round(ms, 1), status=status)
            trace.append(ev)
            if on_event:
                on_event(ev)

        def finish(**kw) -> AgentResult:
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            return AgentResult(question=question, trace=trace, timings=timings, mode=self.mode, warnings=warnings, **kw)

        kb = self.kb
        if not kb.chunks:
            emit("abstain", "Knowledge base is empty", "Upload documents or load the demo corpus first.", status="fail")
            return finish(answer="The knowledge base is empty. Upload documents (PDF, DOCX, CSV, images) or load the demo corpus first.", abstained=True,
                          abstain_reason="empty_knowledge_base", confidence=0.0)

        # ---- 1. perception
        perception: Optional[Perception] = None
        if image:
            t = time.perf_counter()
            try:
                perception = perceive(image, lang=self.s.ocr_lang)
            except ValueError as exc:
                raise AgentError(str(exc)) from exc
            ms = (time.perf_counter() - t) * 1000
            timings["perception_ms"] = round(ms, 1)
            emit("perceive", "Read the uploaded image", f"{perception.kind}; text: '{truncate(perception.text, 90)}'; codes: {perception.codes or '-'}; models: {perception.models or '-'}",
                 perception.public(), ms, "warn" if perception.notes else "ok")
            warnings.extend(perception.notes)

        # ---- 2. analysis
        t = time.perf_counter()
        known = set(kb.known_models())
        a = analyze(question, known, image_models=perception.models if perception else None, image_codes=perception.codes if perception else None,
                    image_text=perception.text if perception else "")
        timings["analyze_ms"] = round((time.perf_counter() - t) * 1000, 1)
        emit("analyze", f"Understood the question as '{a.intent.value}'", self._analysis_detail(a), a.summary(), (time.perf_counter() - t) * 1000)
        p_public = perception.public() if perception else {}

        # ---- 3. early exits
        if a.unknown_models:
            msg = (f"I have no documentation for the {', '.join(a.unknown_models)}. The documented models are {', '.join(sorted(known))}.")
            emit("grade", "Unknown product model", msg, {"label": "incorrect"}, status="fail")
            emit("abstain", "Declined to answer", "Answering for a different model would be unsafe.", status="warn")
            return finish(answer=msg, abstained=True, abstain_reason="unknown_model", confidence=0.0, intent=a.intent.value, analysis={"analysis": a.summary()},
                          perception=p_public)

        analytics: Optional[AnalyticsResult] = None
        spec = SpecResult()
        premise: Optional[PremiseResult] = None
        chart_reads: list[ChartRead] = []
        evidence: list[EvidenceItem] = []
        g = Grade("correct", 1.0, 1.0)
        res: Optional[RetrievalResult] = None

        # ---- 4. analytics route (computed, not retrieved)
        if a.intent == Intent.ANALYTICS:
            t = time.perf_counter()
            analytics = run_analytics(kb, a)
            ms = (time.perf_counter() - t) * 1000
            timings["analytics_ms"] = round(ms, 1)
            emit("tool", "Computed over the maintenance log" if analytics.ok else "Log analytics not applicable",
                 analytics.text if analytics.ok else analytics.reason, {"code": analytics.code, "rows_used": analytics.rows_used, "rows_total": analytics.rows_total}, ms,
                 "ok" if analytics.ok else "warn")

        # ---- 5. retrieval with corrective loop
        if analytics is None or not analytics.ok:
            res, g, evidence = self._retrieve_loop(a, perception, retrieval_cfg, context_cfg, emit, timings)
            if res is None or g.label == "incorrect":
                reason = "; ".join(g.reasons) or "The documents do not contain this information."
                top = [self._cite_hint(e) for e in evidence[:2]]
                msg = ("I could not find this in the indexed documents, so I will not guess. " + reason +
                       (f" Closest material: {'; '.join(top)}." if top else ""))
                emit("abstain", "Declined to answer", reason, {"grade": g.public()}, status="warn")
                return finish(answer=msg, abstained=True, abstain_reason="insufficient_evidence", confidence=round(calibrate(0.3 * g.score, self.calibration), 3),
                              intent=a.intent.value, analysis={"analysis": a.summary(), "grade": g.public()}, perception=p_public)

            # ---- 6. tools over the evidence
            t = time.perf_counter()
            ev_ids = {e.chunk.id for e in evidence}
            cross_ids = {e.chunk.id for e in evidence if e.via == "crossref"}
            if a.intent in (Intent.SPEC, Intent.GENERAL) or a.yes_no:
                spec = lookup_spec(kb, a, restrict_chunks=ev_ids, crossref_chunks=cross_ids)
                if a.yes_no:
                    premise = check_premise(a, spec)
                if spec.selected and not a.yes_no and self._prose_answers_better(a, spec, evidence):
                    emit("tool", "Prose passage matches the question better than the spec table", "using the passage instead of the structured fact", {})
                    spec = SpecResult()
            chart_chunks = [e.chunk for e in evidence if e.chunk.modality == Modality.CHART]
            for c in res.candidates[:10]:
                if c.chunk.modality == Modality.CHART and c.chunk not in chart_chunks:
                    chart_chunks.append(c.chunk)
            if a.quantities and a.intent in (Intent.SPEC, Intent.GENERAL, Intent.FIGURE):
                chart_reads = read_chart_values(kb, a, chart_chunks)
            ms = (time.perf_counter() - t) * 1000
            timings["tools_ms"] = round(ms, 1)
            if spec.selected or chart_reads or premise:
                emit("tool", "Structured lookups", self._tools_detail(spec, premise, chart_reads),
                     {"facts": [{"subject": f.subject, "attribute": f.attribute, "value": f.display, "model": f.model, "source": f.doc_title} for f in spec.selected],
                      "conflicts": [c.model_dump() for c in spec.conflicts], "premise": premise.verdict if premise else None,
                      "chart_reads": [{"x": r.x_label, "y": round(r.y, 2), "interpolated": r.interpolated} for r in chart_reads]}, ms)
            for c in spec.conflicts:
                emit("conflict", f"Sources disagree on '{c.subject}'", f"{c.loser}: {c.loser_value}  vs  {c.winner}: {c.winner_value}. Rule: {c.rule}.",
                     c.model_dump(), 0.0, "warn")

        # ---- 7. compose
        t = time.perf_counter()
        book = CiteBook(kb)
        ctx = ComposeContext(kb=kb, analysis=a, evidence=evidence, spec=spec, premise=premise, chart_reads=chart_reads, analytics=analytics, book=book)
        composed: Composed = self._compose(ctx, question, emit)
        ms = (time.perf_counter() - t) * 1000
        timings["compose_ms"] = round(ms, 1)
        if composed.kind == "extract" and g.label != "correct":
            composed = Composed()  # a loose sentence match is not good enough when retrieval itself is doubtful
        if not composed.text.strip():
            emit("abstain", "Nothing answerable was found in the evidence", "The retrieved passages do not contain a direct answer.", {"grade": g.public()}, status="warn")
            return finish(answer="I found related passages but none that directly answers the question, so I will not guess. Try naming the product model, alarm code or the exact parameter.",
                          abstained=True, abstain_reason="no_direct_answer", confidence=round(calibrate(0.25 * g.score, self.calibration), 3), intent=a.intent.value,
                          analysis={"analysis": a.summary(), "grade": g.public()}, perception=p_public)
        emit("compose", f"Composed a cited answer ({composed.kind})", f"{len(book.order)} evidence item(s) cited", {"kind": composed.kind}, ms)
        warnings.extend(composed.notes)

        # ---- 8. verify (+ repair)
        t = time.perf_counter()
        verifier = Verifier(idf=lambda tok: kb.ix.bm25.idf_of(tok))
        verdicts = verifier.verify(composed.text, book.evidence_docs(), ignore_terms=self._ignore_terms(composed), echo_text=question)
        faith = faithfulness(verdicts)
        if composed.kind == "llm" and (faith < self.s.faithfulness_threshold or any(v.verdict == "contradicted" for v in verdicts)):
            emit("repair", "LLM answer failed verification; fell back to the extractive composer",
                 "; ".join(truncate(v.text, 80) for v in verdicts if v.verdict != "supported"), {}, 0.0, "warn")
            book.reset()
            composed = compose(ctx)
            verdicts = verifier.verify(composed.text, book.evidence_docs(), ignore_terms=self._ignore_terms(composed), echo_text=question) if composed.text.strip() else []
            faith = faithfulness(verdicts) if verdicts else 0.0
            if not composed.text.strip():
                emit("abstain", "Nothing answerable was found in the evidence", "", {}, status="warn")
                return finish(answer="I could not verify an answer against the documents, so I will not present one.", abstained=True, abstain_reason="failed_verification",
                              confidence=0.1, faithfulness=0.0, intent=a.intent.value, analysis={"analysis": a.summary(), "grade": g.public()}, perception=p_public)
        text = composed.text
        status = "ok"
        if faith < self.s.faithfulness_threshold or any(v.verdict == "contradicted" for v in verdicts):
            text2, removed = repair(text, verdicts)
            status = "warn"
            emit("repair", "Removed unsupported statements", "; ".join(truncate(r, 80) for r in removed) or "none", {"removed": removed}, 0.0, "warn")
            if text2.strip():
                text = text2
                verdicts = verifier.verify(text, book.evidence_docs(), ignore_terms=self._ignore_terms(composed), echo_text=question)
                faith = faithfulness(verdicts)
        ms = (time.perf_counter() - t) * 1000
        timings["verify_ms"] = round(ms, 1)
        emit("verify", f"Claim verification: faithfulness {faith:.0%}", f"{sum(v.verdict == 'supported' for v in verdicts)}/{len(verdicts)} claims fully supported",
             {"claims": [{"text": v.text, "verdict": v.verdict, "score": v.score} for v in verdicts]}, ms, status)
        if faith < 0.4:
            emit("abstain", "Declined: answer could not be verified", "", status="fail")
            return finish(answer="I could not verify an answer against the documents, so I will not present one.", abstained=True, abstain_reason="failed_verification",
                          confidence=0.1, faithfulness=faith, intent=a.intent.value, claims=verdicts, analysis={"analysis": a.summary(), "grade": g.public()}, perception=p_public)

        # ---- 9. confidence
        raw = 0.5 * min(1.0, g.score) + 0.5 * faith
        if g.label == "ambiguous":
            raw *= 0.85
            warnings.append("The retrieved evidence only partly matches the question; please verify against the source page.")
        if a.needs_model:
            warnings.append("No product model was specified, so the answer covers every documented model that matches.")
        conf = round(calibrate(raw, self.calibration), 3)
        emit("finalize", f"Confidence {conf:.0%}", f"raw {raw:.2f} (grade {g.score:.2f}, faithfulness {faith:.2f})", {"confidence": conf})
        return finish(answer=text, confidence=conf, faithfulness=round(faith, 3), intent=a.intent.value, citations=book.citations(), claims=verdicts, conflicts=spec.conflicts,
                      analysis={"analysis": a.summary(), "grade": g.public(), "kind": composed.kind}, perception=p_public)

    def _prose_answers_better(self, a: QueryAnalysis, spec: SpecResult, evidence: list[EvidenceItem]) -> bool:
        """A numeric sentence in running text can answer the question more precisely than a spec row that merely shares a few words with it."""
        terms = {t for t in a.tokens if not t.startswith(("hx-", "e-")) and t not in GENERIC_TERMS}
        if len(terms) < 2:
            return False
        idf = self.kb.ix.bm25.idf_of

        def cover(tokens: set[str]) -> float:
            tot = sum(max(idf(t), 0.5) for t in terms)
            return sum(max(idf(t), 0.5) for t in terms if t in tokens) / tot if tot else 0.0

        best_fact = 0.0
        for f in spec.selected:
            ch = self.kb.chunk(f.chunk_id)
            best_fact = max(best_fact, cover(set(tokenize(f.subject + " " + f.attribute + " " + (ch.content if ch else "")))))
        best_sent = 0.0
        for it in evidence:
            if it.chunk.modality != Modality.TEXT:
                continue
            for sent in split_sentences(it.chunk.content):
                if re.search(r"\d", sent):
                    best_sent = max(best_sent, cover(set(tokenize(sent))))
        return best_sent >= best_fact + 0.15

    # ------------------------------------------------------------------------------------------ hooks (overridden by the LLM agent)
    def _compose(self, ctx: ComposeContext, question: str, emit) -> Composed:
        return compose(ctx)

    @staticmethod
    def _ignore_terms(composed: Composed) -> frozenset:
        return TEMPLATE_TERMS if composed.kind != "llm" else frozenset()  # free-form model text gets no template allowance

    # ------------------------------------------------------------------------------------------ retrieval loop
    def _retrieve_loop(self, a: QueryAnalysis, perception: Optional[Perception], rcfg: Optional[RetrievalConfig], ccfg: Optional[ContextConfig], emit, timings: dict):
        base_cfg = rcfg or RetrievalConfig()
        best: Optional[tuple[RetrievalResult, Grade, list[EvidenceItem]]] = None
        attempts = [("full pipeline", a, base_cfg, False)]
        if a.models:
            attempts.append(("relaxed model filter", a, base_cfg, True))
        kw = [t for t in dict.fromkeys(a.index_tokens) if len(t) > 2][:10]
        if kw:
            a3 = QueryAnalysis(**{**a.__dict__, "tokens": kw, "index_tokens": kw})
            attempts.append(("keyword-only rewrite", a3, base_cfg, bool(a.models)))
        rounds = 0
        for name, qa, cfg, relax in attempts[: self.s.max_retrieval_rounds]:
            rounds += 1
            t = time.perf_counter()
            res = self.retriever.retrieve(qa, cfg, perception, exclude_models=relax)
            ev = self._guard_models(select_evidence(res, self.kb, ccfg), a)
            g = grade(res.candidates, a, self.kb)  # always judged against the user's real question, never the rewrite
            ms = (time.perf_counter() - t) * 1000
            timings[f"retrieve_{rounds}_ms"] = round(ms, 1)
            emit("retrieve", f"Retrieved ({name})", f"{len(res.candidates)} candidates, {len(ev)} evidence items; grade {g.label} ({g.score:.2f})",
                 {"round": rounds, "channels": sorted({ch for c in res.candidates[:20] for ch in c.channels()}), "prf_terms": res.prf_terms,
                  "visual_used": res.visual_used, "timings": {k: round(v, 1) for k, v in res.timings.items()},
                  "top": [{"id": c.chunk.id, "modality": c.chunk.modality.value, "score": round(c.final, 3), "channels": c.channels(), "text": truncate(c.chunk.content, 90)}
                          for c in res.candidates[:6]]}, ms)
            emit("grade", f"Evidence grade: {g.label}", "; ".join(g.reasons) or f"coverage {g.coverage:.0%}", g.public(), 0.0,
                 {"correct": "ok", "ambiguous": "warn", "incorrect": "fail"}[g.label])
            if ev:
                for it in ev:
                    if it.via != "retrieval":
                        emit("expand", f"Added {it.via} evidence", it.note or truncate(it.chunk.content, 90), {"chunk": it.chunk.id, "via": it.via},
                             0.0)
            if best is None or g.score > best[1].score:
                best = (res, g, ev)
            if g.label == "correct":
                break
            if rounds < len(attempts[: self.s.max_retrieval_rounds]):
                emit("retry", "Evidence insufficient: trying a corrective retrieval", "; ".join(g.reasons[:2]), {"next": attempts[rounds][0]}, 0.0, "warn")
        assert best is not None
        return best

    def _guard_models(self, evidence: list[EvidenceItem], a: QueryAnalysis) -> list[EvidenceItem]:
        """Never let evidence about another product slip in when the user named a model."""
        if not a.models:
            return evidence
        out = []
        for it in evidence:
            d = self.kb.doc(it.chunk.doc_id)
            if d is None or not d.models or set(d.models) & set(a.models):
                out.append(it)
        return out

    # ------------------------------------------------------------------------------------------ text helpers
    @staticmethod
    def _analysis_detail(a: QueryAnalysis) -> str:
        parts = []
        if a.models:
            parts.append(f"model {', '.join(a.models)}" + (" (from photo)" if a.models_from_image else ""))
        if a.codes:
            parts.append(f"alarm code {', '.join(a.codes)}")
        if a.quantities:
            parts.append("numbers: " + ", ".join(f"{q.raw.strip()}" for q in a.quantities))
        if a.yes_no:
            parts.append("yes/no claim to verify")
        if a.needs_model:
            parts.append("no model named")
        return "; ".join(parts) or "no explicit entities"

    @staticmethod
    def _tools_detail(spec: SpecResult, premise: Optional[PremiseResult], reads: list[ChartRead]) -> str:
        bits = []
        if spec.selected:
            bits.append("; ".join(f"{f.subject} = {f.display} ({f.doc_type})" for f in spec.selected[:3]))
        if premise:
            bits.append(f"premise {premise.verdict}")
        if reads:
            bits.append("chart: " + "; ".join(f"{r.x_label} -> {r.y:.1f}" for r in reads))
        return " | ".join(bits)

    def _cite_hint(self, it: EvidenceItem) -> str:
        d = self.kb.doc(it.chunk.doc_id)
        return f"{d.title if d else it.chunk.doc_id} p.{it.chunk.page}"
