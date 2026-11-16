"""Claim-level grounding verifier (hallucination detector).

An answer is split into atomic claims. Each claim is checked against the evidence it cites (or all evidence when it cites none):

  numeric check   every number in the claim must exist, unit-aware (45 Nm == 33.2 lb-ft), in the evidence; a same-dimension
                  number that differs is a *contradiction*, an absent number is *unsupported*
  entity check    alarm codes / model names / part numbers in the claim must appear in the evidence
  lexical check   IDF-weighted coverage of the claim's content terms by the best evidence sentence / the cited union
  polarity check  a claim that matches an evidence sentence closely but flips its negation ("do NOT ..." vs "do ...") is contradicted

Deterministic, no LLM: it can audit LLM answers, extractive answers, and is cheap enough to run inside the repair loop.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..schemas import ClaimVerdict, Quantity
from ..utils import units as U
from ..utils.text import (content_tokens, find_citations, has_negation, split_sentences, strip_citations, tokenize)

_ID_RE = re.compile(r"\b[A-Z]{1,4}-\d{2,5}(?:-\d{1,3})?[A-Z]?\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_META_PREFIXES = ("table", "figure", "section", "step", "page", "p.")
_NON_CLAIM = re.compile(r"^(note|safety|warning|caution|sources?|evidence|answer|procedure)\b[^.]{0,40}:?\s*$", re.I)


@dataclass
class EvidenceDoc:
    index: int
    text: str
    doc_id: str = ""
    kind: str = "chunk"  # chunk | tool
    _sents: list[list[str]] = field(default_factory=list, repr=False)
    _qty: list[Quantity] = field(default_factory=list, repr=False)
    _toks: set[str] = field(default_factory=set, repr=False)
    _ids: set[str] = field(default_factory=set, repr=False)
    _dates: set[str] = field(default_factory=set, repr=False)

    def prepare(self) -> "EvidenceDoc":
        sents = split_sentences(self.text) or [self.text]
        # a table row or line break is its own unit; also add overlapping 2-sentence windows for claims that span sentences
        units = list(sents) + [" ".join(sents[i:i + 2]) for i in range(len(sents) - 1)]
        self._sents = [content_tokens(s) for s in units]
        self._units = units
        self._qty = U.extract_quantities(self.text)
        self._toks = set(content_tokens(self.text))
        self._ids = {m.group(0).upper() for m in _ID_RE.finditer(U.normalize_unit_text(self.text))}
        self._dates = set(_DATE_RE.findall(self.text))
        return self


@dataclass
class VerifierConfig:
    supported_at: float = 0.72
    partial_at: float = 0.48
    numeric_rel_tol: float = 0.004


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """(claim text without citations, cited evidence indices). Bullet / numbered lines are separate claims."""
    claims: list[tuple[str, list[int]]] = []
    for line in re.split(r"\n+", answer or ""):
        line = line.strip().lstrip("-*•> ").strip()
        line = re.sub(r"\*\*", "", line)
        if not line:
            continue
        for sent in split_sentences(line):
            cites = find_citations(sent)
            text = strip_citations(sent).strip()
            text = re.sub(r"^\d{1,2}[.)]\s+", "", text)
            if text.endswith(":"):  # a list intro / heading, not a factual claim
                continue
            if len(re.findall(r"[A-Za-z0-9]+", text)) < 2 or _NON_CLAIM.match(text):
                continue
            claims.append((text, cites))
    return claims


def _claim_quantities(text: str) -> list[Quantity]:
    """Numbers that carry factual weight: skip identifiers (HX-200, E-47), table/step/section numbers and list markers."""
    cleaned = _DATE_RE.sub(" ", _ID_RE.sub(" ", U.normalize_unit_text(text)))
    out = []
    for q in U.extract_quantities(cleaned):
        prefix = cleaned[max(0, q.start - 14): q.start].lower()
        if q.unit is None and (re.search(r"(table|figure|fig\.?|section|step|steps|page|item|rev(?:ision)?|sb)\s*$", prefix) or re.search(r"\bto\s*$", prefix) and re.search(r"steps?\s+\d+\s+to\s*$", prefix)):
            continue
        out.append(q)
    return out


def _weighted_coverage(claim_tokens: list[str], ev_tokens: set[str], idf: Callable[[str], float]) -> float:
    if not claim_tokens:
        return 1.0
    total = got = 0.0
    for t in dict.fromkeys(claim_tokens):
        w = max(idf(t), 0.4)
        total += w
        if t in ev_tokens:
            got += w
    return got / total if total else 0.0


_REF_RE = re.compile(r"\b(?:Table|Figure|Fig\.|Section)\s+\d+(?:\.\d+)*", re.I)


def echo_refutation(text: str, echo: Sequence[Quantity]) -> bool:
    """True when the claim quotes the user's own number back in a correction ('... is 140 degC, not 150 degC')."""
    if not echo:
        return False
    nq = _claim_quantities(text)
    return any(U.quantities_match(q, e) for q in nq for e in echo if q.unit and e.unit)


class Verifier:
    def __init__(self, idf: Optional[Callable[[str], float]] = None, cfg: Optional[VerifierConfig] = None):
        self.idf = idf or (lambda t: 1.0)
        self.cfg = cfg or VerifierConfig()

    def verify(self, answer: str, evidence: Sequence[EvidenceDoc], *, ignore_terms: frozenset = frozenset(), echo_text: str = "") -> list[ClaimVerdict]:
        """`ignore_terms`: connective words of a trusted template (composer). `echo_text`: the user's question; numbers the answer merely
        *quotes back* while refuting them ("... is 140 degC, not 150 degC") are not hallucinations."""
        docs = [e if e._sents else e.prepare() for e in evidence]
        by_index = {d.index: d for d in docs}
        echo = U.extract_quantities(echo_text) if echo_text else []
        out: list[ClaimVerdict] = []
        for text, cites in split_claims(answer):
            out.append(self._verify_claim(text, cites, docs, by_index, ignore_terms, echo))
        return out

    # ------------------------------------------------------------------------------------------ one claim
    def _verify_claim(self, text: str, cites: list[int], docs: list[EvidenceDoc], by_index: dict[int, EvidenceDoc], ignore_terms: frozenset = frozenset(),
                      echo: Sequence[Quantity] = ()) -> ClaimVerdict:
        reasons: list[str] = []
        cited_docs = [by_index[c] for c in cites if c in by_index]
        bad_cites = [c for c in cites if c not in by_index]
        if bad_cites:
            reasons.append(f"cites non-existent evidence [{', '.join(map(str, bad_cites))}]")
        pool = cited_docs or docs  # fall back to all evidence when uncited
        if cites and not cited_docs:
            pool = docs
        # --- lexical support
        stripped = U.normalize_unit_text(text)
        for q in sorted(U.extract_quantities(stripped), key=lambda q: -q.start):  # numbers/units are validated separately
            stripped = stripped[:q.start] + " " + stripped[q.end:]
        ctoks = [t for t in content_tokens(stripped) if t not in ignore_terms]
        best_sent = 0.0
        best_unit, best_doc = "", None
        for d in pool:
            for toks, unit in zip(d._sents, d._units):
                cov = _weighted_coverage(ctoks, set(toks), self.idf)
                if cov > best_sent:
                    best_sent, best_unit, best_doc = cov, unit, d
        union = set().union(*(d._toks for d in pool)) if pool else set()
        union_cov = _weighted_coverage(ctoks, union, self.idf)
        lex = max(best_sent, 0.92 * union_cov)

        # --- numeric check
        nq = _claim_quantities(text)
        numeric = bool(nq)
        contradicted = False
        unmatched: list[Quantity] = []
        if nq:
            ev_q = [q for d in pool for q in d._qty]
            all_q = [q for d in docs for q in d._qty]
            refuting = bool(re.search(r"\b(not|no longer|outdated|instead of|rather than|superseded|differs?)\b", text, re.I))
            for q in nq:
                if any(self._qty_match(q, e) for e in ev_q):
                    continue
                if refuting and any(self._qty_match(q, e) for e in echo):
                    continue  # the user's own number, quoted back in a correction
                unmatched.append(q)
                same_dim = [e for e in ev_q if e.unit and q.unit and U.unit_dimension(e.unit) == U.unit_dimension(q.unit)]
                if same_dim:
                    contradicted = True
                    reasons.append(f"{U.format_quantity(q.value, q.unit)} does not match the evidence ({U.format_quantity(same_dim[0].value, same_dim[0].unit)})")
                elif any(self._qty_match(q, e) for e in all_q):
                    reasons.append(f"{U.format_quantity(q.value, q.unit)} appears only in evidence that was not cited")
                else:
                    reasons.append(f"number {U.format_quantity(q.value, q.unit)} is not in the evidence")
        # --- entity check
        claim_ids = {m.group(0).upper() for m in _ID_RE.finditer(U.normalize_unit_text(text))}
        pool_ids = set().union(*(d._ids for d in pool)) if pool else set()
        missing_ids = sorted(i for i in claim_ids if i not in pool_ids)
        pool_dates = set().union(*(d._dates for d in pool)) if pool else set()
        missing_ids += sorted(d for d in _DATE_RE.findall(text) if d not in pool_dates)
        if missing_ids:
            reasons.append(f"identifier(s) not in the evidence: {', '.join(missing_ids)}")
        # --- document references (Table 3, Section 6.1, Figure 2) must exist in the cited evidence
        pool_text = " ".join(d.text for d in pool).lower()
        for m in _REF_RE.finditer(text):
            ref = re.sub(r"\s+", " ", m.group(0).lower()).replace("fig.", "figure")
            if ref not in pool_text.replace("fig.", "figure"):
                reasons.append(f"reference '{m.group(0)}' is not in the evidence")
                missing_ids.append(m.group(0))
        # --- a negation the evidence never makes (not a quoted correction of the user's premise) flips the meaning
        introduced_neg = False
        if has_negation(text) and not echo_refutation(text, echo) and not re.match(r"\s*(no|yes)\b", text, re.I) and len(ctoks) >= 2:
            if pool and not any(has_negation(u) for d in pool for u in d._units) and not has_negation(pool_text):
                introduced_neg = True
                contradicted = True
                reasons.append("the claim is negated but nothing in the evidence is")
        # --- polarity check against the best-matching evidence unit
        if best_doc is not None and best_sent >= 0.6 and best_unit:
            if has_negation(text) != has_negation(best_unit) and len(ctoks) >= 3:
                # only flag when the evidence sentence is a near-paraphrase (high mutual overlap)
                utoks = set(content_tokens(best_unit))
                mutual = len(set(ctoks) & utoks) / max(len(set(ctoks) | utoks), 1)
                if mutual >= 0.5:
                    contradicted = True
                    reasons.append("negation differs from the matching evidence sentence")

        score = lex
        if nq:
            ok_ratio = (len(nq) - len(unmatched)) / len(nq)
            score = min(score, 0.35 + 0.65 * ok_ratio) if unmatched else score
        if missing_ids:
            score = min(score, 0.45)
        if contradicted:
            verdict, score = "contradicted", min(score, 0.2)
        elif unmatched and not any(self._qty_match(q, e) for q in unmatched for d in docs for e in d._qty):
            verdict, score = "unsupported", min(score, 0.4)
        elif score >= self.cfg.supported_at and not missing_ids:
            verdict = "supported"
        elif score >= self.cfg.partial_at:
            verdict = "partial"
        else:
            verdict = "unsupported"
            reasons.append("claim terms are not covered by the evidence")
        if unmatched and verdict in ("supported", "partial"):
            verdict, score = "partial", min(score, 0.6)
        return ClaimVerdict(text=text, verdict=verdict, score=round(float(score), 3), cited=cites, reasons=reasons, numeric=numeric)

    def _qty_match(self, a: Quantity, b: Quantity) -> bool:
        if a.unit and b.unit:
            return U.quantities_match(a, b, rel_tol=self.cfg.numeric_rel_tol)
        if a.unit and not b.unit:
            return False
        return abs(a.value - b.value) <= max(1e-9, self.cfg.numeric_rel_tol * max(abs(a.value), abs(b.value), 1.0))


def faithfulness(verdicts: Sequence[ClaimVerdict]) -> float:
    """Share of supported claims (partial counts half; contradictions/unsupported count zero). 1.0 for an empty answer."""
    if not verdicts:
        return 1.0
    w = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "contradicted": 0.0}
    return sum(w[v.verdict] for v in verdicts) / len(verdicts)


def repair(answer: str, verdicts: Sequence[ClaimVerdict]) -> tuple[str, list[str]]:
    """Remove lines that carry only unsupported/contradicted claims. Returns (new answer, removed claim texts)."""
    bad = {v.text for v in verdicts if v.verdict in ("unsupported", "contradicted")}
    if not bad:
        return answer, []
    kept, removed = [], []
    for line in (answer or "").split("\n"):
        plain = strip_citations(re.sub(r"\*\*", "", line)).strip().lstrip("-*•> ").strip()
        plain = re.sub(r"^\d{1,2}[.)]\s+", "", plain)
        sents = [strip_citations(s).strip() for s in split_sentences(plain)] or [plain]
        if sents and all(s in bad or not s for s in sents) and plain:
            removed.append(plain)
            continue
        if any(s in bad for s in sents):  # drop only the offending sentences inside a mixed line
            keep = [s for s in split_sentences(line) if strip_citations(re.sub(r"\*\*", "", s)).strip().lstrip("-*•> ") not in bad]
            removed.extend(s for s in sents if s in bad)
            line = " ".join(keep)
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), removed
