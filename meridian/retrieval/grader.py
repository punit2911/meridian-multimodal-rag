"""Corrective-RAG style retrieval grader: decide whether the evidence can support an answer before spending an LLM call."""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field

from ..kb import KnowledgeBase
from ..utils.text import tokenize
from .analyzer import QueryAnalysis
from .pipeline import Candidate

# words that carry no answerability signal on their own
GENERIC = {"maximum", "minimum", "rate", "level", "give", "tell", "need", "use", "hold", "make", "work", "get", "want", "approxim", "much", "many",
           "pump", "machin", "unit", "compressor", "correct", "right", "true", "mean", "caus", "check", "keep", "show", "pictur"}


@dataclass
class Grade:
    label: str  # correct | ambiguous | incorrect
    score: float
    coverage: float
    missing: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {"label": self.label, "score": round(self.score, 3), "coverage": round(self.coverage, 3), "missing": self.missing, "reasons": self.reasons}


THRESH_CORRECT = 0.62
THRESH_AMBIGUOUS = 0.40


def grade(cands: list[Candidate], a: QueryAnalysis, kb: KnowledgeBase, depth: int = 4) -> Grade:
    reasons: list[str] = []
    if a.unknown_models:
        return Grade("incorrect", 0.0, 0.0, list(a.unknown_models),
                     [f"No documentation exists for {', '.join(a.unknown_models)}. Known models: {', '.join(kb.known_models())}."])
    if not cands:
        return Grade("incorrect", 0.0, 0.0, list(a.tokens), ["Nothing retrieved."])
    top = cands[:depth]
    bm = kb.ix.bm25
    n = max(len(kb.ix.chunks), 2)
    vocab_terms = set(bm.vocab)
    covered_text = set()
    for c in top:
        covered_text |= set(tokenize(c.chunk.content + " " + c.chunk.section))
    covered_list = list(covered_text)
    total_w, got_w = 0.0, 0.0
    missing: list[str] = []
    q_terms = [t for t in dict.fromkeys(a.tokens) if t not in GENERIC and not t.startswith("e-") and "-" not in t or t.startswith("e-")]
    q_terms = [t for t in q_terms if not (a.models and any(t == m.lower() for m in a.models))]
    # A term is "informative" when it is in the vocabulary and rare enough to discriminate (a company name that sits in every chunk is not).
    floor = math.log(1.0 + (n - 0.25 * n + 0.5) / (0.25 * n + 0.5))
    title_terms = {t for d in kb.docs.values() for t in tokenize(d.title)}  # branding / document-type words say nothing about the topic
    informative = [t for t in q_terms if t in vocab_terms and bm.idf_of(t) >= floor and t not in title_terms]

    def is_hit(t: str) -> bool:
        if t in covered_text:
            return True
        return len(t) >= 5 and bool(difflib.get_close_matches(t, covered_list, n=1, cutoff=0.86))

    anchored = any(is_hit(t) for t in informative)  # at least one specific query term is supported by the evidence
    for t in q_terms:
        in_vocab = t in vocab_terms
        if in_vocab:
            w = bm.idf_of(t)
        else:
            # An out-of-vocabulary word is the rarest word there is. But when the question is anchored on a specific term that *is* found,
            # it is more likely a paraphrase ("generate", "handle", "give") than a missing topic, so it counts as an ordinary rare word.
            w = floor if anchored else math.log(1.0 + n)
        total_w += w
        if is_hit(t):
            got_w += w
        else:
            missing.append(t)
    coverage = (got_w / total_w) if total_w else 0.5
    oov = [t for t in missing if t not in vocab_terms]
    if "brand" in q_terms and not is_hit("brand"):
        return Grade("incorrect", 0.2 * coverage, coverage, missing,
                     ["the requested brand detail is not present in the evidence"])
    if oov and not anchored:
        oov_share = sum(math.log(1.0 + n) for _ in oov) / total_w if total_w else 0.0
        if oov_share >= 0.3:
            return Grade("incorrect", 0.2 * coverage, coverage, missing,
                         [f"key term(s) that appear in no indexed document: {', '.join(oov[:5])}"])
    top_final = max((c.final for c in top), default=0.0)
    late = max((c.feats.get("late", 0.0) for c in top), default=0.0)
    score = 0.55 * coverage + 0.25 * late + 0.20 * min(1.0, top_final)
    if a.codes:
        present = {e for c in top for e in c.chunk.entities}
        miss_codes = [c for c in a.codes if c not in present]
        if miss_codes:
            score *= 0.5
            reasons.append(f"alarm code {', '.join(miss_codes)} not found in the top evidence")
    if missing:
        reasons.append("query terms with no support in the evidence: " + ", ".join(missing[:6]))
    label = "correct" if score >= THRESH_CORRECT else "ambiguous" if score >= THRESH_AMBIGUOUS else "incorrect"
    return Grade(label, score, coverage, missing, reasons)
