"""Deterministic tools used by the agents. None of them calls an LLM and none evaluates user-supplied code."""
from __future__ import annotations

import math

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..index.facts import resolve_conflicts
from ..ingest.figures import interpolate_series
from ..kb import KnowledgeBase
from ..retrieval.analyzer import QueryAnalysis
from ..schemas import Chunk, ConflictNote, Fact, Modality, Quantity
from ..utils import units as U
from ..utils.text import content_tokens, tokenize


# ====================================================================================================== spec lookup
@dataclass
class SpecResult:
    selected: list[Fact] = field(default_factory=list)
    conflicts: list[ConflictNote] = field(default_factory=list)
    superseded: list[Fact] = field(default_factory=list)  # values that lost a conflict
    scored: list[tuple[Fact, float]] = field(default_factory=list)


def known_attributes(kb: KnowledgeBase) -> set[str]:
    return {t for f in kb.ix.facts.facts if f.attribute not in ("value", "chart value") for t in content_tokens(f.attribute)}


def lookup_spec(kb: KnowledgeBase, a: QueryAnalysis, restrict_chunks: Optional[set[str]] = None, crossref_chunks: Optional[set[str]] = None,
                top_n: int = 4, min_score: float = 0.45) -> SpecResult:
    """Find the exact fact(s) the question asks for, then resolve conflicts between sources.

    Attribute awareness: asking about a "torque" boosts torque facts and demotes interval facts of a similar-looking row.
    """
    attrs = known_attributes(kb)
    q_attrs = attrs & set(a.tokens)
    hits = kb.ix.facts.search(a.tokens, models=a.models or None, unit_dim=a.unit_dim if a.yes_no else None, min_score=min_score * (0.7 if a.yes_no else 1.0),
                              include_approx=False)
    scored: list[tuple[Fact, float]] = []
    bm = kb.ix.bm25
    n_chunks = max(len(kb.ix.chunks), 4)
    rare_floor = math.log(1.0 + (n_chunks - 3 + 0.5) / (3 + 0.5))  # a term found in at most ~3 chunks
    rare_terms = [t for t in a.tokens if t in bm.vocab and bm.idf_of(t) >= rare_floor and not t.startswith(("hx-", "e-"))]
    for f, s in hits:
        if rare_terms:  # the question names something specific (e.g. 'baseplate'); a fact that never mentions it answers a different question
            ch = kb.chunk(f.chunk_id)
            hay = set(tokenize((ch.content if ch else "") + " " + f.subject + " " + f.attribute))
            if not any(t in hay for t in rare_terms):
                continue
        if restrict_chunks is not None and f.chunk_id not in restrict_chunks:
            continue
        ftoks = set(content_tokens(f.attribute)) if f.attribute not in ("value", "chart value") else set()
        if q_attrs:
            if ftoks & q_attrs:
                s += 0.5
            elif f.attribute == "value" or (ftoks and not ftoks & q_attrs):
                s -= 0.3
        if crossref_chunks and f.chunk_id in crossref_chunks and q_attrs & ftoks:
            s += 0.2
        scored.append((f, s))
    scored.sort(key=lambda x: -x[1])
    if not scored:
        return SpecResult()
    best = scored[0][1]
    near = [(f, s) for f, s in scored if s >= max(min_score, 0.8 * best)]
    from ..index.facts import group_key
    present = {id(f) for f, _ in near}
    for f, s in list(near):  # companions: the same quantity as stated by other sources, so a conflict can never be missed
        for g in kb.ix.facts.by_key.get(group_key(f), []):
            if id(g) not in present:
                near.append((g, s))
                present.add(id(g))
    selected, notes = resolve_conflicts([f for f, _ in near])
    chosen_ids = {id(f) for f in selected}
    # keep the strongest few distinct (model, subject) answers
    out, seen = [], set()
    ranking = {id(f): s for f, s in near}
    for f in sorted(selected, key=lambda f: -ranking.get(id(f), 0)):
        key = (f.model, f.subject.lower(), f.attribute)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= top_n:
            break
    losers = [f for f, _ in near if id(f) not in chosen_ids and not f.previous]
    return SpecResult(selected=out, conflicts=notes, superseded=losers, scored=scored[:10])


# ====================================================================================================== premise check
@dataclass
class PremiseResult:
    verdict: str  # confirmed | refuted | superseded | unknown
    claimed: Optional[Quantity] = None
    fact: Optional[Fact] = None
    note: str = ""


def check_premise(a: QueryAnalysis, spec: SpecResult) -> Optional[PremiseResult]:
    if not a.yes_no or not a.premise or not spec.selected:
        return None
    claim = a.premise[0]
    fact = spec.selected[0]
    if claim.unit and fact.unit and U.unit_dimension(claim.unit) != U.unit_dimension(fact.unit):
        return PremiseResult("unknown", claim, fact, "the stated quantity has a different dimension than the documented value")
    actual = Quantity(value=fact.value, unit=fact.unit)
    if U.quantities_match(claim, actual, rel_tol=0.003) if claim.unit and fact.unit else abs(claim.value - fact.value) < 1e-9:
        return PremiseResult("confirmed", claim, fact)
    for old in spec.superseded:
        if (old.unit and claim.unit and U.quantities_match(claim, Quantity(value=old.value, unit=old.unit), rel_tol=0.003)) or (not claim.unit and abs(old.value - claim.value) < 1e-9):
            return PremiseResult("superseded", claim, fact, f"{U.format_quantity(old.value, old.unit)} comes from {old.doc_title}")
    return PremiseResult("refuted", claim, fact)


# ====================================================================================================== unit conversion
_CONVERT_RE = re.compile(r"\b(?:in|to|into)\s+(lb[-\s]?ft|lb[-\s]?in|nm|n·m|psi|bar|mpa|kpa|kw|hp|w|gpm|l/min|m3/h|m³/h|kg|lb|lbs|°?f|°?c|mm|in|ft|l|gal)\b", re.I)


def requested_unit(question: str) -> Optional[str]:
    m = _CONVERT_RE.search(U.normalize_unit_text(question))
    return U.canonical_unit(m.group(1)) if m else None


def convert_fact(f: Fact, target: Optional[str]) -> Optional[str]:
    if not target or not f.unit or target == f.unit:
        return None
    try:
        v = U.convert(f.value, f.unit, target)
    except ValueError:
        return None
    return U.format_quantity(round(v, 2), target)


# ====================================================================================================== chart reading
@dataclass
class ChartRead:
    chunk: Chunk
    x_label: str
    x_value: Optional[float]
    y: float
    y_title: str
    y_unit: str
    interpolated: bool
    figure_no: str
    caption: str
    reliability: float

    def display(self) -> str:
        return f"{self.y:.1f}" + (f" {self.y_unit}" if self.y_unit else "")


def _unit_in_parens(title: str) -> str:
    m = re.search(r"\(([^)]+)\)", title or "")
    return m.group(1).strip() if m else ""


def read_chart_values(kb: KnowledgeBase, a: QueryAnalysis, chunks: list[Chunk]) -> list[ChartRead]:
    reads: list[ChartRead] = []
    qtok = set(a.tokens)
    for ch in chunks:
        chart = ch.meta.get("chart")
        if ch.modality != Modality.CHART or not chart or not chart.get("series"):
            continue
        title_tokens = set(content_tokens(" ".join([chart.get("y_title", ""), chart.get("x_title", ""), chart.get("title", ""), ch.meta.get("caption", "")])))
        y_tokens = set(content_tokens(chart.get("y_title", "")))
        if not (qtok & title_tokens):
            continue
        pts = chart["series"][0]["points"]
        x_unit = U.canonical_unit(_unit_in_parens(chart.get("x_title", "")) or chart.get("x_unit", ""))
        y_title_raw = re.sub(r"\s*\([^)]*\)", "", chart.get("y_title", "")).strip()
        y_unit = _unit_in_parens(chart.get("y_title", "")) or chart.get("y_unit", "")
        categorical = isinstance(pts[0][0], str)
        # candidate x values from the question
        qs = [q for q in a.quantities]
        chosen: Optional[Quantity] = None
        for q in qs:
            if x_unit and q.unit == x_unit:
                chosen = q
                break
        if chosen is None:
            for q in qs:
                if categorical and q.unit == "%":
                    chosen = q
                    break
        if chosen is None:  # an unlabelled number ("at 100") is accepted only when no quantity has a conflicting unit
            loose = [q for q in qs if q.unit is None]
            chosen = loose[0] if loose else None
        if chosen is None:
            continue
        rel = min(1.0, float(chart.get("confidence", 0.7)))
        if categorical:
            want = f"{U.format_number(chosen.value)}{'%' if chosen.unit == '%' else ''}"
            for label, y in pts:
                if str(label).replace(" ", "") == want or str(label).rstrip("%") == U.format_number(chosen.value):
                    reads.append(ChartRead(ch, str(label), chosen.value, float(y), y_title_raw, y_unit, False, str(ch.meta.get("figure_no", "")), ch.meta.get("caption", ""), rel))
                    break
        else:
            y = interpolate_series(pts, chosen.value)
            if y is not None:
                exact = any(abs(float(p[0]) - chosen.value) < 1e-6 for p in pts)
                reads.append(ChartRead(ch, U.format_number(chosen.value), chosen.value, float(y), y_title_raw, y_unit, not exact, str(ch.meta.get("figure_no", "")),
                                       ch.meta.get("caption", ""), rel * (1.0 if exact else 0.9)))
    return reads


# ====================================================================================================== log analytics
@dataclass
class AnalyticsResult:
    ok: bool
    op: str = ""
    value: Optional[float] = None
    label: str = ""
    text: str = ""
    code: str = ""
    dataset: str = ""
    rows_used: int = 0
    rows_total: int = 0
    table: list[dict] = field(default_factory=list)
    reason: str = ""


def _norm_val(v: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


_GROUP_HINTS = {"error_code": ("alarm", "error", "code", "fault"), "unit_id": ("unit", "machine", "pump", "compressor"), "technician": ("technician", "engineer", "who"),
                "model": ("model", "product"), "action_taken": ("action", "fix", "remedy")}


def run_analytics(kb: KnowledgeBase, a: QueryAnalysis) -> AnalyticsResult:
    datasets = kb.datasets()
    if not datasets:
        return AnalyticsResult(False, reason="No tabular dataset (CSV/XLSX) has been ingested.")
    q = a.question.lower()
    best: Optional[tuple[int, pd.DataFrame, object]] = None
    for d in datasets:
        df = kb.dataframe(d.id)
        if df is None or df.empty:
            continue
        score = sum(1 for c in df.columns if any(t in q for t in re.split(r"_", c) if len(t) > 3))
        score += sum(1 for c in df.select_dtypes(exclude="number").columns for v in df[c].astype(str).unique()[:60] if _norm_val(v) and _norm_val(v) in _norm_val(q))
        if best is None or score > best[0]:
            best = (score, df, d)
    if best is None:
        return AnalyticsResult(False, reason="The dataset could not be loaded.")
    _, df, doc = best
    total = len(df)
    code_parts, mask = [], pd.Series(True, index=df.index)
    nq = _norm_val(q)
    filters: list[str] = []
    for c in df.select_dtypes(exclude="number").columns:
        if c.endswith("date"):
            continue
        vals = [v for v in df[c].dropna().astype(str).unique() if len(_norm_val(v)) >= 3]
        hit = [v for v in vals if _norm_val(v) in nq]
        # prefer the longest match (HX200-01 over HX200)
        hit = [v for v in hit if not any(v != w and _norm_val(v) in _norm_val(w) and _norm_val(w) in nq for w in hit)]
        if hit and len(hit) < len(vals):
            mask &= df[c].astype(str).isin(hit)
            filters.append(f"{c} in {hit}")
            code_parts.append(f"{c} in {hit}")
    sub = df[mask]
    filt_text = (" for " + " and ".join(f"{c.replace('_', ' ')} {', '.join(map(str, h))}" for c, h in
                                       [(f.split(" in ")[0], eval_list(f.split(" in ", 1)[1])) for f in filters])) if filters else ""
    num_cols = list(df.select_dtypes(include="number").columns)
    ds = doc.title
    code_filter = " & ".join(f"(df.{c.split(' in ')[0]}.isin({c.split(' in ', 1)[1]}))" for c in filters) or "True"
    # --- operation
    if re.search(r"\bmost (often|common|frequent|frequently)\b|\boccurs? most\b|\bhighest (count|number)\b|\bmost\b.*\b(alarm|error|code)\b", q):
        gcol = next((c for c, hints in _GROUP_HINTS.items() if c in df.columns and any(h in q for h in hints)), None) or ("error_code" if "error_code" in df.columns else None)
        if gcol is None:
            return AnalyticsResult(False, reason="Could not decide which column to rank.")
        vc = sub[gcol].value_counts()
        if vc.empty:
            return AnalyticsResult(False, reason="No rows match the filters.")
        top, n = vc.index[0], int(vc.iloc[0])
        tie = [str(i) for i, v in vc.items() if int(v) == n]
        text = f"The most frequent {gcol.replace('_', ' ')}{filt_text} is {top} with {n} of {len(sub)} records" + (f" (tied with {', '.join(t for t in tie if t != str(top))})" if len(tie) > 1 else "") + f" in {ds}."
        return AnalyticsResult(True, "argmax", float(n), str(top), text, f"df[{code_filter}].{gcol}.value_counts().head(5)", ds, len(sub), total,
                               [{gcol: str(i), "count": int(v)} for i, v in vc.head(5).items()])
    ncol = next((c for c in num_cols if any(t in q for t in c.split("_") if len(t) > 3)), None)
    if re.search(r"\baverage\b|\bmean\b", q) and ncol:
        if sub.empty:
            return AnalyticsResult(False, reason="No rows match the filters.")
        v = float(sub[ncol].mean())
        text = f"The average {ncol.replace('_', ' ')}{filt_text} is {v:.2f} (over {len(sub)} records) in {ds}."
        return AnalyticsResult(True, "mean", round(v, 4), ncol, text, f"df[{code_filter}].{ncol}.mean()", ds, len(sub), total)
    if re.search(r"\btotal\b|\bsum\b", q) and ncol:
        v = float(sub[ncol].sum())
        text = f"The total {ncol.replace('_', ' ')}{filt_text} is {U.format_number(v)} (over {len(sub)} records) in {ds}."
        return AnalyticsResult(True, "sum", round(v, 4), ncol, text, f"df[{code_filter}].{ncol}.sum()", ds, len(sub), total)
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", q):
        text = f"There are {len(sub)} matching records{filt_text} in {ds}."
        return AnalyticsResult(True, "count", float(len(sub)), "rows", text, f"len(df[{code_filter}])", ds, len(sub), total)
    return AnalyticsResult(False, reason="The question does not map to a supported log operation (count, sum, average, most frequent).")


def eval_list(s: str) -> list[str]:
    """Parse the repr of a list of plain strings that WE generated (never user text)."""
    return re.findall(r"'([^']*)'", s)
