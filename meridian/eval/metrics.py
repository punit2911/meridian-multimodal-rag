"""Evaluation metrics: retrieval (Recall/MRR/nDCG/Hit), answer correctness (unit-aware numeric + text), calibration (ECE, Brier)."""
from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

from ..schemas import Chunk, Quantity
from ..utils import units as U
from ..utils.text import normalize_for_match


# ------------------------------------------------------------------------------------------------ relevance
def is_relevant(chunk: Chunk, ev: dict) -> bool:
    if chunk.doc_id != ev["doc"]:
        return False
    hay = (chunk.content + " " + chunk.section).lower()
    return all(c in hay for c in ev["contains"])


def relevant_chunks(chunks: Sequence[Chunk], gold_evidence: list[dict]) -> list[Chunk]:
    return [c for c in chunks if any(is_relevant(c, ev) for ev in gold_evidence)]


def positives(chunks: Sequence[Chunk], gold_evidence: list[dict]) -> set[str]:
    """Most-specific relevant chunks: a table *row* beats the whole table that also contains the words."""
    pos: set[str] = set()
    for ev in gold_evidence:
        rel = [c for c in chunks if is_relevant(c, ev)]
        if not rel:
            continue
        shortest = min(len(c.content) for c in rel)
        pos |= {c.id for c in rel if len(c.content) <= 1.6 * shortest + 40}
    return pos


# ------------------------------------------------------------------------------------------------ ranking metrics
def hit_at_k(ranked: Sequence[Chunk], gold_evidence: list[dict], k: int) -> float:
    """Evidence hit: every gold evidence item is covered within the top-k (items are alternatives for multi-hop => all required)."""
    if not gold_evidence:
        return float("nan")
    top = ranked[:k]
    return float(all(any(is_relevant(c, ev) for c in top) for ev in gold_evidence))


def any_hit_at_k(ranked: Sequence[Chunk], gold_evidence: list[dict], k: int) -> float:
    if not gold_evidence:
        return float("nan")
    return float(any(is_relevant(c, ev) for c in ranked[:k] for ev in gold_evidence))


def recall_at_k(ranked: Sequence[Chunk], gold_evidence: list[dict], k: int) -> float:
    if not gold_evidence:
        return float("nan")
    top = ranked[:k]
    got = sum(1 for ev in gold_evidence if any(is_relevant(c, ev) for c in top))
    return got / len(gold_evidence)


def mrr(ranked: Sequence[Chunk], gold_evidence: list[dict], pos_ids: Optional[set[str]] = None) -> float:
    if not gold_evidence:
        return float("nan")
    for i, c in enumerate(ranked, 1):
        if (c.id in pos_ids) if pos_ids is not None else any(is_relevant(c, ev) for ev in gold_evidence):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Chunk], gold_evidence: list[dict], k: int, pos_ids: Optional[set[str]] = None) -> float:
    if not gold_evidence:
        return float("nan")
    gains = []
    for c in ranked[:k]:
        if pos_ids is not None and c.id in pos_ids:
            gains.append(1.0)
        elif any(is_relevant(c, ev) for ev in gold_evidence):
            gains.append(0.5)  # relevant but less specific (e.g. the parent table)
        else:
            gains.append(0.0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_pos = max(1, len(pos_ids) if pos_ids is not None else len(gold_evidence))
    n_half = sum(1 for g in gains if g == 0.5)
    ideal_gains = ([1.0] * n_pos + [0.5] * n_half)[:k]  # best possible ordering of the same relevant material
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / ideal if ideal else 0.0


# ------------------------------------------------------------------------------------------------ answer correctness
def _num_match(answer_qs: list[Quantity], fact: dict) -> bool:
    tol = float(fact.get("tol", 0.01))
    raw_unit = fact.get("unit")
    gold = Quantity(value=float(fact["value"]), unit=U.canonical_unit(raw_unit) or raw_unit)
    for q in answer_qs:
        if gold.unit and q.unit:
            if U.quantities_match(q, gold, rel_tol=tol):
                return True
        else:
            if abs(q.value - gold.value) <= max(1e-9, tol * max(abs(gold.value), 1.0)):
                return True
    return False


def answer_correct(answer: str, answer_facts: list[dict]) -> tuple[bool, list[str]]:
    """All gold facts must be present in the answer. Numbers are compared unit-aware and with a relative tolerance."""
    text = answer or ""
    qs = U.extract_quantities(re.sub(r"\[\d+(?:\s*[,;]\s*\d+)*\]", " ", text))
    norm = normalize_for_match(text)
    missing: list[str] = []
    for f in answer_facts:
        if f["kind"] == "num":
            ok = _num_match(qs, f)
            if not ok:
                missing.append(f"{f['value']} {f.get('unit') or ''}".strip())
        else:
            ok = any(normalize_for_match(v) in norm for v in f["values"])
            if not ok:
                missing.append(" | ".join(f["values"]))
    return (not missing), missing


# ------------------------------------------------------------------------------------------------ calibration
def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / max(len(probs), 1)


def ece(probs: Sequence[float], outcomes: Sequence[int], bins: int = 5) -> tuple[float, list[dict]]:
    n = len(probs)
    if n == 0:
        return 0.0, []
    out, total = [], 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(outcomes[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(conf - acc)
        out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "confidence": round(conf, 3), "accuracy": round(acc, 3)})
    return total, out


def nanmean(xs: Iterable[float]) -> float:
    v = [x for x in xs if x == x]
    return sum(v) / len(v) if v else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 7) -> tuple[float, float]:
    import random

    v = [x for x in values if x == x]
    if not v:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]
