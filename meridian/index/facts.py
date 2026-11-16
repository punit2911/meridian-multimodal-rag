"""Structured fact index + provenance-aware conflict resolution.

Facts are (model, subject, attribute, value, unit) assertions mined from table rows and digitised charts. The index answers
"what is the <attribute> of <subject> for <model>?" with exact numbers, and detects when several sources disagree.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

from ..schemas import ConflictNote, Fact
from ..utils import units as U
from ..utils.text import content_tokens


def fact_tokens(f: Fact) -> list[str]:
    return content_tokens(f"{f.subject} {f.attribute if f.attribute not in ('value', 'chart value') else ''}")


def group_key(f: Fact) -> tuple:
    dim = U.unit_dimension(f.unit)
    subj = tuple(sorted(set(content_tokens(f.subject))))
    attr = f.attribute if f.attribute not in ("value",) else ""
    return (f.model, subj, attr, dim)


class FactIndex:
    def __init__(self, facts: list[Fact]):
        self.facts = facts
        self.tokens = [set(fact_tokens(f)) for f in facts]
        df: Counter = Counter()
        for t in self.tokens:
            df.update(t)
        n = max(len(facts), 1)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.by_key: dict[tuple, list[Fact]] = defaultdict(list)  # same real-world quantity, any source
        for f in facts:
            if not f.approximate:
                self.by_key[group_key(f)].append(f)

    def search(self, query_tokens: Iterable[str], models: Optional[Iterable[str]] = None, unit_dim: Optional[str] = None,
               min_score: float = 0.5, include_approx: bool = True) -> list[tuple[Fact, float]]:
        q = set(query_tokens)
        wanted = {m.upper() for m in (models or [])}
        out: list[tuple[Fact, float]] = []
        for f, toks in zip(self.facts, self.tokens):
            if wanted and f.model and f.model.upper() not in wanted:
                continue
            if f.approximate and not include_approx:
                continue
            if unit_dim and f.unit and U.unit_dimension(f.unit) != unit_dim:
                continue
            if not toks:
                continue
            total = sum(self.idf.get(t, 1.0) for t in toks)
            matched = sum(self.idf.get(t, 1.0) for t in toks if t in q)
            if matched <= 0:
                continue
            coverage = matched / total  # how much of the fact's subject the question covers
            extra = sum(self.idf.get(t, 1.0) for t in q if t not in toks and t in self.idf)
            precision = matched / (matched + 0.15 * extra)
            score = coverage * (0.6 + 0.4 * precision)
            if score >= min_score:
                out.append((f, score))
        out.sort(key=lambda x: (-x[1], x[0].approximate))
        return out


def _same_value(a: Fact, b: Fact) -> bool:
    qa = U.Quantity(value=a.value, unit=a.unit) if hasattr(U, "Quantity") else None
    from ..schemas import Quantity

    return U.quantities_match(Quantity(value=a.value, unit=a.unit), Quantity(value=b.value, unit=b.unit), rel_tol=1e-3)


def _rank(f: Fact) -> tuple:
    return (1 if f.revised else 0, 1 if f.doc_type == "bulletin" else 0, f.effective_date or "0000-00-00")


def resolve_conflicts(facts: list[Fact]) -> tuple[list[Fact], list[ConflictNote]]:
    """Select the authoritative fact per group and explain every override.

    Rules (in order):
      1. A `previous`-column value never competes against a `revised`-column value of the same row.
      2. Among remaining candidates with different values, the most recent effective date wins; a service bulletin beats a
         manual with the same date. The loser is reported, never silently dropped.
    Approximate (chart-read) facts never take part: they are evidence, not specifications.
    """
    exact = [f for f in facts if not f.approximate]
    approx = [f for f in facts if f.approximate]
    groups: dict[tuple, list[Fact]] = defaultdict(list)
    for f in exact:
        groups[group_key(f)].append(f)
    selected: list[Fact] = []
    notes: list[ConflictNote] = []
    for key, members in groups.items():
        if any(m.revised for m in members):
            members = [m for m in members if not m.previous]
        distinct: list[Fact] = []
        for m in members:
            if not any(_same_value(m, d) for d in distinct):
                distinct.append(m)
        if len(distinct) == 1:
            best = max(members, key=_rank)
            selected.append(best)
            continue
        ranked = sorted(members, key=_rank, reverse=True)
        winner = ranked[0]
        selected.append(winner)
        for loser in ranked[1:]:
            if _same_value(winner, loser):
                continue
            rule = ("revised value in a newer service bulletin supersedes the manual"
                    if winner.doc_type == "bulletin" and loser.doc_type != "bulletin" else "most recent effective date wins")
            notes.append(ConflictNote(subject=winner.subject, model=winner.model, winner=winner.doc_title, loser=loser.doc_title,
                                      winner_value=winner.display or U.format_quantity(winner.value, winner.unit),
                                      loser_value=loser.display or U.format_quantity(loser.value, loser.unit), rule=rule,
                                      winner_chunk=winner.chunk_id, loser_chunk=loser.chunk_id))
            break  # one note per group is enough
    return selected + approx, notes
