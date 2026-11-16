"""Evidence assembly: select the final chunks, follow cross-references, and attach safety warnings.

Cross-reference following (multi-hop):  "tighten the gland bolts to the torque listed in Table 2"  ->  the *row* of Table 2 that
the sentence talks about (not the whole table), pulled in even though the question never mentions it.

Safety-critical expansion:  a retrieved procedure always travels with the WARNING/CAUTION call-outs of its section and the
lock-out/tag-out notice of the document, so a maintenance answer can never silently drop the safety precondition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..kb import KnowledgeBase
from ..schemas import Chunk, Modality
from ..utils.text import split_sentences, tokenize
from .analyzer import Intent, QueryAnalysis
from .pipeline import Candidate, RetrievalResult


@dataclass
class EvidenceItem:
    chunk: Chunk
    via: str = "retrieval"  # retrieval | crossref | warning
    score: float = 0.0
    note: str = ""
    channels: list = field(default_factory=list)


@dataclass
class ContextConfig:
    k: int = 6
    follow_refs: bool = True
    add_warnings: bool = True
    min_rel: float = 0.45  # keep candidates scoring at least this fraction of the best
    max_added_refs: int = 3
    max_added_warnings: int = 2


def _sentences_with_ref(text: str, ref_type: str, ref_no: str) -> list[str]:
    pat = re.compile(rf"\b{ref_type}s?\s+{re.escape(ref_no)}\b", re.I)
    return [s for s in split_sentences(text) if pat.search(s)]


def _row_subject_tokens(ch: Chunk) -> set[str]:
    return set(tokenize(str(ch.meta.get("subject", ""))))


def select_evidence(res: RetrievalResult, kb: KnowledgeBase, cfg: Optional[ContextConfig] = None) -> list[EvidenceItem]:
    cfg = cfg or ContextConfig()
    a = res.analysis
    cands = res.candidates
    if not cands:
        return []
    top = cands[0].final or 1e-9
    chosen: list[EvidenceItem] = []
    seen: set[str] = set()
    for c in cands:
        if len(chosen) >= cfg.k:
            break
        if c.final < cfg.min_rel * top:
            continue
        ch = c.chunk
        if ch.id in seen:
            continue
        # prefer the precise row over its parent table (unless the user wants the table/figure itself)
        if ch.modality == Modality.TABLE and a.intent not in (Intent.FIGURE, Intent.ANALYTICS):
            rows_in = [x for x in chosen if x.chunk.meta.get("parent_table") == ch.id]
            if rows_in:
                continue
        if ch.modality == Modality.TABLE_ROW:
            parent = ch.meta.get("parent_table")
            chosen = [x for x in chosen if x.chunk.id != parent] if parent else chosen
        chosen.append(EvidenceItem(ch, "retrieval", c.final, channels=c.channels()))
        seen.add(ch.id)

    added = 0
    if cfg.follow_refs:
        q_tokens = set(a.tokens)
        for item in list(chosen[:3]):
            ch = item.chunk
            if ch.modality not in (Modality.TEXT, Modality.TABLE_ROW, Modality.OCR_PAGE, Modality.WARNING):
                continue
            targets = kb.ix.graph.refs.get(ch.id, [])
            if not targets:
                continue
            by_label: dict[str, list[Chunk]] = {}
            for tid, label in targets:
                t = kb.chunk(tid)
                if t is not None:
                    by_label.setdefault(label, []).append(t)
            for label, tgts in by_label.items():
                if added >= cfg.max_added_refs:
                    break
                ref_type, ref_no = label.split(" ", 1)
                if ref_type == "table":
                    rows = [t for t in tgts if t.modality == Modality.TABLE_ROW]
                    sents = _sentences_with_ref(ch.content, "table", ref_no)
                    if not rows or not sents:
                        continue
                    best, best_score = None, 0.0
                    for r in rows:
                        subj = _row_subject_tokens(r)
                        if not subj:
                            continue
                        s_overlap = max(len(subj & set(tokenize(s))) / len(subj) for s in sents)
                        q_overlap = len(subj & q_tokens) / len(subj)
                        score = s_overlap + 0.5 * q_overlap
                        if score > best_score:
                            best, best_score = r, score
                    if best is not None and best_score >= 0.9 and best.id not in seen:
                        chosen.append(EvidenceItem(best, "crossref", best_score, note=f"followed '{label}' from [{ch.id[-4:]}]"))
                        seen.add(best.id)
                        added += 1
                elif ref_type == "figure":
                    tg = [t for t in tgts if t.modality in (Modality.FIGURE, Modality.CHART)]
                    if tg and tg[0].id not in seen and a.intent in (Intent.FIGURE, Intent.SPEC):
                        chosen.append(EvidenceItem(tg[0], "crossref", 0.5, note=f"followed '{label}'"))
                        seen.add(tg[0].id)
                        added += 1
                # section refs ("steps 1 to 4 of Section 6.1") are navigational; the section chunk is reachable through the graph channel

    if cfg.add_warnings:
        step_items = [x for x in chosen if x.chunk.modality == Modality.TEXT and x.chunk.meta.get("block") == "steps"]
        if step_items or a.intent == Intent.PROCEDURE:
            warn_added = 0
            for item in step_items or chosen[:2]:
                ch = item.chunk
                for w in kb.doc_chunks(ch.doc_id):
                    if w.modality == Modality.WARNING and w.id not in seen and (w.section_no == ch.section_no) and warn_added < cfg.max_added_warnings:
                        chosen.append(EvidenceItem(w, "warning", 0.0, note="safety call-out in the same section"))
                        seen.add(w.id)
                        warn_added += 1
            # always carry the lock-out notice of the document being serviced
            docs_used = {x.chunk.doc_id for x in chosen if x.chunk.modality in (Modality.TEXT, Modality.TABLE_ROW)}
            for doc_id in list(docs_used)[:2]:
                if warn_added >= cfg.max_added_warnings + 1:
                    break
                for w in kb.doc_chunks(doc_id):
                    if w.modality == Modality.WARNING and w.id not in seen and re.search(r"lock\s*out", w.content, re.I):
                        chosen.append(EvidenceItem(w, "warning", 0.0, note="document-level lock-out/tag-out notice"))
                        seen.add(w.id)
                        warn_added += 1
                        break
    return chosen
