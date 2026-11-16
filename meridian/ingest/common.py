"""Shared ingestion helpers: entity/cross-reference extraction, table->facts mining and the chunk builder."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..schemas import BBox, Chunk, Document, Fact, Modality
from ..utils import units as U
from ..utils.text import normalize_text, word_count

_ENTITY_RE = re.compile(r"\b[A-Z]{1,4}-\d{2,5}(?:-\d{1,3})?[A-Z]?\b")
_MODEL_RE = re.compile(r"\b[A-Z]{1,4}-\d{3,4}\b")
_REF_RES = [
    ("table", re.compile(r"\bTables?\s+(\d+)", re.I)),
    ("section", re.compile(r"\bSections?\s+(\d+(?:\.\d+)*)", re.I)),
    ("figure", re.compile(r"\bFig(?:ure)?s?\.?\s+(\d+)", re.I)),
]
_PREV_WORDS = re.compile(r"\b(previous|old|former|superseded|original|prior)\b", re.I)
_NEW_WORDS = re.compile(r"\b(revised|new|updated|current|amended)\b", re.I)


def extract_entities(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _ENTITY_RE.finditer(normalize_text(text)):
        seen.setdefault(m.group(0).upper(), None)
    return list(seen)


def extract_models(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _MODEL_RE.finditer(normalize_text(text)):
        tok = m.group(0).upper()
        if tok.startswith(("SB-", "PT-", "MS-")) or re.match(r"E-\d", tok):
            continue
        seen.setdefault(tok, None)
    return list(seen)


def extract_refs(text: str, skip_self: Optional[tuple[str, str]] = None) -> list[dict]:
    out, seen = [], set()
    for kind, rx in _REF_RES:
        for m in rx.finditer(text):
            key = (kind, m.group(1))
            if key in seen or key == skip_self:
                continue
            seen.add(key)
            out.append({"type": kind, "no": m.group(1)})
    return out


def parse_caption(line: str) -> Optional[tuple[str, str, str]]:
    """'Table 2. Foo' -> ('table', '2', 'Foo');  'Figure 1. Bar' -> ('figure', '1', 'Bar')."""
    m = re.match(r"^\s*(Table|Figure|Fig\.)\s+(\d+)\s*[.:\-–]?\s*(.*)$", line.strip(), re.I)
    if not m:
        return None
    kind = "table" if m.group(1).lower() == "table" else "figure"
    return kind, m.group(2), m.group(3).strip()


# ------------------------------------------------------------------------------------------------ table facts
def _clean_cell(c: Any) -> str:
    return re.sub(r"\s+", " ", str(c or "").replace("\n", " ")).strip()


def _header_unit(header: str) -> tuple[str, Optional[str]]:
    m = re.search(r"\(([^)]+)\)\s*$", header)
    if m:
        unit = U.canonical_unit(m.group(1))
        if unit:
            return header[: m.start()].strip(), unit
    return header, None


def _norm_attr(header: str) -> str:
    h = _PREV_WORDS.sub("", header)
    h = _NEW_WORDS.sub("", h)
    return re.sub(r"\s+", " ", h).strip().lower() or header.lower()


def table_to_records(header: list[str], rows: list[list[str]]) -> list[dict]:
    """Normalise a table into row records and numeric facts. Pure function (easy to unit-test)."""
    header = [_clean_cell(h) for h in header]
    unit_col = next((i for i, h in enumerate(header) if h.lower() in {"unit", "units", "uom"}), None)
    value_cols = [i for i in range(1, len(header)) if i != unit_col]
    records = []
    for row in rows:
        row = [_clean_cell(c) for c in row] + [""] * (len(header) - len(row))
        subject = row[0]
        if not subject:
            continue
        cells, facts = [], []
        for j in value_cols:
            h, val = header[j], row[j]
            if not val:
                continue
            base_h, hunit = _header_unit(h)
            unit = hunit
            if unit_col is not None and row[unit_col] not in ("", "-"):
                unit = U.canonical_unit(row[unit_col]) or unit
            qs = U.extract_quantities(val, include_unitless=True)
            shown = val
            if hunit and re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", val.strip()):
                shown = f"{val} {U.display_unit(hunit)}"
            elif unit_col is not None and row[unit_col] not in ("", "-") and re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", val.strip()):
                shown = f"{val} {row[unit_col]}"
            label = base_h if not (unit_col is not None and h.lower() == "value") else "value"
            cells.append((label, shown))
            if len(qs) == 1 and qs[0].start <= 1 and re.fullmatch(r"[\s\-+]*[\d.,]+\s*[^\d\s]{0,8}(?:/[^\d\s]{1,6})?\s*", val):
                q = qs[0]
                u = q.unit or unit
                facts.append({
                    "attribute": _norm_attr(base_h) if h.lower() != "value" else "value", "value": q.value, "unit": u,
                    "display": U.format_quantity(q.value, u), "previous": bool(_PREV_WORDS.search(h)), "revised": bool(_NEW_WORDS.search(h)),
                })
        records.append({"subject": subject, "cells": cells, "facts": facts})
    return records


def record_text(subject: str, cells: list[tuple[str, str]]) -> str:
    inner = "; ".join(f"{k} {v}" if k.lower() != "value" else v for k, v in cells)
    return f"{subject}: {inner}" if inner else subject


def mine_kv_facts(text: str) -> list[dict]:
    """'Maximum starts per hour: 6' style lines (scanned cards, key-value sheets)."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 /()\-]{3,60}?)\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        subj, val = m.group(1).strip(), m.group(2)
        qs = U.extract_quantities(val)
        if len(qs) == 1:
            q = qs[0]
            out.append({"subject": subj, "attribute": "value", "value": q.value, "unit": q.unit, "display": U.format_quantity(q.value, q.unit)})
    return out


# ------------------------------------------------------------------------------------------------ chunk builder
@dataclass
class ChunkBuilder:
    doc: Document
    chunks: list[Chunk] = field(default_factory=list)
    _n: int = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"{self.doc.id}:{self._n:04d}"

    def add(self, *, modality: Modality, content: str, page: int = 1, section: str = "", section_no: str = "", bbox: Optional[BBox] = None,
            meta: Optional[dict] = None, extra_text: str = "", image_path: Optional[str] = None, visual_vec: Optional[np.ndarray] = None,
            context: str = "", skip_self_ref: Optional[tuple[str, str]] = None) -> Chunk:
        content = content.strip()
        meta = dict(meta or {})
        ctx = context or " | ".join(x for x in [self.doc.title, section] if x)
        text = f"{ctx}\n{content}" + (f"\n{extra_text}" if extra_text else "")
        refs = extract_refs(content, skip_self=skip_self_ref)
        if refs:
            meta["refs"] = refs
        ch = Chunk(
            id=self._next_id(), doc_id=self.doc.id, page=page, modality=modality, text=text, content=content, section=section,
            section_no=section_no, parent_id=f"{self.doc.id}:sec:{section_no or section or 'root'}", bbox=bbox, meta=meta,
            entities=extract_entities(content + " " + section), image_path=image_path,
            visual_vec=visual_vec.astype(float).tolist() if visual_vec is not None else None,
        )
        self.chunks.append(ch)
        return ch

    def add_facts(self, chunk: Chunk, facts: list[dict], default_model: str = "") -> None:
        models = extract_models(chunk.content) or ([default_model] if default_model else (self.doc.models[:1] if len(self.doc.models) == 1 else []))
        out = []
        for f in facts:
            fact = Fact(
                model=f.get("model") or (models[0] if models else ""), subject=f["subject"], attribute=f.get("attribute", "value"),
                value=float(f["value"]), unit=f.get("unit"), display=f.get("display", ""), chunk_id=chunk.id, doc_id=self.doc.id,
                doc_type=self.doc.doc_type, doc_title=self.doc.title, effective_date=self.doc.effective_date,
                approximate=bool(f.get("approximate", False)), revised=bool(f.get("revised", False)), previous=bool(f.get("previous", False)),
            )
            out.append(fact.model_dump())
        if out:
            chunk.meta.setdefault("facts", []).extend(out)


def split_by_words(blocks: list[tuple[str, str]], target: int, max_words: int) -> list[list[tuple[str, str]]]:
    """Greedy packing of (kind, text) blocks into groups of ~target words. Steps lists are split by item when over max."""
    groups: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_w = 0
    expanded: list[tuple[str, str]] = []
    for kind, text in blocks:
        if word_count(text) > max_words and kind == "steps":
            items = [t for t in text.split("\n") if t.strip()]
            buf: list[str] = []
            for it in items:
                if buf and word_count("\n".join(buf + [it])) > max_words:
                    expanded.append((kind, "\n".join(buf)))
                    buf = []
                buf.append(it)
            if buf:
                expanded.append((kind, "\n".join(buf)))
        else:
            expanded.append((kind, text))
    for kind, text in expanded:
        w = word_count(text)
        if cur and (cur_w + w > max_words or (cur_w >= target)):
            groups.append(cur)
            cur, cur_w = [], 0
        cur.append((kind, text))
        cur_w += w
    if cur:
        groups.append(cur)
    return groups
