"""DOCX, tabular (CSV/XLSX) and plain-text parsers."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..schemas import Chunk, Document, Modality
from ..utils.text import split_sentences, word_count
from .common import (ChunkBuilder, extract_models, parse_caption, record_text, split_by_words, table_to_records)
from .pdf import HEADING_RE, WARN_RE, STEP_RE


# ====================================================================================================== DOCX
class DocxParser:
    def __init__(self, settings: Settings):
        self.s = settings

    def parse(self, path: Path, doc: Document) -> tuple[Document, list[Chunk]]:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        d = DocxDocument(str(path))
        pre = " ".join(p.text for p in d.paragraphs[:14])
        doc.models = extract_models(doc.filename + " " + (d.core_properties.title or "") + " " + pre)
        builder = ChunkBuilder(doc)
        stats = {"tables": 0, "table_rows": 0, "figures": 0, "charts": 0, "warnings": 0, "facts": 0}
        state = {"section": "", "section_no": ""}
        title = (d.core_properties.title or "").strip()
        full_text_parts: list[str] = []
        pending: list[tuple[str, str]] = []

        def flush():
            if not pending:
                return
            for g in split_by_words(pending, self.s.chunk_target_words, self.s.chunk_max_words):
                builder.add(modality=Modality.TEXT, content="\n".join(t for _, t in g), section=state["section"], section_no=state["section_no"])
            pending.clear()

        body = d.element.body
        last_para = ""
        for child in body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                p = Paragraph(child, d)
                text = p.text.strip()
                if not text:
                    continue
                style = (p.style.name or "").lower() if p.style is not None else ""
                full_text_parts.append(text)
                if style == "title":
                    title = title or text
                    continue
                if style.startswith("heading"):
                    flush()
                    m = HEADING_RE.match(text)
                    state["section"] = text
                    state["section_no"] = m.group(1) if m else ""
                    continue
                if WARN_RE.match(text):
                    flush()
                    sev = WARN_RE.match(text).group(1).upper()
                    builder.add(modality=Modality.WARNING, content=text, section=state["section"], section_no=state["section_no"], meta={"severity": sev})
                    stats["warnings"] += 1
                    continue
                pending.append(("steps" if STEP_RE.match(text) else "p", text))
                last_para = text
            elif tag == "tbl":
                flush()
                t = Table(child, d)
                rows = [[c.text.strip() for c in r.cells] for r in t.rows]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2:
                    continue
                stats["tables"] += 1
                tno = str(stats["tables"])
                header, body_rows = rows[0], rows[1:]
                records = table_to_records(header, body_rows)
                stats["table_rows"] += len(records)
                caption = re.sub(r"^\d+(?:\.\d+)*\s+", "", state["section"]).strip() or f"Table {tno}"
                md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in body_rows)
                meta = {"table_no": tno, "caption": caption, "header": header, "rows": len(body_rows)}
                tc = builder.add(modality=Modality.TABLE, content=f"Table {tno}: {caption}\n{md}", section=state["section"], section_no=state["section_no"], meta=dict(meta))
                for rec in records:
                    rc = builder.add(modality=Modality.TABLE_ROW, content=f"Table {tno} ({caption}), row: {record_text(rec['subject'], rec['cells'])}",
                                     section=state["section"], section_no=state["section_no"], meta={**meta, "subject": rec["subject"], "cells": [list(c) for c in rec["cells"]], "parent_table": tc.id})
                    builder.add_facts(rc, [{**f, "subject": rec["subject"]} for f in rec["facts"]])
        flush()

        full = "\n".join(full_text_parts)
        subject = re.search(r"Subject:\s*(.+)", full)
        base_title = title or Path(path).stem
        doc.title = f"{base_title}: {subject.group(1).strip()}" if subject and subject.group(1).strip() not in base_title else base_title
        date = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", full) or re.search(r"(\d{4}-\d{2}-\d{2})", full)
        doc.effective_date = date.group(1) if date else None
        if re.search(r"service bulletin|\bSB-\d{4}-\d+", full + " " + doc.title, re.I):
            doc.doc_type = "bulletin"
        sup = re.search(r"supersedes\s+the\s+(.+?)\s+in\s+Table\s+(\d+)\s+of\s+the\s+([A-Z]{1,4}-\d{3,4})\s+(.+?),\s*revision\s+([A-Z0-9]+)", full, re.I)
        if sup:
            doc.meta["supersedes"] = {"what": sup.group(1), "table": sup.group(2), "model": sup.group(3).upper(), "document": sup.group(4),
                                      "revision": sup.group(5)}
        applies = re.search(r"Applies to:\s*(.+)", full)
        if applies:
            doc.meta["applies_to"] = applies.group(1).strip()
        doc.models = extract_models(doc.title + " " + full[:600] + " " + doc.filename)
        doc.pages = 1
        for ch in builder.chunks:
            stats["facts"] += len(ch.meta.get("facts", []))
        # tag every chunk of a bulletin with revised/previous semantics already carried by facts; refresh context prefix with final title
        for ch in builder.chunks:
            if "\n" in ch.text:
                ch.text = " | ".join(x for x in [doc.title, ch.section] if x) + "\n" + ch.text.split("\n", 1)[1]
        doc.stats = stats
        return doc, builder.chunks


# ====================================================================================================== tabular
class TabularParser:
    """CSV / XLSX logs. A cleaned copy is stored for the analytics tool; the index gets a schema card and aggregate cards."""

    DATE_HINTS = ("date", "time", "timestamp", "day")

    def __init__(self, settings: Settings):
        self.s = settings

    def load(self, path: Path) -> pd.DataFrame:
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
        df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
        return df

    def parse(self, path: Path, doc: Document) -> tuple[Document, list[Chunk]]:
        df = self.load(path)
        if df.empty:
            raise ValueError("The table has no rows.")
        tables_dir = self.s.data_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        stored = tables_dir / f"{doc.id}.csv"
        df.to_csv(stored, index=False)
        builder = ChunkBuilder(doc)
        doc.doc_type = "log"
        doc.title = Path(path).stem.replace("_", " ")
        n = len(df)
        cols = list(df.columns)
        cat_cols = [c for c in cols if df[c].dtype == object and df[c].nunique() <= max(30, n // 4)]
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        date_col = next((c for c in cols if any(h in c for h in self.DATE_HINTS)), None)
        span = ""
        if date_col:
            dts = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(dts):
                span = f" covering {dts.min().date()} to {dts.max().date()}"
                doc.effective_date = str(dts.max().date())
        desc = [f"Tabular dataset '{doc.title}' with {n} rows{span}. Columns: " + ", ".join(cols) + "."]
        for c in cat_cols[:6]:
            vals = df[c].value_counts().head(12)
            desc.append(f"Column {c} distinct values: " + ", ".join(f"{k} ({v})" for k, v in vals.items()) + ".")
        for c in num_cols[:4]:
            desc.append(f"Column {c}: min {df[c].min():g}, max {df[c].max():g}, mean {df[c].mean():.2f}, sum {df[c].sum():g}.")
        builder.add(modality=Modality.TABLE, content="\n".join(desc), section="Dataset overview",
                    meta={"dataset": doc.id, "columns": cols, "rows": n, "kind": "schema"})
        # aggregate cards: group by (model-like column, code-like column) so natural-language questions retrieve them
        group_cols = [c for c in cat_cols if c in {"model", "error_code", "code", "alarm", "type", "category"}] or cat_cols[:2]
        group_cols = group_cols[:2]
        if group_cols:
            agg = df.groupby(group_cols, dropna=False)
            for key, g in agg:
                key = key if isinstance(key, tuple) else (key,)
                label = ", ".join(f"{c}={v}" for c, v in zip(group_cols, key))
                parts = [f"{len(g)} records"]
                for c in num_cols[:3]:
                    parts.append(f"{c} total {g[c].sum():g}, mean {g[c].mean():.2f}")
                act = [c for c in cat_cols if c not in group_cols and "action" in c]
                if act:
                    top = g[act[0]].value_counts().head(2)
                    parts.append(f"most common {act[0]}: " + "; ".join(top.index.astype(str)))
                builder.add(modality=Modality.TABLE_ROW, content=f"Log summary for {label}: " + ", ".join(parts) + ".", section="Aggregates",
                            meta={"dataset": doc.id, "group": dict(zip(group_cols, map(str, key))), "kind": "aggregate"})
        doc.models = sorted({m for c in cols if c == "model" for m in df[c].astype(str).unique() if re.match(r"[A-Z]{1,4}-\d{3,4}$", m)})
        doc.stats = {"rows": n, "columns": len(cols), "chunks": len(builder.chunks), "tables": 1, "facts": 0}
        doc.meta["dataset_path"] = str(stored.relative_to(self.s.data_dir))
        doc.meta["columns"] = cols
        return doc, builder.chunks


# ====================================================================================================== plain text / markdown
class TextParser:
    def __init__(self, settings: Settings):
        self.s = settings

    def parse(self, path: Path, doc: Document) -> tuple[Document, list[Chunk]]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        builder = ChunkBuilder(doc)
        state = {"section": "", "section_no": ""}
        pending: list[tuple[str, str]] = []

        def flush():
            if not pending:
                return
            for g in split_by_words(pending, self.s.chunk_target_words, self.s.chunk_max_words):
                builder.add(modality=Modality.TEXT, content="\n".join(t for _, t in g), section=state["section"], section_no=state["section_no"])
            pending.clear()

        first_heading = ""
        for para in re.split(r"\n\s*\n", raw):
            para = para.strip()
            if not para:
                continue
            m = re.match(r"^#{1,4}\s+(.*)$", para)
            if m and "\n" not in para:
                flush()
                state["section"] = m.group(1).strip()
                hm = HEADING_RE.match(state["section"])
                state["section_no"] = hm.group(1) if hm else ""
                first_heading = first_heading or state["section"]
                continue
            if WARN_RE.match(para):
                flush()
                builder.add(modality=Modality.WARNING, content=para, section=state["section"], section_no=state["section_no"],
                            meta={"severity": WARN_RE.match(para).group(1).upper()})
                continue
            pending.append(("p", re.sub(r"\s*\n\s*", " ", para)))
        flush()
        doc.title = first_heading or Path(path).stem.replace("_", " ")
        doc.models = extract_models(doc.title + " " + raw[:800])
        doc.doc_type = "notes"
        doc.stats = {"tables": 0, "facts": 0, "chunks": len(builder.chunks)}
        for ch in builder.chunks:
            if "\n" in ch.text:
                ch.text = " | ".join(x for x in [doc.title, ch.section] if x) + "\n" + ch.text.split("\n", 1)[1]
        return doc, builder.chunks
