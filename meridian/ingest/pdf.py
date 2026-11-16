"""Layout-aware PDF parser.

For every page it separates: tables (cells -> row facts), warning call-outs, figures/charts (cropped, classified,
digitised), body text (headings by font, paragraphs, numbered steps) and scanned pages (OCR with line boxes).
Every chunk keeps a bounding box so the UI can highlight the exact evidence region on the rendered page.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from ..config import Settings
from ..schemas import BBox, Chunk, Document, Modality
from ..utils.text import slugify, word_count
from . import figures as figmod
from . import ocr as ocrmod
from .common import (ChunkBuilder, extract_models, mine_kv_facts, parse_caption, record_text, split_by_words,
                     table_to_records)

WARN_RE = re.compile(r"^\s*(WARNING|CAUTION|DANGER|NOTICE|NOTE|ATTENTION)\b\s*[:!.\-]?", re.I)
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
STEP_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+\S")


def _bbox(x0, y0, x1, y1, page) -> BBox:
    return BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), page_width=float(page.width), page_height=float(page.height))


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _inside(cx: float, cy: float, box: tuple[float, float, float, float], pad: float = 1.5) -> bool:
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad


class _Line:
    __slots__ = ("text", "x0", "x1", "top", "bottom", "size", "bold")

    def __init__(self, words: list[dict]):
        words = sorted(words, key=lambda w: w["x0"])
        self.text = " ".join(w["text"] for w in words)
        self.x0 = min(w["x0"] for w in words)
        self.x1 = max(w["x1"] for w in words)
        self.top = min(w["top"] for w in words)
        self.bottom = max(w["bottom"] for w in words)
        self.size = max(w.get("size", 10.0) for w in words)
        self.bold = sum("bold" in str(w.get("fontname", "")).lower() for w in words) >= max(1, len(words) * 0.6)


def _group_lines(words: list[dict]) -> list[_Line]:
    words = sorted(words, key=lambda w: (round(w["top"], 0), w["x0"]))
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= 2.5:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [_Line(l) for l in lines]


def render_page(pdf: "pdfium.PdfDocument", index: int, dpi: float) -> Image.Image:
    page = pdf[index]
    bitmap = page.render(scale=dpi / 72.0)
    return bitmap.to_pil().convert("RGB")


class PDFParser:
    def __init__(self, settings: Settings):
        self.s = settings

    # ------------------------------------------------------------------------------------------ public
    def parse(self, path: Path, doc: Document) -> tuple[Document, list[Chunk]]:
        pages_dir = self.s.data_dir / "pages" / doc.id
        figs_dir = self.s.data_dir / "figures" / doc.id
        pages_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)
        builder = ChunkBuilder(doc)
        pdfdoc = pdfium.PdfDocument(str(path))
        stats = {"tables": 0, "table_rows": 0, "figures": 0, "charts": 0, "warnings": 0, "ocr_pages": 0, "facts": 0}
        state = {"section": "", "section_no": ""}
        first_page_lines: list[_Line] = []
        with pdfplumber.open(str(path)) as pl:
            doc.pages = len(pl.pages)
            try:
                first_text = pl.pages[0].extract_text() or ""
            except Exception:
                first_text = ""
            doc.models = extract_models(doc.filename + " " + first_text[:400])  # known before chunking so facts inherit the model
            body_sizes = Counter()
            for pg in pl.pages[:3]:
                for w in pg.extract_words(extra_attrs=["size"]):
                    body_sizes[round(w["size"], 1)] += 1
            body = body_sizes.most_common(1)[0][0] if body_sizes else 10.0
            for pno, page in enumerate(pl.pages, 1):
                try:
                    render = render_page(pdfdoc, pno - 1, self.s.render_dpi)
                    render.save(pages_dir / f"p{pno}.png", optimize=True)
                except Exception:
                    pass
                lines = self._page(page, pno, pdfdoc, doc, builder, state, body, figs_dir, stats)
                if pno == 1:
                    first_page_lines = lines
        self._doc_metadata(doc, first_page_lines, path)
        # title may have been refined: refresh contextual text prefix of already-built chunks
        for ch in builder.chunks:
            old_prefix = ch.text.split("\n", 1)[0]
            new_prefix = " | ".join(x for x in [doc.title, ch.section] if x)
            if old_prefix != new_prefix and "\n" in ch.text:
                ch.text = new_prefix + "\n" + ch.text.split("\n", 1)[1]
        for ch in builder.chunks:
            stats["facts"] += len(ch.meta.get("facts", []))
        if not doc.models:
            doc.models = extract_models(doc.title + " " + doc.filename)
        doc.stats = stats
        return doc, builder.chunks

    # ------------------------------------------------------------------------------------------ page
    def _page(self, page, pno: int, pdfdoc, doc: Document, builder: ChunkBuilder, state: dict, body: float, figs_dir: Path,
              stats: dict) -> list[_Line]:
        words = page.extract_words(extra_attrs=["fontname", "size"], keep_blank_chars=False, use_text_flow=False)
        img_boxes = [(i["x0"], i["top"], i["x1"], i["bottom"]) for i in page.images if (i["x1"] - i["x0"]) > 40 and (i["bottom"] - i["top"]) > 40]
        page_area = page.width * page.height
        full_page_image = any((b[2] - b[0]) * (b[3] - b[1]) > 0.55 * page_area for b in img_boxes)
        if full_page_image and len(words) < 15:
            return self._ocr_page(page, pno, pdfdoc, doc, builder, state, stats)

        # --- tables
        table_objs = []
        try:
            for t in page.find_tables():
                table_objs.append(t)
        except Exception:
            table_objs = []
        table_boxes = [tuple(t.bbox) for t in table_objs]
        excl = table_boxes + img_boxes

        body_words = [w for w in words if not any(_inside((w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2, b) for b in excl)]
        footer_y = page.height - 60
        footer_words = [w for w in body_words if w["top"] > footer_y]
        body_words = [w for w in body_words if w["top"] <= footer_y]
        lines = _group_lines(body_words)

        # captions + items sorted by vertical position
        items: list[dict] = []
        used_caption: set[int] = set()
        table_caps: dict[int, tuple[str, str]] = {}
        fig_caps: dict[int, tuple[str, str]] = {}
        for ti, tb in enumerate(table_boxes):
            cand = [(tb[1] - l.bottom, i) for i, l in enumerate(lines) if 0 <= tb[1] - l.bottom <= 36 and parse_caption(l.text) and parse_caption(l.text)[0] == "table"]
            if cand:
                _, li = min(cand)
                cap = parse_caption(lines[li].text)
                table_caps[ti] = (cap[1], lines[li].text.strip())
                used_caption.add(li)
        for ii, ib in enumerate(img_boxes):
            cand = [(l.top - ib[3], i) for i, l in enumerate(lines) if -2 <= l.top - ib[3] <= 60 and parse_caption(l.text) and parse_caption(l.text)[0] == "figure"]
            if cand:
                _, li = min(cand)
                cap = parse_caption(lines[li].text)
                fig_caps[ii] = (cap[1], lines[li].text.strip())
                used_caption.add(li)
                # a wrapped caption may continue on the following line(s)
                nxt = li + 1
                while nxt < len(lines) and lines[nxt].top - lines[nxt - 1].bottom < 3 and lines[nxt].size < body - 0.4 and not parse_caption(lines[nxt].text):
                    fig_caps[ii] = (cap[1], fig_caps[ii][1] + " " + lines[nxt].text.strip())
                    used_caption.add(nxt)
                    nxt += 1

        text_lines = [(i, l) for i, l in enumerate(lines) if i not in used_caption]
        for i, l in text_lines:
            items.append({"kind": "line", "top": l.top, "obj": l})
        for ti, t in enumerate(table_objs):
            items.append({"kind": "table", "top": t.bbox[1], "obj": t, "idx": ti})
        for ii, b in enumerate(img_boxes):
            items.append({"kind": "image", "top": b[1], "obj": b, "idx": ii})
        items.sort(key=lambda it: it["top"])

        pending: list[dict] = []  # text blocks awaiting chunking: {"kind","text","box"}
        cur_para: list[_Line] = []

        def flush_para():
            nonlocal cur_para
            if not cur_para:
                return
            text = " ".join(l.text for l in cur_para).strip()
            box = _union([(l.x0, l.top, l.x1, l.bottom) for l in cur_para])
            if text:
                kind = "warn" if WARN_RE.match(text) else "p"
                pending.append({"kind": kind, "text": text, "box": box})
            cur_para = []

        def flush_text():
            flush_para()
            self._emit_text(pending, builder, pno, page, state, stats)
            pending.clear()

        prev: Optional[_Line] = None
        for it in items:
            if it["kind"] == "line":
                l: _Line = it["obj"]
                m = HEADING_RE.match(l.text)
                is_heading = (l.bold and l.size >= body + 1.5) or (m is not None and l.bold and l.size >= body + 1.0)
                if l.size >= body + 8 and pno == 1:  # document title
                    flush_text()
                    prev = l
                    continue
                if is_heading:
                    flush_text()
                    hm = HEADING_RE.match(l.text)
                    state["section"] = l.text.strip()
                    state["section_no"] = hm.group(1) if hm else ""
                    prev = l
                    continue
                new_block = prev is None or (l.top - prev.bottom) > max(4.5, 0.5 * (prev.bottom - prev.top)) or STEP_RE.match(l.text) or WARN_RE.match(l.text)
                if cur_para and new_block:
                    flush_para()
                cur_para.append(l)
                prev = l
            elif it["kind"] == "table":
                flush_text()
                self._emit_table(it["obj"], it["idx"], table_caps.get(it["idx"]), builder, pno, page, state, stats, doc)
                prev = None
            else:
                flush_text()
                self._emit_figure(it["obj"], it["idx"], fig_caps.get(it["idx"]), builder, pno, page, pdfdoc, doc, state, figs_dir, stats)
                prev = None
        flush_text()
        return lines + _group_lines(footer_words)

    # ------------------------------------------------------------------------------------------ text
    def _emit_text(self, pending: list[dict], builder: ChunkBuilder, pno: int, page, state: dict, stats: dict) -> None:
        if not pending:
            return
        # warnings become their own chunks (safety-critical retrieval); steps merged into one block
        blocks: list[tuple[str, str, tuple]] = []
        step_buf: list[dict] = []

        def flush_steps():
            nonlocal step_buf
            if step_buf:
                blocks.append(("steps", "\n".join(b["text"] for b in step_buf), _union([b["box"] for b in step_buf])))
                step_buf = []

        for b in pending:
            if b["kind"] == "warn":
                flush_steps()
                self._emit_warning(b["text"], b["box"], builder, pno, page, state, stats)
            elif STEP_RE.match(b["text"]):
                step_buf.append(b)
            else:
                flush_steps()
                blocks.append(("p", b["text"], b["box"]))
        flush_steps()
        if not blocks:
            return
        boxes = {id(x): x[2] for x in blocks}
        groups = split_by_words([(k, t) for k, t, _ in blocks], self.s.chunk_target_words, self.s.chunk_max_words)
        idx = 0
        for g in groups:
            gboxes = []
            parts = []
            for kind, text in g:
                # map back to the original block (blocks appear in order; steps may be split by split_by_words)
                while idx < len(blocks) and blocks[idx][1] != text and not blocks[idx][1].startswith(text.split("\n")[0]):
                    idx += 1
                src = blocks[idx] if idx < len(blocks) else blocks[-1]
                gboxes.append(src[2])
                parts.append(text)
                if idx < len(blocks) and blocks[idx][1] == text:
                    idx += 1
            content = "\n".join(parts)
            kind = "steps" if any(k == "steps" for k, _ in g) else "text"
            box = _bbox(*_union(gboxes), page)
            builder.add(modality=Modality.TEXT, content=content, page=pno, section=state["section"], section_no=state["section_no"],
                        bbox=box, meta={"block": kind})

    def _emit_warning(self, text: str, box, builder: ChunkBuilder, pno: int, page, state: dict, stats: dict) -> None:
        m = WARN_RE.match(text)
        sev = m.group(1).upper() if m else "WARNING"
        builder.add(modality=Modality.WARNING, content=text.strip(), page=pno, section=state["section"], section_no=state["section_no"],
                    bbox=_bbox(*box, page), meta={"severity": sev})
        stats["warnings"] += 1

    # ------------------------------------------------------------------------------------------ tables
    def _emit_table(self, table, tidx: int, cap: Optional[tuple[str, str]], builder: ChunkBuilder, pno: int, page, state: dict,
                    stats: dict, doc: Document) -> None:
        try:
            raw = table.extract()
        except Exception:
            return
        rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in raw if r and any((c or "").strip() for c in r)]
        if not rows:
            return
        box = _bbox(*table.bbox, page)
        ncols = max(len(r) for r in rows)
        flat = " ".join(" ".join(r) for r in rows)
        if ncols == 1 and (WARN_RE.match(flat) or len(rows) >= 1):
            for r in rows:  # call-out boxes
                text = r[0].strip()
                if WARN_RE.match(text):
                    self._emit_warning(text, table.bbox, builder, pno, page, state, stats)
                else:
                    builder.add(modality=Modality.TEXT, content=text, page=pno, section=state["section"], section_no=state["section_no"], bbox=box)
            return
        header, body_rows = rows[0], rows[1:]
        if not body_rows:
            return
        tno, cap_text = cap if cap else (str(stats["tables"] + 1), f"Table {stats['tables'] + 1}")
        stats["tables"] += 1
        caption_title = re.sub(r"^\s*Table\s+\d+\s*[.:\-]?\s*", "", cap_text).strip()
        records = table_to_records(header, body_rows)
        stats['table_rows'] += len(records)
        md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in body_rows)
        content = f"{cap_text}\n{md}"
        base_meta = {"table_no": tno, "caption": caption_title, "header": header, "rows": len(body_rows)}
        tchunk = builder.add(modality=Modality.TABLE, content=content, page=pno, section=state["section"], section_no=state["section_no"], bbox=box,
                             meta=dict(base_meta), skip_self_ref=("table", tno))
        # row-level chunks (precise retrieval + fact mining)
        for rec in records:
            text = record_text(rec["subject"], rec["cells"])
            rc = builder.add(modality=Modality.TABLE_ROW, content=f"Table {tno} ({caption_title}), row: {text}", page=pno, section=state["section"],
                             section_no=state["section_no"], bbox=box, meta={**base_meta, "subject": rec["subject"], "cells": [list(c) for c in rec["cells"]], "parent_table": tchunk.id},
                             context=" | ".join(x for x in [doc.title, f"Table {tno} {caption_title}"] if x), skip_self_ref=("table", tno))
            builder.add_facts(rc, [{**f, "subject": rec["subject"]} for f in rec["facts"]])

    # ------------------------------------------------------------------------------------------ figures
    def _emit_figure(self, box, idx: int, cap: Optional[tuple[str, str]], builder: ChunkBuilder, pno: int, page, pdfdoc, doc: Document,
                     state: dict, figs_dir: Path, stats: dict) -> None:
        scale = 250.0 / 72.0
        try:
            pil = pdfdoc[pno - 1].render(scale=scale).to_pil().convert("RGB")
        except Exception:
            return
        pad = 2
        crop = pil.crop((max(0, int((box[0] - pad) * scale)), max(0, int((box[1] - pad) * scale)),
                         min(pil.width, int((box[2] + pad) * scale)), min(pil.height, int((box[3] + pad) * scale))))
        fno = cap[0] if cap else str(stats["figures"] + 1)
        cap_text = cap[1] if cap else f"Figure {fno}"
        stats["figures"] += 1
        path = figs_dir / f"p{pno}_f{fno}.png"
        crop.save(path, optimize=True)
        an = figmod.analyze_figure(crop, lang=self.s.ocr_lang)
        modality = Modality.CHART if (an.kind == "chart" and an.chart is not None) else Modality.FIGURE
        meta = {"figure_no": fno, "caption": cap_text, "kind": an.kind, "ocr": an.ocr_text}
        parts = [cap_text]
        extra = []
        if an.ocr_text:
            parts.append(f"Text visible in the figure: {an.ocr_text}")
            extra.append(an.ocr_text)
        if an.chart is not None:
            stats["charts"] += 1
            meta["chart"] = an.chart.to_meta()
            summary = an.chart.summary()
            parts.append(summary)
            extra.append(summary)
        content = "\n".join(parts)
        ch = builder.add(modality=modality, content=content, page=pno, section=state["section"], section_no=state["section_no"],
                         bbox=_bbox(*box, page), meta=meta, image_path=str(path.relative_to(self.s.data_dir)), visual_vec=an.descriptor,
                         skip_self_ref=("figure", fno))
        # chart points become approximate facts so numeric questions can use them
        if an.chart is not None and an.chart.series:
            title = an.chart.title or cap_text
            facts = []
            for x, y in an.chart.series[0].points:
                facts.append({"subject": f"{title} at {figmod._fmt(x)}", "attribute": "chart value", "value": float(y), "unit": None,
                              "display": f"{y:.1f}", "approximate": True})
            builder.add_facts(ch, facts, default_model=(extract_models(title + " " + cap_text) or [""])[0])

    # ------------------------------------------------------------------------------------------ scanned pages
    def _ocr_page(self, page, pno: int, pdfdoc, doc: Document, builder: ChunkBuilder, state: dict, stats: dict) -> list[_Line]:
        if not ocrmod.ocr_available():
            builder.add(modality=Modality.OCR_PAGE, content="[Scanned page: OCR engine unavailable]", page=pno, meta={"ocr_conf": 0.0})
            return []
        dpi = 200.0
        img = render_page(pdfdoc, pno - 1, dpi)
        lines = ocrmod.ocr_lines(img, lang=self.s.ocr_lang, psm=6, scale=1.5, deskew_img=True)
        stats["ocr_pages"] += 1
        if not lines:
            return []
        k = 72.0 / dpi
        # Group consecutive lines into paragraphs by vertical gap, then pack to chunk size.
        paras: list[list[ocrmod.OCRLine]] = []
        for l in lines:
            if paras and (l.y0 - paras[-1][-1].y1) < 0.9 * (paras[-1][-1].y1 - paras[-1][-1].y0) + 14:
                paras[-1].append(l)
            else:
                paras.append([l])
        title_line = lines[0].text
        for para in paras:
            text = "\n".join(l.text for l in para)
            if word_count(text) < 2:
                continue
            x0, y0 = min(l.x0 for l in para) * k, min(l.y0 for l in para) * k
            x1, y1 = max(l.x1 for l in para) * k, max(l.y1 for l in para) * k
            conf = sum(l.conf for l in para) / len(para)
            ch = builder.add(modality=Modality.OCR_PAGE, content=text, page=pno, section=state["section"] or title_line[:60],
                             bbox=_bbox(x0, y0, x1, y1, page), meta={"ocr_conf": round(conf, 1), "scanned": True})
            kv = mine_kv_facts(text)
            if kv:
                builder.add_facts(ch, kv, default_model=(extract_models(title_line + " " + text) or [""])[0])
        # derive a title from the first OCR line when the doc has none
        doc.meta["ocr_title"] = title_line
        return []

    # ------------------------------------------------------------------------------------------ metadata
    def _doc_metadata(self, doc: Document, first_lines: list[_Line], path: Path) -> None:
        text = "\n".join(l.text for l in first_lines)
        titles = [l.text for l in first_lines if l.size >= 20]
        rev = re.search(r"(?:Document\s+)?revision\s*:?\s*([A-Z0-9]{1,3})\b", text, re.I)
        date = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if titles:
            doc.title = " ".join(titles).strip()
            if re.search(r"service manual", text, re.I) and "manual" not in doc.title.lower():
                doc.title += " Service Manual"
            doc.doc_type = "manual" if re.search(r"manual|handbook|guide", text, re.I) else doc.doc_type
        elif doc.meta.get("ocr_title"):
            raw = str(doc.meta["ocr_title"])
            doc.title = " ".join(w if (any(c.isdigit() for c in w) or not w.isupper()) else w.capitalize() for w in raw.split()) if raw.isupper() else raw
            doc.doc_type = "card"
        if rev:
            doc.revision = rev.group(1)
        if date:
            doc.effective_date = date.group(1)
        doc.models = extract_models(doc.title + " " + doc.filename + " " + str(doc.meta.get("ocr_title", "")))
