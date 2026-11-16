"""Renders the demo corpus (PDF manuals with real tables/charts/diagrams, a scanned card, a DOCX service bulletin,
a CSV maintenance log) and the query photos used by the golden QA set. Fully deterministic (seeded).

Requires the *dev* extras: reportlab, matplotlib, python-docx, Pillow.
"""
from __future__ import annotations

import random
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from . import world as W


def _font_path(name: str) -> str:
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name)


# ============================================================================ figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_exploded(path: Path) -> None:
    plt = _mpl()
    from matplotlib.patches import Ellipse, Rectangle

    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=150)
    ax.set_xlim(0, 100), ax.set_ylim(0, 62), ax.axis("off")
    ax.text(50, 58, "HX-200 EXPLODED VIEW", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.add_patch(Rectangle((6, 28), 88, 4, fc="#b5b5b5", ec="black"))  # shaft
    ax.add_patch(Rectangle((12, 16), 16, 28, fc="#ececec", ec="black"))  # casing
    ax.add_patch(Ellipse((38, 30), 6, 22, fc="#cfcfcf", ec="black"))  # impeller
    ax.add_patch(Rectangle((46, 23), 6, 14, fc="#dcdcdc", ec="black"))  # seal
    ax.add_patch(Rectangle((58, 20), 16, 20, fc="#ececec", ec="black"))  # bearing housing
    ax.add_patch(Rectangle((80, 22), 10, 16, fc="#d2d2d2", ec="black"))  # coupling
    callouts = [("1  Casing", (20, 44), (20, 53)), ("2  Impeller", (38, 41), (38, 50)), ("3  Mechanical seal", (49, 23), (49, 10)),
                ("4  Shaft", (70, 28), (70, 8)), ("5  Bearing housing", (64, 40), (62, 50)), ("6  Coupling", (88, 38), (92, 46))]
    for label, xy, xyt in callouts:
        ax.annotate(label, xy=xy, xytext=xyt, ha="center", fontsize=11,
                    arrowprops=dict(arrowstyle="-", color="black", lw=1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _boxes(ax, items, y, w=17, h=10, gap=4.0, x0=3.0):
    xs = []
    for i, label in enumerate(items):
        x = x0 + i * (w + gap)
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((x, y), w, h, fc="#efefef", ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
        xs.append(x)
    return xs, w, h


def fig_flow(path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    ax.set_xlim(0, 110), ax.set_ylim(0, 60), ax.axis("off")
    ax.text(55, 56, "HX-500 AIR AND OIL FLOW", ha="center", fontsize=13, fontweight="bold")
    xs, w, h = _boxes(ax, ["Intake\nfilter", "Compressor\nelement", "Oil\nseparator", "After-\ncooler", "Air\noutlet"], 34, w=16, gap=5.5, x0=2)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], 39), xytext=(xs[i] + w, 39), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((44, 8), 16, 10, fc="#efefef", ec="black", lw=1.2))
    ax.text(52, 13, "Oil cooler", ha="center", va="center", fontsize=9.5)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((70, 8), 16, 10, fc="#efefef", ec="black", lw=1.2))
    ax.text(78, 13, "Oil filter", ha="center", va="center", fontsize=9.5)
    ax.annotate("", xy=(60, 13), xytext=(xs[2] + w / 2, 34), arrowprops=dict(arrowstyle="->", lw=1.3, ls="--"))
    ax.annotate("", xy=(xs[1] + w / 2, 34), xytext=(70, 13), arrowprops=dict(arrowstyle="->", lw=1.3, ls="--"))
    ax.text(55, 2, "solid = air path    dashed = oil return", ha="center", fontsize=9)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_circuit(path: Path) -> None:
    plt = _mpl()
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    ax.set_xlim(0, 110), ax.set_ylim(0, 60), ax.axis("off")
    ax.text(55, 56, "HX-900 HYDRAULIC CIRCUIT", ha="center", fontsize=13, fontweight="bold")
    boxes = {"Tank": (4, 8), "Pump": (30, 8), "Relief\nvalve": (30, 34), "Return\nfilter": (56, 34), "Actuator": (82, 20)}
    for label, (x, y) in boxes.items():
        ax.add_patch(Rectangle((x, y), 20, 12, fc="#efefef", ec="black", lw=1.2))
        ax.text(x + 10, y + 6, label, ha="center", va="center", fontsize=10)
    for (a, b) in [((24, 14), (30, 14)), ((50, 14), (92, 14)), ((92, 14), (92, 20)), ((92, 32), (92, 40)), ((92, 40), (76, 40)),
                   ((56, 40), (50, 40)), ((40, 20), (40, 34)), ((30, 40), (14, 40)), ((14, 40), (14, 20))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(55, 3, "Pressure line: pump to actuator    Return line: actuator to tank", ha="center", fontsize=9)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_line(path: Path, data: dict, color: str = "#1f77b4", ymax: float = 70, ystep: float = 10) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(data["x"], data["y"], marker="o", color=color, linewidth=2.4, markersize=6)
    ax.set_xticks(data["x"])
    ax.set_yticks(np.arange(0, ymax + 0.1, ystep))
    ax.set_ylim(0, ymax)
    ax.set_xlabel(data["xlabel"], fontsize=11)
    ax.set_ylabel(data["ylabel"], fontsize=11)
    ax.set_title(data["title"], fontsize=12, fontweight="bold")
    ax.grid(True, color="#dcdcdc", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_bars(path: Path, data: dict, color: str = "#ff7f0e", ymax: float = 8, ystep: float = 1) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.bar(data["x"], data["y"], color=color, width=0.6)
    ax.set_yticks(np.arange(0, ymax + 0.1, ystep))
    ax.set_ylim(0, ymax)
    ax.set_xlabel(data["xlabel"], fontsize=11)
    ax.set_ylabel(data["ylabel"], fontsize=11)
    ax.set_title(data["title"], fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", color="#dcdcdc", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_figures(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "hx200_exploded": out_dir / "hx200_exploded.png",
        "hx200_curve": out_dir / "hx200_curve.png",
        "hx500_flow": out_dir / "hx500_flow.png",
        "hx500_bars": out_dir / "hx500_bars.png",
        "hx900_circuit": out_dir / "hx900_circuit.png",
        "hx900_temp": out_dir / "hx900_temp.png",
    }
    fig_exploded(figs["hx200_exploded"])
    fig_line(figs["hx200_curve"], W.HX200_CURVE, ymax=70, ystep=10)
    fig_flow(figs["hx500_flow"])
    fig_bars(figs["hx500_bars"], W.HX500_BARS, ymax=8, ystep=1)
    fig_circuit(figs["hx900_circuit"])
    fig_line(figs["hx900_temp"], W.HX900_TEMP, color="#d62728", ymax=70, ystep=10)
    return figs


# ============================================================================ manuals (reportlab)
def render_manual(spec: W.ManualSpec, figs: dict[str, Path], out_path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from PIL import Image as PILImage

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24, leading=28, spaceAfter=4)
    subtitle = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=17, spaceAfter=2)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19, spaceBefore=12, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    step = ParagraphStyle("step", parent=body, leftIndent=14, spaceAfter=3)
    warn = ParagraphStyle("warn", parent=body, backColor=colors.HexColor("#fff1c2"), borderPadding=5, leftIndent=8, rightIndent=8,
                          spaceBefore=6, spaceAfter=10)
    cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9, leading=12, spaceAfter=8)
    tcap = ParagraphStyle("tcap", fontName="Helvetica-Bold", fontSize=10, leading=13, spaceBefore=4, spaceAfter=4)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=9, leading=11)
    cellh = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=9, leading=11)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, f"{W.MANUFACTURER} | {spec.model} {spec.product} Service Manual | Revision {spec.revision} | Page {doc.page}")
        canvas.restoreState()

    story: list = [
        Paragraph(escape(f"{spec.model} {spec.product}"), title),
        Paragraph("Service Manual", subtitle),
        Paragraph(escape(f"Document revision: {spec.revision}"), subtitle),
        Paragraph(escape(f"Revision date: {spec.date}"), subtitle),
        Paragraph(escape(f"Publisher: {W.MANUFACTURER} GmbH"), subtitle),
        Spacer(1, 10),
    ]
    for block in spec.blocks:
        kind = block[0]
        if kind == "h1":
            story.append(Paragraph(escape(block[1]), h1))
        elif kind == "h2":
            story.append(Paragraph(escape(block[1]), h2))
        elif kind == "p":
            story.append(Paragraph(escape(block[1]), body))
        elif kind == "warn":
            head, _, rest = block[1].partition(":")
            story.append(Paragraph(f"<b>{escape(head)}:</b>{escape(rest)}", warn))
        elif kind == "steps":
            for i, s in enumerate(block[1], 1):
                story.append(Paragraph(escape(f"{i}. {s}"), step))
            story.append(Spacer(1, 4))
        elif kind == "table":
            _, caption, header, rows, widths = block
            data = [[Paragraph(escape(h), cellh) for h in header]] + [[Paragraph(escape(c), cell) for c in r] for r in rows]
            t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story.append(KeepTogether([Paragraph(escape(caption), tcap), t, Spacer(1, 8)]))
        elif kind == "fig":
            _, key, caption, wcm = block
            with PILImage.open(figs[key]) as im:
                ratio = im.height / im.width
            img = Image(str(figs[key]), width=wcm * cm, height=wcm * cm * ratio)
            story.append(KeepTogether([img, Paragraph(escape(caption), cap)]))
        elif kind == "pb":
            story.append(PageBreak())
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"{spec.model} Service Manual", author=W.MANUFACTURER)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ============================================================================ bulletin (DOCX)
def render_bulletin(out_path: Path) -> None:
    from docx import Document

    b = W.BULLETIN
    doc = Document()
    doc.core_properties.title = b["title"]
    doc.core_properties.author = W.MANUFACTURER
    doc.add_heading(b["title"], level=0)
    doc.add_paragraph(f"Date: {b['date']}")
    doc.add_paragraph(f"Subject: {b['subject']}")
    doc.add_paragraph(f"Applies to: {b['applies']}")
    doc.add_heading("1 Background", level=1)
    for p in b["paragraphs"]:
        doc.add_paragraph(p)
    doc.add_heading("2 Revised value", level=1)
    t = doc.add_table(rows=1, cols=len(b["table"]["header"]))
    t.style = "Table Grid"
    for i, h in enumerate(b["table"]["header"]):
        t.rows[0].cells[i].text = h
    for row in b["table"]["rows"]:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    doc.add_paragraph("")
    doc.add_paragraph(b["supersedes"])
    doc.add_heading("3 Action required", level=1)
    doc.add_paragraph(b["action"])
    p = doc.add_paragraph()
    head, _, rest = b["warning"].partition(":")
    p.add_run(head + ":").bold = True
    p.add_run(rest)
    doc.add_heading("4 Affected serial numbers", level=1)
    t2 = doc.add_table(rows=1, cols=2)
    t2.style = "Table Grid"
    for i, h in enumerate(b["serial_table"]["header"]):
        t2.rows[0].cells[i].text = h
    for row in b["serial_table"]["rows"]:
        cells = t2.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    doc.save(str(out_path))


# ============================================================================ scanned card + photos (PIL)
def _noise(img, sigma: float, seed: int):
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0, sigma, arr.shape)
    from PIL import Image

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_scanned_card(out_path: Path, seed: int = 7) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(_font_path("DejaVuSans-Bold.ttf"), 40)
    f_body = ImageFont.truetype(_font_path("DejaVuSans.ttf"), 34)
    y = 150
    for i, line in enumerate(W.FIELD_CARD_LINES):
        d.text((90, y), line, font=f_title if i == 0 else f_body, fill=(25, 25, 25))
        y += 82 if i == 0 else 74
    img = img.rotate(0.7, resample=Image.BICUBIC, fillcolor="white")
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    img = _noise(img, 7.0, seed)
    img.save(out_path, "PDF", resolution=150.0)


def _display_image(lines: list[str], out_path: Path, seed: int) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    img = Image.new("RGB", (900, 520), (10, 38, 26))
    d = ImageDraw.Draw(img)
    f_big = ImageFont.truetype(_font_path("DejaVuSansMono-Bold.ttf"), 64)
    f_small = ImageFont.truetype(_font_path("DejaVuSansMono-Bold.ttf"), 44)
    y = 70
    for i, line in enumerate(lines):
        d.text((50, y), line, font=f_big if i == 1 else f_small, fill=(140, 255, 150))
        y += 120
    img = img.rotate(1.5, resample=Image.BICUBIC, fillcolor=(10, 38, 26))
    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img = _noise(img, 6.0, seed)
    img.save(out_path)


def _data_plate(lines: list[str], out_path: Path, seed: int) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 960, 600
    base = np.zeros((h, w, 3), np.uint8)
    grad = np.linspace(185, 150, h)[:, None]
    base[:] = grad[..., None].astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    d.rectangle([14, 14, w - 14, h - 14], outline=(70, 70, 70), width=6)
    f = ImageFont.truetype(_font_path("DejaVuSans-Bold.ttf"), 52)
    y = 60
    for line in lines:
        d.text((70, y), line, font=f, fill=(20, 20, 20))
        y += 96
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img = _noise(img, 5.0, seed)
    img.save(out_path)


def render_query_images(eval_dir: Path, figs: dict[str, Path]) -> None:
    from PIL import Image, ImageFilter

    img_dir = eval_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(W.DISPLAY_ALARMS):
        _display_image(spec["lines"], img_dir / spec["file"], seed=11 + i)
    _data_plate(W.DATA_PLATE["lines"], img_dir / W.DATA_PLATE["file"], seed=21)
    # a degraded re-photograph of the exploded view (smaller, rotated, noisy, JPEG compressed)
    with Image.open(figs["hx200_exploded"]) as im:
        im = im.convert("RGB")
        im = im.resize((int(im.width * 0.7), int(im.height * 0.7)))
        im = im.rotate(2.5, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        im = im.filter(ImageFilter.GaussianBlur(0.6))
        im = _noise(im, 6.0, 31)
        im.save(img_dir / "query_diagram_exploded.jpg", quality=55)


# ============================================================================ maintenance log
def render_log(out_path: Path, seed: int = 2026, n: int = 150) -> list[dict]:
    import csv
    from datetime import date, timedelta

    rng = random.Random(seed)
    models = ["HX-200"] * 5 + ["HX-500"] * 3 + ["HX-900"] * 2
    weights = {
        "HX-200": [2, 2, 5, 3, 1],  # E-47 most common on pumps
        "HX-500": [2, 4, 2, 2],
        "HX-900": [2, 2, 4, 1],
    }
    base = {"E-12": 3.0, "E-31": 2.0, "E-47": 1.5, "E-52": 6.0, "E-63": 4.0, "E-05": 2.5, "E-58": 3.5, "E-71": 1.0,
            "E-09": 1.0, "E-20": 2.0, "E-88": 5.0}
    start = date(2026, 1, 5)
    rows: list[dict] = []
    for _ in range(n):
        model = rng.choice(models)
        codes = W.LOG_CODES[model]
        code, action = rng.choices(codes, weights=weights[model], k=1)[0]
        rows.append({
            "date": (start + timedelta(days=rng.randint(0, 300))).isoformat(),
            "unit_id": rng.choice(W.LOG_UNITS[model]),
            "model": model,
            "error_code": code,
            "action_taken": action,
            "downtime_h": round(base[code] * rng.uniform(0.5, 1.6) * 2) / 2,
            "technician": rng.choice(W.TECHNICIANS),
        })
    rows.sort(key=lambda r: r["date"])
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows


# ============================================================================ orchestration
def build_demo_corpus(corpus_dir: Path, eval_dir: Path, work_dir: Path | None = None) -> dict:
    """Render everything. Returns a manifest dict (also useful for QA generation)."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    work = work_dir or (eval_dir / "_figures")
    figs = render_figures(work)
    files: list[str] = []
    for spec in W.MANUALS:
        render_manual(spec, figs, corpus_dir / spec.filename)
        files.append(spec.filename)
    render_bulletin(corpus_dir / W.BULLETIN["filename"])
    files.append(W.BULLETIN["filename"])
    render_scanned_card(corpus_dir / "HX-200_Field_Card_scan.pdf")
    files.append("HX-200_Field_Card_scan.pdf")
    log_rows = render_log(corpus_dir / "maintenance_log_2026.csv")
    files.append("maintenance_log_2026.csv")
    render_query_images(eval_dir, figs)
    from .qa import write_golden
    write_golden(eval_dir, corpus_dir)
    return {"files": files, "log_rows": len(log_rows), "figures": {k: str(v) for k, v in figs.items()}}
