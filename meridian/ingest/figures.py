"""Figure understanding: visual descriptors, figure-type classification and a pixel-level chart digitiser.

The digitiser turns a raster bar/line chart into numbers: it finds the axes box, OCRs the tick labels, fits a
pixel->value calibration, isolates series by colour and reads bar tops / line heights. Results are always flagged
as approximate and carry a confidence score; low-confidence reads are discarded rather than indexed.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from . import ocr as ocrmod
from .ocr import Word

DESCRIPTOR_DIM = 72 + 128 + 64


# ----------------------------------------------------------------------------------------- visual descriptor
def _letterbox(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    s = size / max(h, w)
    resized = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def visual_descriptor(img: Image.Image) -> np.ndarray:
    """Compact, L2-normalised global descriptor (colour histogram + gradient-orientation grid + ink layout).

    Robust to rescaling, mild noise and JPEG artefacts; used for image-to-figure retrieval.
    """
    arr = np.asarray(img.convert("RGB"))
    sq = _letterbox(arr, 256)
    hsv = cv2.cvtColor(sq, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
    hist = np.sqrt(hist / max(hist.sum(), 1e-9))
    gray = cv2.cvtColor(cv2.resize(sq, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 8.0
    bins = np.minimum(ang.astype(int), 7)
    hog = np.zeros((4, 4, 8), np.float32)
    cell = 32
    for i in range(4):
        for j in range(4):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            b = bins[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            hog[i, j] = np.bincount(b.ravel(), weights=m.ravel(), minlength=8)[:8]
    hog = np.sqrt(hog.flatten() / max(hog.sum(), 1e-9))
    ink = (255 - cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)).astype(np.float32).flatten()
    ink = np.sqrt(ink / max(ink.sum(), 1e-9))
    vec = np.concatenate([0.5 * hist, 1.0 * hog, 0.8 * ink]).astype(np.float32)
    return vec / max(np.linalg.norm(vec), 1e-9)


# ----------------------------------------------------------------------------------------- classification
@dataclass
class ImageStats:
    white_frac: float
    sat_frac: float
    dark_bg: bool
    has_axes: bool
    n_words: int
    numeric_ratio: float


def _axes_geometry(gray: np.ndarray) -> Optional[dict]:
    """Locate the plot box from long dark horizontal/vertical lines."""
    h, w = gray.shape
    dark = (gray < 90).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(0.30 * w)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(0.30 * h))))
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    rows = np.where(hl.sum(axis=1) > 0.25 * w)[0]
    cols = np.where(vl.sum(axis=0) > 0.25 * h)[0]
    if rows.size == 0 or cols.size == 0:
        return None

    def centers(idx: np.ndarray) -> list[float]:
        groups, cur = [], [idx[0]]
        for v in idx[1:]:
            if v - cur[-1] <= 2:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        return [float(np.mean(g)) for g in groups]

    ry, cx = centers(rows), centers(cols)
    left, bottom = min(cx), max(ry)
    right_candidates = np.where(hl[int(round(bottom))] > 0)[0] if hl[int(round(bottom))].any() else np.array([int(left) + 1])
    right = float(right_candidates.max())
    vcol = vl[:, int(round(left))]
    vrows = np.where(vcol > 0)[0]
    top = float(vrows.min()) if vrows.size else float(min(ry))
    if right - left < 0.25 * w or bottom - top < 0.2 * h:
        return None
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def image_stats(img: Image.Image, words: Optional[list[Word]] = None) -> ImageStats:
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = (hsv[..., 1] > 90) & (hsv[..., 2] > 70)
    words = words or []
    nums = sum(1 for w in words if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?%?", w.text))
    return ImageStats(
        white_frac=float((gray > 235).mean()), sat_frac=float(sat.mean()), dark_bg=bool(gray.mean() < 110),
        has_axes=_axes_geometry(gray) is not None, n_words=len(words), numeric_ratio=(nums / len(words)) if words else 0.0,
    )


def classify_figure(img: Image.Image, words: Optional[list[Word]] = None) -> str:
    """Return 'chart' | 'diagram' | 'photo' | 'document'."""
    st = image_stats(img, words)
    if st.dark_bg:
        return "photo"
    if st.has_axes and st.white_frac > 0.5 and (st.sat_frac > 0.002 or st.numeric_ratio > 0.3):
        return "chart"
    if st.n_words >= 60 and st.white_frac > 0.6:
        return "document"
    if st.dark_bg or st.white_frac < 0.45:
        return "photo"
    return "diagram"


# ----------------------------------------------------------------------------------------- chart reader
@dataclass
class ChartSeries:
    name: str
    points: list[tuple[object, float]]  # (x label | x value, y value)


@dataclass
class ChartData:
    kind: str  # bar | line
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    x_unit: str = ""
    y_unit: str = ""
    series: list[ChartSeries] = field(default_factory=list)
    confidence: float = 0.0
    calib: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.series:
            return ""
        parts = [f"{self.kind.capitalize()} chart"]
        if self.title:
            parts.append(f"titled '{self.title}'")
        axes = []
        if self.x_title:
            axes.append(f"x axis: {self.x_title}")
        if self.y_title:
            axes.append(f"y axis: {self.y_title}")
        head = " ".join(parts) + ("; " + "; ".join(axes) if axes else "") + "."
        pts = self.series[0].points
        if self.kind == "line" and len(pts) > 14:
            step = max(1, len(pts) // 12)
            pts = pts[::step]
        body = ", ".join(f"{_fmt(x)} -> {_fmt(y)}" for x, y in pts)
        return f"{head} Approximate values read from the chart (x -> y): {body}."

    def to_meta(self) -> dict:
        return {"kind": self.kind, "title": self.title, "x_title": self.x_title, "y_title": self.y_title, "x_unit": self.x_unit,
                "y_unit": self.y_unit, "confidence": round(self.confidence, 3),
                "series": [{"name": s.name, "points": [[x, round(float(y), 4)] for x, y in s.points]} for s in self.series]}


def _fmt(v: object) -> str:
    if isinstance(v, (int, float, np.floating)):
        v = float(v)
        return str(int(round(v))) if abs(v - round(v)) < 0.05 else f"{v:.1f}"
    return str(v)


def _num(text: str) -> Optional[float]:
    t = text.strip().replace("−", "-").replace(",", ".")
    t = t.rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def _fit(pairs: list[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    """Robust linear fit value = a*pixel + b. Returns (a, b, rms)."""
    pts = list(pairs)
    while len(pts) >= 2:
        px = np.array([p for p, _ in pts])
        vals = np.array([v for _, v in pts])
        if np.ptp(px) < 1e-6:
            return None
        a, b = np.polyfit(px, vals, 1)
        res = vals - (a * px + b)
        rms = float(np.sqrt(np.mean(res ** 2)))
        span = float(np.ptp(vals)) or 1.0
        if len(pts) > 3 and np.max(np.abs(res)) > 0.04 * span:
            pts.pop(int(np.argmax(np.abs(res))))
            continue
        return float(a), float(b), rms / span
    return None


def _tick_marks(dark: np.ndarray, axis: str, pos: float, lo: int, hi: int, k: float = 1.0) -> list[float]:
    """Pixel positions of tick marks just outside the axis (rows for y axis, columns for x axis)."""
    out: list[float] = []
    near, far, need = max(2, int(2 * k)), max(6, int(8 * k)), max(2, int(3 * k))
    if axis == "y":
        x = int(pos)
        strip = dark[lo:hi, max(0, x - far):max(1, x - near)]
        hits = np.where(strip.sum(axis=1) >= need)[0] + lo
    else:
        y = int(pos)
        strip = dark[y + near:y + far, lo:hi]
        hits = np.where(strip.sum(axis=0) >= need)[0] + lo
    if hits.size == 0:
        return out
    cur = [hits[0]]
    for v in hits[1:]:
        if v - cur[-1] <= 2:
            cur.append(v)
        else:
            out.append(float(np.mean(cur)))
            cur = [v]
    out.append(float(np.mean(cur)))
    return out


def _snap(value: float, ticks: list[float], tol: float = 9.0) -> float:
    if not ticks:
        return value
    best = min(ticks, key=lambda t: abs(t - value))
    return best if abs(best - value) <= tol else value


def _ocr_region(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, **kw) -> list[Word]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 6:
        return []
    crop = img[y0:y1, x0:x1]
    words = ocrmod.ocr_words(crop, **kw)
    for wd in words:
        wd.x0 += x0; wd.x1 += x0; wd.y0 += y0; wd.y1 += y0
    return words


def _ocr_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, rotate: bool = False) -> str:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 10 or y1 - y0 < 8:
        return ""
    crop = img[y0:y1, x0:x1]
    if rotate:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    pad = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    try:
        words = ocrmod.ocr_words(pad, psm=7, scale=3.0, min_conf=30, invert_if_dark=False)
    except ocrmod.OCRUnavailable:
        return ""
    return " ".join(w.text for w in sorted(words, key=lambda w: w.x0)).strip()



def _read_tick_label(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, allow: str = "0123456789.-") -> Optional[str]:
    """Read one short numeric label by *voting* across scales and binarisations.

    A single Tesseract pass misreads short strings with high confidence ("50%" -> "90%"); voting over 2x-6x scales and
    raw/Otsu variants recovers them reliably. The vote stops early once one reading clearly dominates.
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = img[y0:y1, x0:x1]
    variants = [crop]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    otsu = cv2.cvtColor(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB)
    variants.append(otsu)
    votes: Counter = Counter()
    n = 0
    for scale in (3.0, 4.0, 2.0, 5.0, 6.0):
        for v in variants:
            pad = cv2.copyMakeBorder(v, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            try:
                words = ocrmod.ocr_words(pad, psm=7, scale=scale, min_conf=0, whitelist=allow, invert_if_dark=False)
            except ocrmod.OCRUnavailable:
                return None
            text = "".join(wd.text for wd in sorted(words, key=lambda wd: wd.x0)).strip(".-")
            if text:
                votes[text] += 1
            n += 1
            if n >= 4 and votes:
                (top, c1), *rest = votes.most_common(2)
                if c1 >= 4 and (not rest or c1 >= 3 * rest[0][1]):
                    return top
    if not votes:
        return None
    top, c1 = votes.most_common(1)[0]
    return top if c1 >= 2 else None


def _repair_labels(labels: list[Optional[str]]) -> list[Optional[str]]:
    """Rebuild missing/implausible numeric category labels from an arithmetic progression (25%, 50%, 75%, 100%).

    Needs at least two labels that agree on a common step; with fewer, labels are left untouched (a wrong label is worse than none).
    """
    vals: list[Optional[float]] = []
    for l in labels:
        vals.append(float(l.rstrip("%")) if l and re.fullmatch(r"\d+(?:\.\d+)?%?", l) else None)
    idx = [i for i, v in enumerate(vals) if v is not None]
    if len(labels) < 3 or len(idx) < 2:
        return labels
    best = None
    for a_i in range(len(idx)):
        for b_i in range(a_i + 1, len(idx)):
            i, j = idx[a_i], idx[b_i]
            step = (vals[j] - vals[i]) / (j - i)
            if step <= 0:
                continue
            base = vals[i] - step * i
            inliers = [m for m in idx if abs(base + step * m - vals[m]) < 1e-6 * max(1.0, abs(vals[m]))]
            if best is None or len(inliers) > best[0]:
                best = (len(inliers), base, step)
    if best is None or best[0] < 3 or best[1] < 0:  # demand 3 agreeing labels: a wrong label is worse than none
        return labels
    pct = sum(1 for l in labels if l and l.endswith("%")) >= max(1, len([l for l in labels if l]) // 2)
    out = []
    for m in range(len(labels)):
        v = best[1] + best[2] * m
        out.append((f"{v:g}" + ("%" if pct else "")))
    return out


def _sane_title(text: str) -> str:
    """Drop OCR garbage: keep only titles made of plausible words."""
    text = (text or "").strip()
    if not text:
        return ""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*|\([^)]*\)", text)
    letters = sum(c.isalpha() for c in text)
    if letters < 0.6 * len(text.replace(" ", "")) or any(c in text for c in "\\|_~"):
        return ""
    good = [w for w in words if len(w) >= 3 or w.startswith("(")]
    return text if len(good) >= max(1, int(0.6 * len(words))) and re.search(r"[aeiouAEIOU]", text) and len(text) <= 60 else ""


def _split_unit(title: str) -> tuple[str, str]:
    title = _sane_title(title)
    m = re.search(r"\(([^)]{1,16})\)", title)
    return (title, m.group(1).strip()) if m else (title, "")


def read_chart(img: Image.Image) -> Optional[ChartData]:
    """Digitise a raster bar/line chart. Returns None when the chart cannot be read with adequate confidence."""
    if not ocrmod.ocr_available():
        return None
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    box = _axes_geometry(gray)
    if box is None:
        return None
    left, right, top, bottom = box["left"], box["right"], box["top"], box["bottom"]
    dark = (gray < 110).astype(np.uint8)
    k = max(0.6, W / 850.0)  # all pixel offsets are relative to a ~850px wide chart

    # ---- y calibration from tick labels
    yticks = _tick_marks(dark, "y", left, int(top) - 4, int(bottom) + 5, k)
    ylab_w = int(min(0.075 * W, 70 * k))
    ypairs = []
    for ty in yticks:
        label = _read_tick_label(rgb, int(left) - ylab_w, int(ty - 13 * k), int(left - 9 * k), int(ty + 13 * k))
        if not label:
            label = _read_tick_label(rgb, int(left) - int(ylab_w * 1.2), int(ty - 16 * k), int(left - 5 * k), int(ty + 16 * k))
        v = _num(label) if label else None
        if v is not None:
            ypairs.append((ty, v))
    yfit = _fit(ypairs)
    if yfit is None or len(ypairs) < 2:
        return None
    ya, yb, yrms = yfit
    if ya >= 0:  # values must decrease as pixel rows increase
        return None

    def yval(py: float) -> float:
        return ya * py + yb

    # ---- x tick labels (numeric for line charts, text for bar charts)
    xticks = _tick_marks(dark, "x", bottom, int(left) - 4, int(right) + 5, k)
    xwords: list[Word] = []
    for tx in xticks:
        label = _read_tick_label(rgb, int(tx - 34 * k), int(bottom + 8 * k), int(tx + 34 * k), int(bottom + 44 * k), allow="0123456789.%-")
        if not label:
            label = _read_tick_label(rgb, int(tx - 40 * k), int(bottom + 5 * k), int(tx + 40 * k), int(bottom + 56 * k), allow="0123456789.%-")
        if label:
            xwords.append(Word(label, 90.0, tx - 1, bottom + 8, tx + 1, bottom + 30))
    if len(xwords) < 2:  # bar charts have no ticks: OCR the label band as words
        band = _ocr_region(rgb, int(left) - 25, int(bottom) + 3, int(right) + 30, int(bottom) + int(0.085 * H) + 6, psm=11, scale=3.0,
                           whitelist="0123456789.%", invert_if_dark=False)
        xwords = [w for w in band if (w.y1 - w.y0) < 45]

    # ---- series by colour inside the plot box
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    inner = np.zeros((H, W), bool)
    inner[int(top) + 3:int(bottom) - 2, int(left) + 3:int(right) - 2] = True
    colored = (hsv[..., 1] > 90) & (hsv[..., 2] > 60) & inner
    if colored.sum() < 60:
        return None
    hue = hsv[..., 0][colored]
    hist = np.bincount((hue // 10).astype(int), minlength=18)
    order = [int(b) for b in np.argsort(hist)[::-1] if hist[b] >= 120][:4]
    if not order:
        return None

    chart = ChartData(kind="line")
    chart.title = _ocr_line(rgb, int(left), 2, int(right), int(top) - 3)
    xt = _ocr_line(rgb, int(left), int(bottom + 38 * k), int(right), H - 2)
    yt = _ocr_line(rgb, 0, int(top), max(int(left) - 48, 30), int(bottom), rotate=True)
    chart.x_title, chart.x_unit = _split_unit(xt)
    chart.y_title, chart.y_unit = _split_unit(yt)

    for si, hb in enumerate(order):  # NB: not `k` -- that is the pixel-scale factor used throughout this function
        mask = (colored & ((hsv[..., 0] // 10) == hb)).astype(np.uint8)
        # merge neighbouring hue bins (anti-aliasing) into the mask
        for nb in (hb - 1, hb + 1):
            if 0 <= nb < 18 and hist[nb] < hist[hb] * 0.5:
                mask |= (colored & ((hsv[..., 0] // 10) == nb)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        comps = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 40]
        if not comps:
            continue
        bars = [c for c in comps if c[cv2.CC_STAT_WIDTH] >= 8 and c[cv2.CC_STAT_AREA] / max(1, c[cv2.CC_STAT_WIDTH] * c[cv2.CC_STAT_HEIGHT]) >= 0.85]
        if len(bars) >= 2 and all(abs((b[cv2.CC_STAT_TOP] + b[cv2.CC_STAT_HEIGHT]) - (bars[0][cv2.CC_STAT_TOP] + bars[0][cv2.CC_STAT_HEIGHT])) <= 4 for b in bars):
            chart.kind = "bar"
            bars.sort(key=lambda b: b[cv2.CC_STAT_LEFT])
            pts = []
            labels: list[Optional[str]] = []
            for i, b in enumerate(bars):
                cx = b[cv2.CC_STAT_LEFT] + b[cv2.CC_STAT_WIDTH] / 2
                half = max(b[cv2.CC_STAT_WIDTH] / 2 + 6 * k, 30 * k)
                label = None
                for y0, y1 in ((14, 58), (11, 64), (17, 54), (5, 52)):  # start below the tick marks, whose strokes OCR reads as digits  # retry with different bands: OCR is fragile on short numeric strings
                    cand = _read_tick_label(rgb, int(cx - half), int(bottom + y0 * k), int(cx + half), int(bottom + y1 * k), allow="0123456789.%-")
                    if cand and re.fullmatch(r"\d+(?:\.\d+)?%?", cand):
                        label = cand
                        break
                labels.append(label)
            labels = _repair_labels(labels)
            for i, b in enumerate(bars):
                pts.append((labels[i] or f"bar {i + 1}", yval(float(b[cv2.CC_STAT_TOP]))))
            chart.series.append(ChartSeries(f"series {si + 1}", pts))
        else:
            ys = {}
            ptsmask = np.argwhere(mask > 0)
            if ptsmask.shape[0] < 30:
                continue
            for row, col in ptsmask:
                ys.setdefault(int(col), []).append(int(row))
            cols = sorted(ys)
            if cols[-1] - cols[0] < 0.25 * (right - left):
                continue
            curve_x = np.array(cols, float)
            curve_y = np.array([np.mean(ys[c]) for c in cols], float)
            xpairs = []
            for wd in xwords:
                v = _num(wd.text)
                if v is not None and "%" not in wd.text:
                    xpairs.append((_snap(wd.cx, xticks), v))
            xfit = _fit(xpairs) if len(xpairs) >= 3 else None
            if xfit is not None and (xfit[0] <= 0 or xfit[2] > 0.02):
                xfit = None
            if xfit is not None:
                xa, xb, _ = xfit
                xs_val = xa * curve_x + xb
                slack = 0.04 * float(np.ptp(xs_val))
                tick_vals = sorted({v for _, v in xpairs if xs_val.min() - slack <= v <= xs_val.max() + slack})
                if len(tick_vals) < 3:
                    tick_vals = list(np.linspace(xs_val.min(), xs_val.max(), 9))
                pts = [(float(tv), yval(float(np.interp(tv, xs_val, curve_y)))) for tv in tick_vals]
                chart.calib["x"] = {"a": xa, "b": xb}
            else:
                continue  # an uncalibrated x axis would yield meaningless numbers: refuse instead of guessing
            chart.kind = "line"
            chart.series.append(ChartSeries(f"series {si + 1}", pts))
    if not chart.series:
        return None
    conf = 0.35 + 0.2 * min(len(ypairs), 5) / 5 + 0.2 * (1.0 - min(1.0, yrms * 25)) + 0.15 * (1 if chart.title else 0) + 0.1 * (1 if xwords else 0)
    chart.confidence = float(min(conf, 0.99))
    chart.calib["y"] = {"a": ya, "b": yb, "rms_rel": yrms, "n": len(ypairs)}
    return chart if chart.confidence >= 0.55 else None


def interpolate_series(points: list[list], x: float) -> Optional[float]:
    """Linear interpolation over numeric (x, y) points; None when x is outside the digitised range."""
    try:
        xs = np.array([float(p[0]) for p in points])
        ys = np.array([float(p[1]) for p in points])
    except (TypeError, ValueError):
        return None
    if xs.size < 2 or x < xs.min() - 1e-9 or x > xs.max() + 1e-9:
        return None
    order = np.argsort(xs)
    return float(np.interp(x, xs[order], ys[order]))


# ----------------------------------------------------------------------------------------- analyzer facade
@dataclass
class FigureAnalysis:
    kind: str
    ocr_text: str
    words: list[Word]
    descriptor: np.ndarray
    chart: Optional[ChartData] = None
    codes: list[str] = field(default_factory=list)


def analyze_figure(img: Image.Image, *, lang: str = "eng") -> FigureAnalysis:
    img = img.convert("RGB")
    words: list[Word] = []
    text = ""
    codes: list[str] = []
    if ocrmod.ocr_available():
        try:
            best = ocrmod.ocr_best(img, lang=lang)
            words, text, codes = best.words, best.text, best.codes
        except Exception:  # OCR is best-effort for figures
            words, text = [], ""
    kind = classify_figure(img, words)
    chart = read_chart(img) if kind == "chart" else None
    return FigureAnalysis(kind=kind, ocr_text=text, words=words, descriptor=visual_descriptor(img), chart=chart, codes=codes)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))
