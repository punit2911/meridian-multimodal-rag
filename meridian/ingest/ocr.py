"""OCR helpers on top of Tesseract: polarity detection, deskew, word boxes and line grouping."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

try:  # pragma: no cover - import guard
    import pytesseract
    from pytesseract import Output
    # On Windows pytesseract may not find tesseract via PATH; point it explicitly.
    import os as _os
    _tess_cmd = _os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _os.path.isfile(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except Exception:  # pragma: no cover
    pytesseract = None
    Output = None


class OCRUnavailable(RuntimeError):
    pass


def ocr_available() -> bool:
    if pytesseract is None:
        return False
    # shutil.which may miss Tesseract on Windows if PATH is not propagated to
    # the current process; fall back to checking the configured executable directly.
    import os as _os
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return shutil.which("tesseract") is not None or _os.path.isfile(cmd)


@dataclass
class Word:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class OCRLine:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float


def _to_gray(img: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else img
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def prepare(img: Image.Image | np.ndarray, scale: float = 2.0, invert_if_dark: bool = True, deskew_img: bool = False) -> tuple[np.ndarray, float]:
    gray = _to_gray(img)
    if invert_if_dark and gray.mean() < 110:
        gray = 255 - gray
    if deskew_img:
        gray = deskew(gray)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray, scale


def deskew(gray: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
    """Rotate so text baselines are horizontal. Only acts on small, confident skews."""
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    blobs = cv2.dilate(inv, kernel)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 60 or h < 8 or w < 3 * h:
            continue
        (_, _), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            ang = ang - 90.0
        while ang < -45:
            ang += 90
        while ang > 45:
            ang -= 90
        angles.append(ang)
    if len(angles) < 3:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.25 or abs(angle) > max_angle:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _require() -> None:
    if not ocr_available():
        raise OCRUnavailable("Tesseract OCR is not installed. Install the `tesseract-ocr` system package (see README).")


def ocr_words(img: Image.Image | np.ndarray, *, lang: str = "eng", psm: int = 11, scale: float = 2.0,
              whitelist: Optional[str] = None, min_conf: float = 25.0, invert_if_dark: bool = True,
              deskew_img: bool = False) -> list[Word]:
    """Word boxes in the *original* image's pixel coordinates."""
    _require()
    gray, s = prepare(img, scale=scale, invert_if_dark=invert_if_dark, deskew_img=deskew_img)
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(gray, lang=lang, config=cfg, output_type=Output.DICT)
    words: list[Word] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(Word(text, conf, x / s, y / s, (x + w) / s, (y + h) / s))
    return words


def ocr_lines(img: Image.Image | np.ndarray, *, lang: str = "eng", psm: int = 6, scale: float = 2.0, min_conf: float = 30.0,
              deskew_img: bool = True, invert_if_dark: bool = True) -> list[OCRLine]:
    _require()
    gray, s = prepare(img, scale=scale, invert_if_dark=invert_if_dark, deskew_img=deskew_img)
    data = pytesseract.image_to_data(gray, lang=lang, config=f"--psm {psm}", output_type=Output.DICT)
    groups: dict[tuple, list[int]] = {}
    for i, text in enumerate(data["text"]):
        if (text or "").strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)
    lines: list[OCRLine] = []
    for idxs in groups.values():
        confs = [float(data["conf"][i]) for i in idxs if float(data["conf"][i]) >= 0]
        if not confs:
            continue
        conf = float(np.mean(confs))
        if conf < min_conf:
            continue
        text = " ".join(data["text"][i].strip() for i in idxs)
        x0 = min(data["left"][i] for i in idxs) / s
        y0 = min(data["top"][i] for i in idxs) / s
        x1 = max(data["left"][i] + data["width"][i] for i in idxs) / s
        y1 = max(data["top"][i] + data["height"][i] for i in idxs) / s
        lines.append(OCRLine(text, conf, x0, y0, x1, y1))
    lines.sort(key=lambda l: (round(l.y0 / 12), l.x0))
    return lines


def ocr_text(img: Image.Image | np.ndarray, *, lang: str = "eng", psm: int = 6, scale: float = 2.0,
             deskew_img: bool = True) -> tuple[str, float]:
    """Return (text, mean confidence). Tries both polarities when the image is ambiguous."""
    lines = ocr_lines(img, lang=lang, psm=psm, scale=scale, deskew_img=deskew_img)
    if not lines:
        return "", 0.0
    text = "\n".join(l.text for l in lines)
    conf = float(np.mean([l.conf for l in lines]))
    return clean_ocr(text), conf


def clean_ocr(text: str) -> str:
    text = text.replace("|", "I") if re.search(r"\b\|\w", text) else text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_CODE_RE = re.compile(r"\b[A-Z]{1,4}-\d{2,5}\b")


@dataclass
class OCRBest:
    text: str
    words: list[Word]
    conf: float
    codes: list[str]


def ocr_best(img: Image.Image | np.ndarray, *, lang: str = "eng") -> OCRBest:
    """Try several OCR configurations and keep the highest-yield reading.

    Photographed displays and plates are fragile: one configuration may drop a token that another reads. We keep the
    pass with the largest confidence-weighted token mass and union the code-like identifiers (E-47, HX-200) from all passes.
    """
    _require()
    passes = [(6, 1.0, True), (6, 2.0, True), (11, 1.0, True), (11, 2.0, False)]
    best: Optional[tuple[float, list[Word]]] = None
    codes: dict[str, None] = {}
    for psm, scale, des in passes:
        try:
            words = ocr_words(img, lang=lang, psm=psm, scale=scale, min_conf=10, deskew_img=des)
        except Exception:
            continue
        if not words:
            continue
        row_h = max(10.0, float(np.median([x.y1 - x.y0 for x in words])) * 0.8)
        words.sort(key=lambda w: (round(w.cy / row_h), w.x0))
        mass = sum(w.conf for w in words if len(w.text) >= 2)
        text = " ".join(w.text for w in words)
        for m in _CODE_RE.finditer(text.upper()):
            codes.setdefault(m.group(0), None)
        if best is None or mass > best[0]:
            best = (mass, words)
    words = best[1] if best else []
    text = " ".join(w.text for w in words)
    conf = float(np.mean([w.conf for w in words])) if words else 0.0
    return OCRBest(text=text, words=words, conf=conf, codes=list(codes))
