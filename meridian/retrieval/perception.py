"""Image-query perception: turn an uploaded photo/screenshot into structured evidence (text, codes, models, descriptor)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from ..ingest import figures as figmod
from ..ingest import ocr as ocrmod
from ..ingest.common import extract_models
from ..utils.text import truncate


class PerceptionError(ValueError):
    pass


@dataclass
class Perception:
    kind: str = "unknown"
    text: str = ""
    codes: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    descriptor: Optional[np.ndarray] = None
    chart: Optional[dict] = None
    ocr_conf: float = 0.0
    size: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {"kind": self.kind, "text": self.text, "codes": self.codes, "models": self.models, "ocr_conf": round(self.ocr_conf, 1),
                "size": list(self.size), "notes": self.notes, "chart": self.chart}


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise PerceptionError("The uploaded image is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise PerceptionError("The uploaded file is not a readable image (PNG, JPEG, WebP, BMP or TIFF).") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 3000:
        img.thumbnail((3000, 3000))
    if min(img.size) < 40:
        raise PerceptionError("The image is too small to analyse.")
    return img


def perceive(data: bytes, lang: str = "eng") -> Perception:
    img = load_image(data)
    p = Perception(size=img.size)
    if not ocrmod.ocr_available():
        p.notes.append("OCR engine unavailable: only visual similarity can be used.")
        p.kind = figmod.classify_figure(img)
        p.descriptor = figmod.visual_descriptor(img)
        return p
    an = figmod.analyze_figure(img, lang=lang)
    p.kind, p.text, p.codes, p.descriptor = an.kind, an.ocr_text, an.codes, an.descriptor
    p.models = extract_models(an.ocr_text.upper())
    best_conf = [w.conf for w in an.words]
    p.ocr_conf = float(np.mean(best_conf)) if best_conf else 0.0
    if an.chart is not None:
        p.chart = an.chart.to_meta()
    if not p.text.strip():
        p.notes.append("No text could be read from the image; relying on visual similarity only.")
    elif p.ocr_conf < 55:
        p.notes.append(f"Low OCR confidence ({p.ocr_conf:.0f}%): read '{truncate(p.text, 80)}'. Verify the alarm code on the display.")
    return p
