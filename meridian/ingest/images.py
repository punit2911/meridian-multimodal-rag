"""Standalone image ingestion (photos of displays, data plates, diagrams, screenshots)."""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageOps

from ..config import Settings
from ..schemas import Chunk, Document, Modality
from . import figures as figmod
from .common import ChunkBuilder, extract_models


class ImageParser:
    def __init__(self, settings: Settings):
        self.s = settings

    def parse(self, path: Path, doc: Document) -> tuple[Document, list[Chunk]]:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        if max(img.size) > 3000:
            img.thumbnail((3000, 3000))
        figs_dir = self.s.data_dir / "figures" / doc.id
        figs_dir.mkdir(parents=True, exist_ok=True)
        out = figs_dir / "image.png"
        img.save(out, optimize=True)
        an = figmod.analyze_figure(img, lang=self.s.ocr_lang)
        builder = ChunkBuilder(doc)
        doc.doc_type = "image"
        doc.title = Path(path).stem.replace("_", " ")
        parts = [f"Image '{doc.title}' ({an.kind})."]
        if an.ocr_text:
            parts.append(f"Text visible in the image: {an.ocr_text}")
        meta = {"kind": an.kind, "ocr": an.ocr_text}
        modality = Modality.IMAGE
        if an.chart is not None:
            modality = Modality.CHART
            meta["chart"] = an.chart.to_meta()
            parts.append(an.chart.summary())
        ch = builder.add(modality=modality, content="\n".join(parts), meta=meta, image_path=str(out.relative_to(self.s.data_dir)),
                         visual_vec=an.descriptor, section="Image")
        doc.models = extract_models(doc.title + " " + an.ocr_text)
        doc.stats = {"tables": 0, "figures": 1, "charts": 1 if an.chart else 0, "facts": 0}
        doc.pages = 1
        return doc, builder.chunks
