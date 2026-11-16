"""Ingestion orchestrator: file -> (Document, chunks)."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..schemas import Chunk, Document
from ..utils.text import slugify
from .images import ImageParser
from .office import DocxParser, TabularParser, TextParser
from .pdf import PDFParser

SUPPORTED = {
    ".pdf": "pdf", ".docx": "docx", ".csv": "table", ".xlsx": "table", ".xlsm": "table", ".txt": "text", ".md": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".bmp": "image", ".tif": "image", ".tiff": "image",
}


class UnsupportedFileType(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_doc_id(path: Path) -> str:
    return slugify(path.stem)


def ingest_file(path: Path, settings: Settings, doc_id: str | None = None) -> tuple[Document, list[Chunk]]:
    path = Path(path)
    kind = SUPPORTED.get(path.suffix.lower())
    if kind is None:
        raise UnsupportedFileType(f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED))}")
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"File is missing or empty: {path.name}")
    doc = Document(id=doc_id or make_doc_id(path), title=path.stem.replace("_", " "), filename=path.name, sha256=sha256_file(path),
                   added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    parser = {"pdf": PDFParser, "docx": DocxParser, "table": TabularParser, "text": TextParser, "image": ImageParser}[kind](settings)
    try:
        doc, chunks = parser.parse(path, doc)
    except UnsupportedFileType:
        raise
    except Exception as exc:  # surface a clean, actionable error
        raise ValueError(f"Could not parse {path.name}: {type(exc).__name__}: {exc}") from exc
    if not chunks:
        raise ValueError(f"No content could be extracted from {path.name}.")
    finalize_fact_provenance(doc, chunks)
    doc.stats = {**doc.stats, "chunks": len(chunks)}
    return doc, chunks


def finalize_fact_provenance(doc: Document, chunks: list[Chunk]) -> None:
    """Parsers learn a document's type/date/title only after reading it; stamp the final values onto every mined fact."""
    for ch in chunks:
        for f in ch.meta.get("facts", []):
            f["doc_type"] = doc.doc_type
            f["doc_title"] = doc.title
            f["effective_date"] = doc.effective_date
            if not f.get("model") and len(doc.models) == 1:
                f["model"] = doc.models[0]
