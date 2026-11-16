"""Typed data contracts shared by ingestion, indexing, retrieval, the agent, and the API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    TABLE_ROW = "table_row"
    FIGURE = "figure"
    CHART = "chart"
    WARNING = "warning"
    OCR_PAGE = "ocr_page"
    IMAGE = "image"


VISUAL_MODALITIES = {Modality.FIGURE, Modality.CHART, Modality.IMAGE}


class BBox(BaseModel):
    """Bounding box in PDF points, origin top-left (pdfplumber convention)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class Chunk(BaseModel):
    id: str
    doc_id: str
    page: int = 1
    modality: Modality
    text: str = Field(description="Contextualised retrieval text (what gets indexed).")
    content: str = Field(description="Clean display/evidence text (what the answer may quote).")
    section: str = ""
    section_no: str = ""
    parent_id: Optional[str] = None
    bbox: Optional[BBox] = None
    meta: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    image_path: Optional[str] = None
    visual_vec: Optional[list[float]] = None

    def public(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"visual_vec"})
        d["modality"] = self.modality.value
        return d


class Document(BaseModel):
    id: str
    title: str
    filename: str
    doc_type: str = "manual"  # manual | bulletin | log | card | image | notes
    pages: int = 1
    models: list[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    revision: Optional[str] = None
    sha256: str = ""
    added_at: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class Quantity(BaseModel):
    value: float
    unit: Optional[str] = None  # canonical unit string, e.g. "nm", "bar", "m3/h"
    raw: str = ""
    start: int = 0
    end: int = 0
    approximate: bool = False


class Fact(BaseModel):
    """A structured (model, subject, attribute, value, unit) assertion mined from a table row or chart."""

    model: str = ""
    subject: str
    attribute: str
    value: float
    unit: Optional[str] = None
    display: str = ""
    chunk_id: str
    doc_id: str
    doc_type: str = "manual"
    doc_title: str = ""
    effective_date: Optional[str] = None
    approximate: bool = False
    revised: bool = False  # value comes from a "revised/new" column
    previous: bool = False  # value comes from a "previous/old" column


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    modality: str
    section: str = ""
    snippet: str = ""
    bbox: Optional[list[float]] = None
    page_size: Optional[list[float]] = None
    via: str = "retrieval"  # retrieval | crossref | warning | tool | calculator
    score: float = 0.0
    has_image: bool = False


class ClaimVerdict(BaseModel):
    text: str
    verdict: str  # supported | partial | unsupported | contradicted
    score: float
    cited: list[int] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    numeric: bool = False


class ConflictNote(BaseModel):
    subject: str
    model: str
    winner: str
    loser: str
    winner_value: str
    loser_value: str
    rule: str
    winner_chunk: str
    loser_chunk: str


class TraceEvent(BaseModel):
    step: str
    title: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ms: float = 0.0
    status: str = "ok"  # ok | warn | fail


class AgentResult(BaseModel):
    question: str
    answer: str
    abstained: bool = False
    abstain_reason: str = ""
    confidence: float = 0.0
    faithfulness: float = 1.0
    intent: str = "general"
    citations: list[Citation] = Field(default_factory=list)
    claims: list[ClaimVerdict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    conflicts: list[ConflictNote] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    mode: str = "policy"
    perception: dict[str, Any] = Field(default_factory=dict)
