"""Query understanding: entities (models, alarm codes), intent, quantities, premises, unknown-model detection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..schemas import Quantity
from ..utils import units as U
from ..utils.text import content_tokens, normalize_text, tokenize


class Intent(str, Enum):
    SPEC = "spec_lookup"
    PROCEDURE = "procedure"
    TROUBLESHOOT = "troubleshoot"
    FIGURE = "figure_find"
    ANALYTICS = "analytics"
    GENERAL = "general"


_MODEL_RE = re.compile(r"\b([A-Za-z]{1,4})[-\s]?(\d{3,4})\b")
_CODE_RE = re.compile(r"\b[Ee][-\s]?(\d{2})\b")
_ANALYTICS_RE = re.compile(
    r"\bhow many (?:times|events|alarms|records|entries|incidents|failures)\b|\btotal\b|\baverage\b|\bthe mean\b|\bmean (?:downtime|value|time)\b|\bsum\b|\bmost (?:often|common|frequent|frequently)\b|"
    r"\boccurs? most\b|\bhighest (?:count|number)\b|\bin the (?:maintenance )?(?:log|logs|records|data|dataset)\b|\blogged\b|\brecorded\b|\bcount of\b", re.I)
_FIGURE_RE = re.compile(r"\b(show me|diagram|figure|schematic|drawing|picture|exploded|illustration|photo|where can i find|which figure)\b", re.I)
_PROCEDURE_RE = re.compile(r"\b(how (?:do|can|should|to) (?:i|we|you)?|steps?|procedure|instructions?|walk me through|how is .* (?:done|replaced|changed|performed))\b", re.I)
_PROC_VERBS = re.compile(r"\b(replace|change|install|remove|bleed|fit|start|commission|drain|refill|clean|reset|adjust|align|prime|assemble)\b", re.I)
_NOT_PROC = re.compile(r"\bhow (?:often|many|much|long|heavy|noisy|level|hot|high|big|large|frequently)\b", re.I)
_TROUBLE_RE = re.compile(
    r"\b(alarm|error|fault|trip|tripped|leak|leaking|vibrat\w*|overheat\w*|too hot|noisy|shut ?down|shuts? down|not working|cavitat\w*|overload|"
    r"loses? prime|what is wrong|what's wrong|what should i (?:do|check)|keeps? (?:stopping|failing|losing)|warning|displays?|shows?)\b", re.I)
_SPEC_RE = re.compile(
    r"\b(maximum|minimum|max|min|rated|capacity|weight|heavy|power|pressure|temperature|flow|speed|rpm|interval|how often|how frequently|torque|filtration|grade|"
    r"noise|noisy|level|volume|how much|how many|what is the|what's the|dimension|size|rating|setting|tank|head|delivery|oil|litres?|liters?)\b", re.I)
_YESNO_RE = re.compile(r"^\s*(is|are|does|do|can|did|will|should|was)\b|\b(correct|right|true|accurate)\s*\?|\bi (?:read|heard|was told)\b|\bthey said\b", re.I)
_QUESTION_WORDS = {"what", "which", "how", "does", "show", "tell", "give", "need", "want", "please", "much", "many", "often", "correct", "right", "true",
                   "read", "heard", "use", "used", "using", "get", "make", "find", "know", "mean", "means", "meant"}


@dataclass
class QueryAnalysis:
    question: str
    models: list[str] = field(default_factory=list)  # explicit, known
    unknown_models: list[str] = field(default_factory=list)  # model-looking names with no documentation
    codes: list[str] = field(default_factory=list)
    intent: Intent = Intent.GENERAL
    quantities: list[Quantity] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)  # content tokens (stemmed, canonical, compounds not split)
    index_tokens: list[str] = field(default_factory=list)  # tokens including compound sub-parts, for BM25
    wants_visual: bool = False
    yes_no: bool = False
    premise: list[Quantity] = field(default_factory=list)  # numbers the user asserts (only for yes/no style questions)
    unit_dim: Optional[str] = None
    models_from_image: bool = False
    needs_model: bool = False  # spec/procedure-like question that does not pin a product
    image_text: str = ""

    def summary(self) -> dict:
        return {"intent": self.intent.value, "models": self.models, "unknown_models": self.unknown_models, "codes": self.codes,
                "quantities": [U.format_quantity(q.value, q.unit) for q in self.quantities], "wants_visual": self.wants_visual,
                "yes_no": self.yes_no, "needs_model": self.needs_model, "models_from_image": self.models_from_image}


def normalize_model(prefix: str, digits: str) -> str:
    return f"{prefix.upper()}-{digits}"


def detect_models(text: str, known: set[str]) -> tuple[list[str], list[str]]:
    """Return (known models, unknown models that share a known product-line prefix)."""
    prefixes = {k.split("-")[0] for k in known}
    found, unknown = [], []
    for m in _MODEL_RE.finditer(normalize_text(text)):
        prefix, digits = m.group(1).upper(), m.group(2)
        if prefix not in prefixes:
            continue
        name = normalize_model(prefix, digits)
        if name in known:
            if name not in found:
                found.append(name)
        elif name not in unknown:
            unknown.append(name)
    return found, unknown


def detect_codes(text: str) -> list[str]:
    out = []
    for m in _CODE_RE.finditer(text):
        c = f"E-{m.group(1)}"
        if c not in out:
            out.append(c)
    return out


def classify_intent(question: str, codes: list[str]) -> Intent:
    q = question
    if _ANALYTICS_RE.search(q):
        return Intent.ANALYTICS
    if _FIGURE_RE.search(q) and not re.search(r"\bhow (?:do|to)\b", q, re.I):
        return Intent.FIGURE
    if codes or _TROUBLE_RE.search(q):
        if not (_NOT_PROC.search(q) and not codes):
            if re.search(r"\b(alarm|error|fault|code)\b|^\s*what should i do|what is wrong|what's wrong|vibrat|leak|overheat|too hot|shut|cavitat|loses? prime|overload", q, re.I) or codes:
                return Intent.TROUBLESHOOT
    if (_PROCEDURE_RE.search(q) or (_PROC_VERBS.search(q) and re.search(r"\bhow\b|\bsteps?\b", q, re.I))) and not _NOT_PROC.search(q):
        return Intent.PROCEDURE
    if _SPEC_RE.search(q) or _NOT_PROC.search(q):
        return Intent.SPEC
    return Intent.GENERAL


def analyze(question: str, known_models: set[str], *, image_models: Optional[list[str]] = None, image_codes: Optional[list[str]] = None,
            image_text: str = "") -> QueryAnalysis:
    q = normalize_text(question).strip()
    models, unknown = detect_models(q, known_models)
    codes = detect_codes(q)
    from_image = False
    if not models and image_models:
        models = [m for m in image_models if m in known_models]
        from_image = bool(models)
    if image_codes:
        for c in image_codes:
            if c.startswith("E-") and c not in codes:
                codes.append(c)
    intent = classify_intent(q + (" alarm" if image_codes and not codes else ""), codes) if not (image_codes and re.search(r"alarm|mean|do about", q, re.I)) else Intent.TROUBLESHOOT
    if image_codes and intent == Intent.GENERAL:
        intent = Intent.TROUBLESHOOT
    prefixes = {k.split("-")[0] for k in known_models}
    q_no_models = _MODEL_RE.sub(lambda m: " " if m.group(1).upper() in prefixes else m.group(0), q)  # only real product-line names, never "at 150"
    quantities = [x for x in U.extract_quantities(q_no_models) if not (x.unit is None and f"E-{int(x.value):02d}" in codes)]
    yes_no = bool(_YESNO_RE.search(q))
    premise = [x for x in quantities if x.unit is not None or yes_no] if yes_no else []
    base_tokens = [t for t in content_tokens(q) if t not in _QUESTION_WORDS]
    idx_tokens = [t for t in tokenize(q) if t not in _QUESTION_WORDS]
    if image_text:
        idx_tokens += tokenize(image_text)
        base_tokens += [t for t in content_tokens(image_text) if t not in base_tokens]
    dims = [U.unit_dimension(x.unit) for x in quantities if x.unit]
    needs_model = (not models and not unknown and len(known_models) > 1 and intent in (Intent.SPEC, Intent.PROCEDURE, Intent.TROUBLESHOOT))
    return QueryAnalysis(
        question=q, models=models, unknown_models=unknown, codes=codes, intent=intent, quantities=quantities, tokens=base_tokens,
        index_tokens=idx_tokens, wants_visual=bool(_FIGURE_RE.search(q)), yes_no=yes_no, premise=premise, unit_dim=dims[0] if dims else None,
        models_from_image=from_image, needs_model=needs_model, image_text=image_text,
    )
