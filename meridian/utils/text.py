"""Tokenisation, light stemming, synonym canonicalisation, sentence splitting and small text helpers."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with without from into onto over under about as is are was were be been
    being do does did done have has had having it its this that these those there here which who whom whose what when where why how
    can could should would may might must shall will not no nor so than too very just also only more most some any each every all both
    such own same other another per via up down out off again further once i you he she we they me my your our their them us""".split()
)
NEGATION_CUES = frozenset(
    ["not", "no", "never", "without", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "won't", "shouldn't", "mustn't",
     "prohibited", "forbidden", "avoid", "neither", "nor", "n't", "disable", "disabled", "unsafe"]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
_ABBREV = ("fig", "no", "approx", "vs", "sec", "min", "max", "e.g", "i.e", "ref", "eq", "tab", "dr", "mr", "inc", "etc", "al")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("µ", "u").replace("μ", "u").replace("−", "-").replace("–", "-").replace("—", "-")
    return text


@lru_cache(maxsize=1)
def _synonym_map() -> dict[str, str]:
    path = RESOURCE_DIR / "synonyms.json"
    mapping: dict[str, str] = {}
    try:
        groups = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return mapping
    for group in groups:
        canon = group[0]
        for word in group:
            mapping[word] = canon
    return mapping


def stem(word: str) -> str:
    """A small, deterministic suffix stripper. Consistency matters more than linguistic correctness."""
    if len(word) <= 3 or "-" in word or "/" in word or any(ch.isdigit() for ch in word):
        return word
    w = word
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ment") and len(w) > 6:
        w = w[:-4]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith(("ches", "shes", "xes", "zes")) and len(w) > 5:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if w.endswith("ly") and len(w) > 5:
        w = w[:-2]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w


def tokenize(text: str, *, canon: bool = True, keep_stop: bool = False, split_compounds: bool = True) -> list[str]:
    """Lowercase tokens. Compound identifiers ('hx-200', 'm3/h') are emitted whole *and* as parts."""
    text = normalize_text(text).lower()
    syn = _synonym_map() if canon else {}
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        pieces = [tok]
        if split_compounds and re.search(r"[-./]", tok):
            parts = [p for p in re.split(r"[-./]", tok) if p]
            pieces.extend(p for p in parts if len(p) >= 2 or p.isdigit())
        for p in pieces:
            if not keep_stop and p in STOPWORDS:
                continue
            p = syn.get(p, p)
            out.append(stem(p))
    return out


def content_tokens(text: str) -> list[str]:
    """Stemmed, synonym-canonical, stop-word-free tokens (whole compound tokens only, no sub-parts)."""
    return tokenize(text, split_compounds=False)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that respects abbreviations, decimals and numbered steps."""
    pieces: list[str] = []
    for line in re.split(r"\n+", text or ""):
        line = line.strip()
        if not line:
            continue
        # protect abbreviations / decimals / numbered list markers
        protected = line
        for ab in _ABBREV:
            protected = re.sub(rf"\b({re.escape(ab)})\.", r"\1<DOT>", protected, flags=re.IGNORECASE)
        protected = re.sub(r"^(\d+)\.\s", r"\1<DOT> ", protected)
        protected = re.sub(r"(?<=\s)(\d{1,2})\.(?=\s+[A-Z])", r"\1<DOT>", protected)
        protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(])", protected)
        for part in parts:
            part = part.replace("<DOT>", ".").strip()
            if part:
                pieces.append(part)
    return pieces


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9%./\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def short_hash(*parts: str, n: int = 10) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8", "ignore")).hexdigest()
    return h[:n]


def slugify(text: str, max_len: int = 48) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "doc"


_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[(?:\d+(?:\s*[,;]\s*\d+)*)\]", "", text)


def find_citations(text: str) -> list[int]:
    found: list[int] = []
    for m in _CITATION_RE.finditer(text):
        found.extend(int(x) for x in re.findall(r"\d+", m.group(1)))
    return found


def truncate(text: str, n: int = 240) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def has_negation(text: str) -> bool:
    toks = re.findall(r"[a-z']+", normalize_text(text).lower())
    return any(t in NEGATION_CUES for t in toks)
