"""Reranking: ColBERT-style late interaction (MaxSim over token vectors), feature extraction and the learned linear ranker."""
from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..schemas import Chunk, Modality
from ..utils.text import tokenize
from .analyzer import Intent, QueryAnalysis

FEATURES = [
    "rrf", "bm25", "dense", "char", "late", "entity", "fact", "modality", "model_match", "q_cov", "section_overlap",
    "is_row", "is_warning", "is_bulletin", "graph", "visual", "prf",
]

RESOURCE = Path(__file__).resolve().parent.parent / "resources" / "ltr_weights.json"


class LateInteraction:
    """MaxSim late-interaction scorer.

    Query and chunk are *bags of token vectors* (not single vectors). Each query token is matched to its best chunk token:
        exact stem match -> 1.0 | typo-level fuzzy match -> 0.75 | LSA term-vector cosine -> scaled similarity
    and the per-token maxima are combined with IDF weights. This keeps ColBERT's key property (fine-grained token-level
    matching) while remaining fully offline; with a neural backend the same code can consume contextual token embeddings.
    """

    def __init__(self, chunks: list[Chunk], dense_backend, bm25):
        self.chunks = chunks
        self.bm25 = bm25
        self.term_vecs: dict[str, np.ndarray] = {}
        vec = getattr(dense_backend, "vec", None)
        svd = getattr(dense_backend, "svd", None)
        if vec is not None and svd is not None:
            comps = svd.components_.T  # (n_ngrams, k)
            norms = np.linalg.norm(comps, axis=1, keepdims=True)
            comps = comps / np.maximum(norms, 1e-9)
            for term, j in vec.vocabulary_.items():
                if " " not in term:
                    self.term_vecs[term] = comps[j].astype(np.float32)
        self._cache: dict[int, tuple[list[str], np.ndarray]] = {}

    def _chunk_tokens(self, i: int) -> tuple[list[str], np.ndarray]:
        hit = self._cache.get(i)
        if hit is not None:
            return hit
        ch = self.chunks[i]
        toks = sorted(set(tokenize(ch.content + " " + ch.section + " " + " ".join(ch.entities))))
        mat = np.array([self.term_vecs[t] for t in toks if t in self.term_vecs], dtype=np.float32) if self.term_vecs else np.zeros((0, 1))
        self._cache[i] = (toks, mat)
        return toks, mat

    def score(self, q_terms: list[str], i: int, weights: Optional[dict[str, float]] = None) -> tuple[float, dict[str, float]]:
        toks, mat = self._chunk_tokens(i)
        tokset = set(toks)
        total_w, acc = 0.0, 0.0
        per_term: dict[str, float] = {}
        for t in q_terms:
            w = (weights or {}).get(t) or max(self.bm25.idf_of(t), 0.6)
            total_w += w
            if t in tokset:
                s = 1.0
            else:
                s = 0.0
                if len(t) >= 5:
                    near = difflib.get_close_matches(t, toks, n=1, cutoff=0.86)
                    if near:
                        s = 0.75
                if s == 0.0 and t in self.term_vecs and mat.size:
                    cos = float(np.max(mat @ self.term_vecs[t]))
                    s = max(0.0, (cos - 0.55) / 0.45) * 0.7
            per_term[t] = s
            acc += w * s
        return (acc / total_w if total_w else 0.0), per_term


def modality_prior(intent: Intent, ch: Chunk, analysis: QueryAnalysis) -> float:
    m = ch.modality
    if intent == Intent.SPEC:
        base = {Modality.TABLE_ROW: 1.0, Modality.OCR_PAGE: 0.7, Modality.TABLE: 0.45, Modality.CHART: 0.45, Modality.TEXT: 0.4,
                Modality.WARNING: 0.1, Modality.FIGURE: 0.15, Modality.IMAGE: 0.3}
    elif intent == Intent.TROUBLESHOOT:
        base = {Modality.TABLE_ROW: 1.0, Modality.OCR_PAGE: 0.6, Modality.TEXT: 0.5, Modality.TABLE: 0.35, Modality.WARNING: 0.25,
                Modality.IMAGE: 0.3, Modality.CHART: 0.1, Modality.FIGURE: 0.1}
    elif intent == Intent.PROCEDURE:
        base = {Modality.TEXT: 1.0, Modality.WARNING: 0.5, Modality.OCR_PAGE: 0.4, Modality.TABLE_ROW: 0.25, Modality.TABLE: 0.2,
                Modality.FIGURE: 0.2, Modality.CHART: 0.1, Modality.IMAGE: 0.1}
    elif intent == Intent.FIGURE:
        base = {Modality.FIGURE: 1.0, Modality.CHART: 1.0, Modality.IMAGE: 0.8, Modality.TEXT: 0.2, Modality.TABLE_ROW: 0.05,
                Modality.TABLE: 0.05, Modality.WARNING: 0.0, Modality.OCR_PAGE: 0.1}
    elif intent == Intent.ANALYTICS:
        base = {Modality.TABLE_ROW: 0.9 if ch.meta.get("kind") == "aggregate" else 0.2, Modality.TABLE: 1.0 if ch.meta.get("kind") == "schema" else 0.2}
        return base.get(m, 0.1)
    else:
        return 0.5
    return base.get(m, 0.3)


def default_weights() -> dict[str, float]:
    return {"rrf": 0.30, "bm25": 0.15, "dense": 0.10, "char": 0.04, "late": 0.28, "entity": 0.14, "fact": 0.18, "modality": 0.14,
            "model_match": 0.06, "q_cov": 0.12, "section_overlap": 0.08, "is_row": 0.0, "is_warning": 0.0, "is_bulletin": 0.02,
            "graph": 0.05, "visual": 0.12, "prf": 0.03}


@dataclass
class LearnedRanker:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    meta: dict = field(default_factory=dict)

    def score(self, vec: np.ndarray) -> float:
        z = float(((vec - self.mean) / self.scale) @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    @staticmethod
    def load(path: Path = RESOURCE) -> Optional["LearnedRanker"]:
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            return LearnedRanker(d["features"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), float(d["intercept"]), d.get("meta", {}))
        except (OSError, ValueError, KeyError):
            return None

    def save(self, path: Path = RESOURCE) -> None:
        Path(path).write_text(json.dumps({"features": self.features, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                                          "coef": self.coef.tolist(), "intercept": self.intercept, "meta": self.meta}, indent=1), encoding="utf-8")


def feature_vector(feats: dict[str, float], names: list[str] = FEATURES) -> np.ndarray:
    return np.array([float(feats.get(n, 0.0)) for n in names], dtype=np.float64)


def linear_default_score(feats: dict[str, float]) -> float:
    w = default_weights()
    return float(sum(w[k] * feats.get(k, 0.0) for k in w))
