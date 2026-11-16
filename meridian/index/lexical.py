"""Sparse retrieval channels: BM25 (with field boosts) and a character n-gram TF-IDF channel for OCR-noisy text."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ..schemas import Chunk
from ..utils.text import normalize_for_match, tokenize


def chunk_terms(chunk: Chunk) -> list[str]:
    """Index terms for a chunk: full contextual text, plus boosted heading/subject/entity terms (BM25F-lite)."""
    terms = tokenize(chunk.text)
    boost = tokenize(" ".join([chunk.section, str(chunk.meta.get("subject", "")), str(chunk.meta.get("caption", ""))] + chunk.entities))
    return terms + boost + boost  # field weight ~3x for headings / row subjects / entities


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.72):
        self.k1, self.b = k1, b
        self.n = len(chunks)
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(self.n, dtype=np.float64)
        for i, ch in enumerate(chunks):
            tf = Counter(chunk_terms(ch))
            dl[i] = sum(tf.values())
            for term, c in tf.items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i), cols.append(j), vals.append(float(c))
        self.vocab = vocab
        tf_mat = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, max(1, len(vocab))))
        self.avgdl = float(dl.mean()) if self.n else 1.0
        df = np.asarray((tf_mat > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
        tf_coo = tf_mat.tocoo()
        norm = self.k1 * (1.0 - self.b + self.b * dl[tf_coo.row] / max(self.avgdl, 1e-9))
        w = self.idf[tf_coo.col] * tf_coo.data * (self.k1 + 1.0) / (tf_coo.data + norm)
        self.W = sp.csc_matrix((w, (tf_coo.row, tf_coo.col)), shape=tf_mat.shape)

    def score(self, terms: dict[str, float]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        for term, weight in terms.items():
            j = self.vocab.get(term)
            if j is None:
                continue
            col = self.W.getcol(j)
            out[col.indices] += weight * col.data
        return out

    def explain(self, chunk_idx: int, terms: dict[str, float]) -> list[tuple[str, float]]:
        res = []
        for term, weight in terms.items():
            j = self.vocab.get(term)
            if j is None:
                continue
            v = self.W[chunk_idx, j]
            if v:
                res.append((term, float(weight * v)))
        return sorted(res, key=lambda x: -x[1])

    def idf_of(self, term: str) -> float:
        j = self.vocab.get(term)
        return float(self.idf[j]) if j is not None else 0.0

    def top_terms(self, chunk_idx: int, k: int = 8) -> list[tuple[str, float]]:
        row = self.W.getrow(chunk_idx) if False else self.W[chunk_idx, :]
        coo = row.tocoo()
        inv = {j: t for t, j in self.vocab.items()}
        pairs = sorted(((inv[j], float(v)) for j, v in zip(coo.col, coo.data)), key=lambda x: -x[1])
        return pairs[:k]


class CharNgramIndex:
    """Character n-gram TF-IDF: tolerant to OCR typos ('suctlon') and spacing/hyphen variants."""

    def __init__(self, chunks: list[Chunk]):
        self.n = len(chunks)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, lowercase=False, norm="l2")
        docs = [normalize_for_match(c.content + " " + c.section) for c in chunks]
        self.X = self.vec.fit_transform(docs) if docs else sp.csr_matrix((0, 0))

    def score(self, query: str) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        q = self.vec.transform([normalize_for_match(query)])
        return np.asarray((self.X @ q.T).todense()).ravel()


def zscore_filter(scores: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    return scores if mask is None else np.where(mask, scores, 0.0)


def top_k(scores: np.ndarray, k: int, mask: Optional[np.ndarray] = None, min_score: float = 1e-9) -> list[tuple[int, float]]:
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    if k >= len(s):
        order = np.argsort(-s)
    else:
        part = np.argpartition(-s, k)[:k]
        order = part[np.argsort(-s[part])]
    return [(int(i), float(s[i])) for i in order if np.isfinite(s[i]) and s[i] > min_score]
