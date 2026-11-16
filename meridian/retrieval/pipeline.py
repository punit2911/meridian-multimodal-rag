"""Hybrid retrieval pipeline.

  analyze -> multi-channel candidates (BM25, dense, char-ngram, entity, fact, visual, PRF-BM25)
          -> weighted Reciprocal Rank Fusion -> graph (Personalized PageRank) expansion
          -> late-interaction (MaxSim) rerank + learned linear ranker -> MMR diversification

Every technique has a switch in `RetrievalConfig`, which is what the ablation study in `meridian.eval` toggles.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..index.lexical import top_k
from ..kb import KnowledgeBase
from ..schemas import Chunk, Modality
from ..utils.text import tokenize
from .analyzer import Intent, QueryAnalysis
from .perception import Perception
from .rerank import (FEATURES, LateInteraction, LearnedRanker, feature_vector, linear_default_score, modality_prior)


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_char: bool = True
    use_entity: bool = True
    use_fact: bool = True
    use_visual: bool = True
    use_prf: bool = True
    use_graph: bool = True
    use_late: bool = True
    use_ltr: bool = True
    use_mmr: bool = True
    model_filter: bool = True
    pool: int = 60
    rerank_depth: int = 40
    rrf_k: int = 60
    mmr_lambda: float = 0.82
    channel_weights: dict = field(default_factory=lambda: {
        "bm25": 1.0, "dense": 0.55, "char": 0.30, "entity": 0.9, "fact": 1.0, "visual": 1.0, "prf": 0.45, "graph": 0.5})


def preset(name: str) -> RetrievalConfig:
    """Named configurations used by the benchmark."""
    off = dict(use_bm25=False, use_dense=False, use_char=False, use_entity=False, use_fact=False, use_visual=False, use_prf=False,
               use_graph=False, use_late=False, use_ltr=False, use_mmr=False)
    table = {
        "bm25": dict(off, use_bm25=True, model_filter=False),
        "dense": dict(off, use_dense=True, model_filter=False),
        "hybrid_rrf": dict(off, use_bm25=True, use_dense=True, use_char=True, model_filter=False),
        "hybrid+entity+fact": dict(off, use_bm25=True, use_dense=True, use_char=True, use_entity=True, use_fact=True, model_filter=True),
        "+prf+graph": dict(off, use_bm25=True, use_dense=True, use_char=True, use_entity=True, use_fact=True, use_prf=True, use_graph=True, use_visual=True),
        "+late_rerank": dict(off, use_bm25=True, use_dense=True, use_char=True, use_entity=True, use_fact=True, use_prf=True, use_graph=True, use_visual=True,
                             use_late=True),
        "full": {},
    }
    return RetrievalConfig(**table[name])


@dataclass
class Candidate:
    idx: int
    chunk: Chunk
    ranks: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    rrf: float = 0.0
    feats: dict = field(default_factory=dict)
    final: float = 0.0
    via: str = "retrieval"
    reasons: list = field(default_factory=list)

    def channels(self) -> list[str]:
        return sorted(self.ranks, key=lambda k: self.ranks[k])


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    candidates: list[Candidate]
    timings: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    query_terms: dict = field(default_factory=dict)
    prf_terms: list = field(default_factory=list)
    fact_hits: list = field(default_factory=list)
    visual_used: bool = False


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, ranker: Optional[LearnedRanker] = None):
        self.kb = kb
        self.ranker = ranker if ranker is not None else LearnedRanker.load()
        self._li_for: Optional[int] = None
        self._li: Optional[LateInteraction] = None

    # ------------------------------------------------------------------------------------------ helpers
    def _late(self) -> LateInteraction:
        ix = self.kb.ix
        if self._li is None or self._li_for != id(ix):
            self._li = LateInteraction(ix.chunks, ix.dense.backend, ix.bm25)
            self._li_for = id(ix)
        return self._li

    def _query_terms(self, a: QueryAnalysis) -> dict[str, float]:
        terms: dict[str, float] = defaultdict(float)
        for t in a.index_tokens:
            terms[t] += 1.0
        for e in a.codes + a.models:
            for t in tokenize(e):
                terms[t] += 0.8
            terms[e.lower()] += 1.4
        return dict(terms)

    # ------------------------------------------------------------------------------------------ main
    def retrieve(self, analysis: QueryAnalysis, cfg: Optional[RetrievalConfig] = None, perception: Optional[Perception] = None,
                 exclude_models: bool = False) -> RetrievalResult:
        with self.kb._lock:
            return self._retrieve_locked(analysis, cfg, perception, exclude_models)

    def _retrieve_locked(self, analysis: QueryAnalysis, cfg: Optional[RetrievalConfig] = None, perception: Optional[Perception] = None,
                         exclude_models: bool = False) -> RetrievalResult:
        cfg = cfg or RetrievalConfig()
        ix = self.kb.ix
        res = RetrievalResult(analysis=analysis, candidates=[])
        n = len(ix.chunks)
        if n == 0:
            return res
        t_all = time.perf_counter()
        mask = None
        if cfg.model_filter and analysis.models and not exclude_models:
            mask = self.kb.mask(models=analysis.models)
        qterms = self._query_terms(analysis)
        res.query_terms = qterms
        query_text = analysis.question + (" " + analysis.image_text if analysis.image_text else "")
        channels: dict[str, list[tuple[int, float]]] = {}

        t = time.perf_counter()
        bm = ix.bm25.score(qterms) if (cfg.use_bm25 or cfg.use_prf or cfg.use_graph) else np.zeros(n)
        if cfg.use_bm25:
            channels["bm25"] = top_k(bm, cfg.pool, mask)
        res.timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense_limit = n if ix.dense.vector_store == "memory" else max(cfg.pool, cfg.rerank_depth)
        dense_hits = ix.dense.search(query_text, dense_limit, analysis.models if mask is not None else None) \
            if (cfg.use_dense or cfg.use_ltr or cfg.use_late) else []
        dense = dict(dense_hits)
        if cfg.use_dense:
            channels["dense"] = [(i, score) for i, score in dense_hits if (mask is None or mask[i]) and score >= 0.02][:cfg.pool]
        res.timings["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        char = ix.char.score(query_text) if (cfg.use_char or cfg.use_ltr) else np.zeros(n)
        if cfg.use_char:
            channels["char"] = top_k(char, cfg.pool, mask, min_score=0.02)
        res.timings["char_ms"] = (time.perf_counter() - t) * 1000

        # entity channel: exact alarm-code / model matching
        ent_scores = np.zeros(n)
        if cfg.use_entity and (analysis.codes or analysis.models):
            for c in analysis.codes:
                for i in ix.entity_map.get(c, []):
                    ent_scores[i] += 1.0 / len(analysis.codes)
            for i in range(n):
                if ent_scores[i] > 0 or not analysis.codes:
                    d = self.kb.doc(ix.chunks[i].doc_id)
                    if d and analysis.models and set(analysis.models) & set(d.models):
                        ent_scores[i] += 0.3 if analysis.codes else 0.05
            if analysis.codes:
                for i in range(n):  # a code hit is only meaningful inside a table row / text, not as noise elsewhere
                    if ent_scores[i] > 0 and ix.chunks[i].modality == Modality.TABLE:
                        ent_scores[i] *= 0.6
            channels["entity"] = top_k(ent_scores, cfg.pool, mask)

        # fact channel: structured value lookups
        fact_chunk: dict[str, float] = {}
        if cfg.use_fact:
            t = time.perf_counter()
            hits = ix.facts.search(analysis.tokens, models=analysis.models or None, unit_dim=None, min_score=0.45, include_approx=True)
            res.fact_hits = hits[:12]
            for f, sc in hits:
                fact_chunk[f.chunk_id] = max(fact_chunk.get(f.chunk_id, 0.0), sc)
            fs = np.zeros(n)
            for cid, sc in fact_chunk.items():
                j = ix.by_id.get(cid)
                if j is not None:
                    fs[j] = sc
            channels["fact"] = top_k(fs, cfg.pool, mask)
            res.timings["fact_ms"] = (time.perf_counter() - t) * 1000

        # visual channel
        vis = np.zeros(n)
        if cfg.use_visual and perception is not None and perception.descriptor is not None and perception.kind in {"diagram", "chart", "document"}:
            vis = ix.visual.score(perception.descriptor)
            ranked = top_k(vis, 10, mask, min_score=0.5)
            if ranked:
                channels["visual"] = ranked
                res.visual_used = True

        # pseudo-relevance feedback (RM3-flavoured): expand with discriminative terms from the top first-pass chunks
        prf = np.zeros(n)
        if cfg.use_prf and cfg.use_bm25 and channels.get("bm25"):
            top = [i for i, _ in channels["bm25"][:3]]
            cand_terms: Counter = Counter()
            for i in top:
                for term, w in ix.bm25.top_terms(i, 10):
                    if term not in qterms and len(term) > 2 and not term.isdigit():
                        cand_terms[term] += w
            expansion = {t_: 0.25 for t_, _ in cand_terms.most_common(6)}
            res.prf_terms = list(expansion)
            if expansion:
                prf = ix.bm25.score({**qterms, **expansion})
                channels["prf"] = top_k(prf, cfg.pool, mask)

        # fusion (weighted RRF)
        rrf: dict[int, float] = defaultdict(float)
        cands: dict[int, Candidate] = {}
        for name, ranked in channels.items():
            w = cfg.channel_weights.get(name, 1.0)
            for rank, (i, sc) in enumerate(ranked, 1):
                rrf[i] += w / (cfg.rrf_k + rank)
                c = cands.setdefault(i, Candidate(idx=i, chunk=ix.chunks[i]))
                c.ranks[name] = rank
                c.scores[name] = sc

        # graph expansion (Personalized PageRank from the current top)
        if cfg.use_graph and rrf:
            seeds = sorted(rrf, key=rrf.get, reverse=True)[:6]
            tot = sum(rrf[i] for i in seeds) or 1.0
            ppr = ix.graph.ppr({ix.chunks[i].id: rrf[i] / tot for i in seeds}, seed_entities=analysis.codes + analysis.models)
            if ppr:
                ranked = sorted(((ix.by_id[cid], v) for cid, v in ppr.items() if cid in ix.by_id), key=lambda x: -x[1])
                if mask is not None:
                    ranked = [(i, v) for i, v in ranked if mask[i]]
                ranked = ranked[:cfg.pool // 2]
                w = cfg.channel_weights["graph"]
                for rank, (i, sc) in enumerate(ranked, 1):
                    rrf[i] += w / (cfg.rrf_k + rank)
                    c = cands.setdefault(i, Candidate(idx=i, chunk=ix.chunks[i]))
                    c.ranks["graph"] = rank
                    c.scores["graph"] = sc
                    if "bm25" not in c.ranks and "dense" not in c.ranks:
                        c.via = "graph"
        if not cands:
            return res
        for i, c in cands.items():
            c.rrf = rrf[i]

        ordered = sorted(cands.values(), key=lambda c: -c.rrf)[:cfg.rerank_depth]
        max_rrf = max(c.rrf for c in ordered) or 1.0
        max_bm = float(bm.max()) or 1.0
        max_graph = max((c.scores.get("graph", 0.0) for c in ordered), default=0.0) or 1.0

        t = time.perf_counter()
        li = self._late() if (cfg.use_late or cfg.use_ltr) else None
        q_late_terms = list(dict.fromkeys(analysis.tokens + [e.lower() for e in analysis.codes + analysis.models]))
        qcodes, qmodels = set(analysis.codes), set(analysis.models)
        qtok_set = set(analysis.tokens)
        for c in ordered:
            ch = c.chunk
            f: dict[str, float] = {}
            f["rrf"] = c.rrf / max_rrf
            f["bm25"] = float(bm[c.idx]) / max_bm
            f["dense"] = float(max(dense.get(c.idx, 0.0), 0.0))
            f["char"] = float(max(char[c.idx], 0.0))
            late, per_term = li.score(q_late_terms, c.idx) if li is not None and q_late_terms else (0.0, {})
            f["late"] = late
            ent_hit = 0.0
            if qcodes:
                ent_hit += 0.7 * (len(qcodes & set(ch.entities)) / len(qcodes))
            d = self.kb.doc(ch.doc_id)
            mm = 1.0 if (d and qmodels and qmodels & set(d.models)) else 0.0
            if qmodels:
                ent_hit += 0.3 * mm
            f["entity"] = ent_hit
            f["model_match"] = mm if qmodels else 0.0
            f["fact"] = fact_chunk.get(ch.id, 0.0)
            f["modality"] = modality_prior(analysis.intent, ch, analysis)
            ctoks = set(tokenize(ch.content + " " + ch.section))
            f["q_cov"] = (sum(1 for t_ in qtok_set if t_ in ctoks) / len(qtok_set)) if qtok_set else 0.0
            sec = set(tokenize(ch.section + " " + str(ch.meta.get("subject", "")) + " " + str(ch.meta.get("caption", ""))))
            f["section_overlap"] = (len(sec & qtok_set) / max(len(sec), 1)) if sec else 0.0
            f["is_row"] = 1.0 if ch.modality == Modality.TABLE_ROW else 0.0
            f["is_warning"] = 1.0 if ch.modality == Modality.WARNING else 0.0
            f["is_bulletin"] = 1.0 if (d and d.doc_type == "bulletin") else 0.0
            f["graph"] = c.scores.get("graph", 0.0) / max_graph
            f["visual"] = float(max(vis[c.idx], 0.0)) if res.visual_used else 0.0
            f["prf"] = float(prf[c.idx]) / (float(prf.max()) or 1.0) if prf.any() else 0.0
            c.feats = f
            c.reasons = [k for k in ("bm25", "dense", "char", "entity", "fact", "visual", "prf", "graph") if k in c.ranks]
            if cfg.use_ltr and self.ranker is not None and list(self.ranker.features) == FEATURES:
                c.final = self.ranker.score(feature_vector(f))
            elif cfg.use_late:
                c.final = linear_default_score(f)
            else:
                c.final = f["rrf"]
        res.timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        ordered.sort(key=lambda c: -c.final)

        if cfg.use_mmr:
            ordered = self._mmr(ordered, cfg.mmr_lambda)
        res.candidates = ordered
        res.timings["total_ms"] = (time.perf_counter() - t_all) * 1000
        return res

    def _mmr(self, ordered: list[Candidate], lam: float, keep: int = 24) -> list[Candidate]:
        """Maximal Marginal Relevance on dense vectors; only near-duplicates of the same modality are penalised."""
        if len(ordered) <= 2:
            return ordered
        vectors = self.kb.ix.dense.vectors([c.idx for c in ordered[:keep]])
        if not vectors:
            return ordered
        top = ordered[:keep]
        rest = ordered[keep:]
        hi = max(c.final for c in top) or 1.0
        remaining = list(top)
        chosen: list[Candidate] = []
        while remaining:
            best, best_val = None, -1e9
            for c in remaining:
                rel = c.final / hi
                pen = 0.0
                for s in chosen:
                    if s.chunk.modality == c.chunk.modality and s.chunk.doc_id == c.chunk.doc_id:
                        a, b = vectors.get(c.idx), vectors.get(s.idx)
                        sim = float(a @ b) if a is not None and b is not None and a.size > 1 else 0.0
                        pen = max(pen, sim if sim > 0.88 else 0.0)
                val = lam * rel - (1 - lam) * pen
                if val > best_val:
                    best, best_val = c, val
            chosen.append(best)
            remaining.remove(best)
        return chosen + rest
