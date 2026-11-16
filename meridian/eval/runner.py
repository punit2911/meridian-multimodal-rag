"""Benchmark runner: retrieval ablations, end-to-end answers, hallucination-injection benchmark, LTR training and calibration."""
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..agent.agent import Orchestrator
from ..agent.compose import TEMPLATE_TERMS, CiteBook
from ..kb import KnowledgeBase
from ..retrieval.analyzer import analyze
from ..retrieval.perception import perceive
from ..retrieval.pipeline import HybridRetriever, RetrievalConfig, preset
from ..retrieval.rerank import FEATURES, LearnedRanker, feature_vector
from ..verify.verifier import Verifier, faithfulness
from . import metrics as M

PRESETS = ["bm25", "dense", "hybrid_rrf", "hybrid+entity+fact", "+prf+graph", "+late_rerank", "full"]


# ------------------------------------------------------------------------------------------------ loading
def load_golden(eval_dir: Path) -> list[dict]:
    return json.loads((Path(eval_dir) / "golden_qa.json").read_text(encoding="utf-8"))["questions"]


def _image_bytes(eval_dir: Path, q: dict) -> Optional[bytes]:
    return (Path(eval_dir) / q["image"]).read_bytes() if q.get("image") else None


def _analysis_for(kb: KnowledgeBase, q: dict, eval_dir: Path):
    img = _image_bytes(eval_dir, q)
    per = perceive(img) if img else None
    a = analyze(q["question"], set(kb.known_models()), image_models=per.models if per else None, image_codes=per.codes if per else None,
                image_text=per.text if per else "")
    return a, per


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) == r.get(key) and r.get(key) is not None]
    if not vals:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")], "n": 0}
    lo, hi = M.bootstrap_ci(vals)
    return {"mean": round(sum(vals) / len(vals), 4), "ci": [round(lo, 3), round(hi, 3)], "n": len(vals)}


# ------------------------------------------------------------------------------------------------ retrieval
def retrieval_rows(kb: KnowledgeBase, questions: list[dict], eval_dir: Path, cfg: RetrievalConfig, retriever: Optional[HybridRetriever] = None) -> list[dict]:
    retriever = retriever or HybridRetriever(kb)
    rows = []
    for q in questions:
        if not q["gold_evidence"]:
            continue
        a, per = _analysis_for(kb, q, eval_dir)
        t = time.perf_counter()
        res = retriever.retrieve(a, cfg, per)
        ms = (time.perf_counter() - t) * 1000
        ranked = [c.chunk for c in res.candidates]
        pos = M.positives(kb.chunks, q["gold_evidence"])
        rows.append({"id": q["id"], "type": q["type"], "split": q["split"], "hit@1": M.hit_at_k(ranked, q["gold_evidence"], 1),
                     "hit@3": M.hit_at_k(ranked, q["gold_evidence"], 3), "hit@5": M.hit_at_k(ranked, q["gold_evidence"], 5),
                     "recall@5": M.recall_at_k(ranked, q["gold_evidence"], 5), "mrr": M.mrr(ranked, q["gold_evidence"], pos),
                     "ndcg@10": M.ndcg_at_k(ranked, q["gold_evidence"], 10, pos), "ms": ms})
    return rows


def summarize_retrieval(rows: list[dict]) -> dict:
    return {k: _agg(rows, k) for k in ("hit@1", "hit@3", "hit@5", "recall@5", "mrr", "ndcg@10", "ms")}


def ablation(kb: KnowledgeBase, questions: list[dict], eval_dir: Path, presets: list[str] = PRESETS, split: Optional[str] = None,
             retriever: Optional[HybridRetriever] = None, log: Callable[[str], None] = lambda s: None) -> dict:
    qs = [q for q in questions if split is None or q["split"] == split]
    out = {}
    for name in presets:
        log(f"  retrieval preset {name}")
        rows = retrieval_rows(kb, qs, eval_dir, preset(name), retriever)
        out[name] = summarize_retrieval(rows)
        out[name]["by_type"] = {t: _agg([r for r in rows if r["type"] == t], "hit@5")["mean"] for t in sorted({r["type"] for r in rows})}
    return out


# ------------------------------------------------------------------------------------------------ end to end
def end_to_end(kb: KnowledgeBase, agent: Orchestrator, questions: list[dict], eval_dir: Path, log: Callable[[str], None] = lambda s: None) -> dict:
    rows = []
    for q in questions:
        img = _image_bytes(eval_dir, q)
        t = time.perf_counter()
        try:
            r = agent.answer(q["question"], img)
            err = ""
        except Exception as exc:  # a benchmark must keep going and count the failure
            r, err = None, f"{type(exc).__name__}: {exc}"
        ms = (time.perf_counter() - t) * 1000
        if r is None:
            rows.append({"id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"], "error": err, "correct": 0.0, "abstained": 1.0,
                         "expect_abstain": float(q["expect_abstain"]), "confidence": 0.0, "faithfulness": 0.0, "ms": ms, "answer": "", "missing": []})
            continue
        if q["expect_abstain"]:
            correct, missing = (1.0 if r.abstained else 0.0), []
        else:
            ok, missing = (False, ["abstained"]) if r.abstained else M.answer_correct(r.answer, q["answer_facts"])
            correct = float(ok)
        cited_docs = {c.doc_id for c in r.citations}
        gold_docs = {e["doc"] for e in q["gold_evidence"]}
        rows.append({"id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"], "answer": r.answer, "correct": correct,
                     "missing": missing, "abstained": float(r.abstained), "expect_abstain": float(q["expect_abstain"]), "confidence": r.confidence,
                     "faithfulness": r.faithfulness, "citation_recall": (float(gold_docs <= cited_docs) if gold_docs and not r.abstained else float("nan")),
                     "ms": ms, "error": err})
    return {"rows": rows, "summary": summarize_e2e(rows)}


def summarize_e2e(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["expect_abstain"]]
    unans = [r for r in rows if r["expect_abstain"]]
    probs = [r["confidence"] for r in rows if not r["abstained"]]
    outs = [int(r["correct"]) for r in rows if not r["abstained"]]
    e, bins = M.ece(probs, outs)
    abst = [r for r in rows if r["abstained"]]
    return {
        "n": len(rows),
        "answer_correctness": _agg(answerable, "correct"),
        "abstention_recall": _agg(unans, "correct"),  # of the unanswerable questions, share correctly declined
        "abstention_precision": round(sum(r["expect_abstain"] for r in abst) / len(abst), 4) if abst else 1.0,
        "false_abstention_rate": round(sum(r["abstained"] for r in answerable) / max(len(answerable), 1), 4),
        "faithfulness": _agg([r for r in rows if not r["abstained"]], "faithfulness"),
        "citation_recall": _agg(answerable, "citation_recall"),
        "ece": round(e, 4), "brier": round(M.brier(probs, outs), 4) if probs else float("nan"), "reliability": bins,
        "latency_ms": _agg(rows, "ms"),
        "by_type": {t: _agg([r for r in rows if r["type"] == t], "correct")["mean"] for t in sorted({r["type"] for r in rows})},
        "errors": [r["id"] for r in rows if r.get("error")],
    }


# ------------------------------------------------------------------------------------------------ hallucination injection
_NUM = re.compile(r"(?<![\w\[\-])(\d+(?:\.\d+)?)(?![\d\]])")


def _perturb_number(ans: str, rng: random.Random) -> Optional[str]:
    ms = [m for m in _NUM.finditer(ans)]
    if not ms:
        return None
    m = rng.choice(ms)
    v = float(m.group(1))
    nv = v * rng.choice([1.5, 0.5, 2.0]) if v != 0 else 7.0
    if abs(nv - v) < 1e-9:
        nv = v + 11
    s = f"{nv:.0f}" if "." not in m.group(1) else f"{nv:.1f}"
    return ans[:m.start()] + s + ans[m.end():]


def _fabricate(ans: str, rng: random.Random) -> str:
    extra = rng.choice([
        "The unit also requires a monthly recalibration with a laser alignment kit",
        "Operators must wear a certified respirator during every inspection",
        "The warranty covers this component for seven years",
        "A firmware update released in 2027 removes this limitation"])
    n = max(1, len(re.findall(r"\[\d+\]", ans)))
    return ans.rstrip() + f" {extra} [{rng.randint(1, n)}]."


def _flip_negation(ans: str) -> Optional[str]:
    for pat, rep in ((r"\b(is|are) (?!not\b)", r"\1 not "), (r"\bmust (?!not\b)", "must not "), (r"\bshould (?!not\b)", "should not "), (r"\blists\b", "does not list"),
                     (r"\bDo not\b", "Do"), (r"\bNo:", "Yes:")):
        new, n = re.subn(pat, rep, ans, count=1)
        if n:
            return new
    return None


def _wrong_entity(ans: str, rng: random.Random) -> Optional[str]:
    for pat, pool in ((r"HX-\d{3}", ["HX-300", "HX-700", "HX-100"]), (r"\bE-\d{2}\b", ["E-99", "E-13", "E-88"]), (r"\b20\d\d-\d\d-\d\d\b", ["2019-01-05", "2031-09-09"]),
                      (r"SB-\d{4}-\d{2}", ["SB-2024-99", "SB-2031-01"])):
        ms = list(re.finditer(pat, ans))
        if ms:
            m = rng.choice(ms)
            return ans[:m.start()] + rng.choice(pool) + ans[m.end():]
    return None


def hallucination_benchmark(kb: KnowledgeBase, e2e_rows: list[dict], agent: Orchestrator, questions: list[dict], eval_dir: Path, seed: int = 11) -> dict:
    """Take verified-clean answers, inject one hallucination of a known kind, and measure whether the verifier flags it.
    Also reports the false-positive rate on the clean originals."""
    rng = random.Random(seed)
    verifier = Verifier(idf=lambda tok: kb.ix.bm25.idf_of(tok))
    byid = {q["id"]: q for q in questions}
    kinds = {"numeric": 0, "fabricated": 0, "negation": 0, "entity": 0}
    caught = dict.fromkeys(kinds, 0)
    clean_n = clean_fp = 0
    for r in e2e_rows:
        if r["abstained"] or r["expect_abstain"] or r["correct"] < 1:
            continue
        q = byid[r["id"]]
        res = agent.answer(q["question"], _image_bytes(eval_dir, q))
        if res.abstained or not res.citations:
            continue
        book = CiteBook(kb)
        for c in res.citations:
            ch = kb.chunk(c.chunk_id)
            if ch:
                book.cite(ch)
        ev = book.evidence_docs()
        ign = Orchestrator._ignore_terms(type("C", (), {"kind": "spec"})())
        clean = verifier.verify(res.answer, ev, ignore_terms=ign, echo_text=q["question"])
        clean_n += 1
        if faithfulness(clean) < 1.0 - 1e-9:
            clean_fp += 1
            continue  # only measure detection on originals that verify cleanly
        variants = {"numeric": _perturb_number(res.answer, rng), "fabricated": _fabricate(res.answer, rng),
                    "negation": _flip_negation(res.answer), "entity": _wrong_entity(res.answer, rng)}
        for kind, text in variants.items():
            if text is None or text == res.answer:
                continue
            kinds[kind] += 1
            v = verifier.verify(text, ev, ignore_terms=ign, echo_text=q["question"])
            if faithfulness(v) < 1.0 - 1e-9 or any(x.verdict in ("unsupported", "contradicted") for x in v):
                caught[kind] += 1
    total = sum(kinds.values())
    return {"clean_answers": clean_n, "false_positive_rate": round(clean_fp / max(clean_n, 1), 4),
            "injected": kinds, "detected": caught,
            "detection_rate": {k: round(caught[k] / kinds[k], 4) if kinds[k] else float("nan") for k in kinds},
            "overall_detection_rate": round(sum(caught.values()) / max(total, 1), 4), "total_injected": total}


# ------------------------------------------------------------------------------------------------ LTR training
def _collect_ltr(kb: KnowledgeBase, questions: list[dict], eval_dir: Path, retriever: HybridRetriever):
    cfg = RetrievalConfig(use_ltr=False)
    X, y, groups = [], [], []
    for q in questions:
        if not q["gold_evidence"]:
            continue
        a, per = _analysis_for(kb, q, eval_dir)
        res = retriever.retrieve(a, cfg, per)
        pos = M.positives(kb.chunks, q["gold_evidence"])
        for c in res.candidates[:cfg.rerank_depth]:
            X.append(feature_vector(c.feats))
            y.append(1 if c.chunk.id in pos else 0)
            groups.append(q["id"])
    return np.array(X), np.array(y), groups


def train_ranker(kb: KnowledgeBase, questions: list[dict], eval_dir: Path, log: Callable[[str], None] = lambda s: None) -> dict:
    """Fit a logistic-regression ranker on the *train* split only, then compare it with the hand-tuned default on the *test* split."""
    from sklearn.linear_model import LogisticRegression

    retriever = HybridRetriever(kb, ranker=None)
    retriever.ranker = None
    train = [q for q in questions if q["split"] == "train"]
    test = [q for q in questions if q["split"] == "test"]
    X, y, _ = _collect_ltr(kb, train, eval_dir, retriever)
    if len(set(y)) < 2:
        raise RuntimeError("training data needs both positive and negative candidates")
    mean, scale = X.mean(0), np.maximum(X.std(0), 1e-6)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit((X - mean) / scale, y)
    ranker = LearnedRanker(list(FEATURES), mean, scale, clf.coef_[0], float(clf.intercept_[0]),
                           {"trained_on": "train split", "n_pairs": int(len(y)), "n_pos": int(y.sum()), "questions": len(train)})
    base = summarize_retrieval(retrieval_rows(kb, test, eval_dir, RetrievalConfig(use_ltr=False), retriever))
    learned_r = HybridRetriever(kb, ranker=ranker)
    learned = summarize_retrieval(retrieval_rows(kb, test, eval_dir, RetrievalConfig(use_ltr=True), learned_r))
    log(f"  LTR test hit@1 {base['hit@1']['mean']} -> {learned['hit@1']['mean']}, mrr {base['mrr']['mean']} -> {learned['mrr']['mean']}")
    return {"ranker": ranker, "test_default": base, "test_learned": learned,
            "weights": {n: round(float(w), 3) for n, w in zip(FEATURES, clf.coef_[0])}}


# ------------------------------------------------------------------------------------------------ calibration
def fit_platt(confs: list[float], outcomes: list[int]) -> Optional[dict]:
    from sklearn.linear_model import LogisticRegression

    if len(set(outcomes)) < 2 or len(outcomes) < 8:
        return None
    clf = LogisticRegression(C=10.0, max_iter=1000)
    clf.fit(np.array(confs).reshape(-1, 1), outcomes)
    return {"a": float(clf.coef_[0][0]), "b": float(clf.intercept_[0])}
