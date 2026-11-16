"""Full evaluation pipeline -> reports/eval_report.{json,md} + PNG charts. Used by scripts/run_eval.py and the API."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

from ..agent.agent import CALIBRATION_PATH, Orchestrator, calibrate
from ..agent.factory import make_agent
from ..config import Settings
from ..kb import KnowledgeBase
from ..retrieval.pipeline import HybridRetriever
from ..retrieval.rerank import RESOURCE as LTR_PATH
from . import metrics as M
from . import runner as R


def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def run_full_evaluation(kb: KnowledgeBase, settings: Settings, out_dir: Optional[Path] = None, *, train_models: bool = True,
                        calibration_path: Optional[Path] = None, log: Callable[[str], None] = print) -> dict:
    out_dir = Path(out_dir or settings.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    questions = R.load_golden(settings.eval_dir)
    manifest_path = Path(settings.eval_dir) / "_stress_manifest.json"
    dataset = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "dataset": "golden_qa.json",
        "synthetic": None,
    }
    report: dict = {"questions": len(questions), "chunks": len(kb.chunks), "documents": len(kb.docs), "splits": {
        s: sum(q["split"] == s for q in questions) for s in ("train", "test")},
        "dataset": dataset, "vector_store": kb.ix.dense.vector_store}

    # 1. learned ranker (train split only)
    if train_models:
        log("Training the learned ranker on the train split ...")
        tr = R.train_ranker(kb, questions, settings.eval_dir, log)
        b, l = tr["test_default"], tr["test_learned"]
        adopt = l["hit@1"]["mean"] >= b["hit@1"]["mean"] and l["mrr"]["mean"] >= b["mrr"]["mean"] - 1e-9
        if adopt:
            tr["ranker"].save(LTR_PATH)
        report["ltr"] = {"adopted": bool(adopt), "test_default": b, "test_learned": l, "weights": tr["weights"], "meta": tr["ranker"].meta}
        log(f"  learned ranker {'adopted' if adopt else 'NOT adopted (no gain on held-out test)'}")
    retriever = HybridRetriever(kb)

    # 2. retrieval ablation on the held-out test split and on everything
    log("Retrieval ablation (test split) ...")
    report["ablation_test"] = R.ablation(kb, questions, settings.eval_dir, split="test", retriever=retriever, log=log)
    log("Retrieval ablation (all questions) ...")
    report["ablation_all"] = R.ablation(kb, questions, settings.eval_dir, retriever=retriever, log=log)

    # 3. end to end (calibration fitted on train, reported on test)
    log("End-to-end answers ...")
    agent = make_agent(kb, settings)
    agent.calibration = None  # measure raw confidence first
    e2e = R.end_to_end(kb, agent, questions, settings.eval_dir, log)
    rows = e2e["rows"]
    tr_rows = [r for r in rows if r["split"] == "train" and not r["abstained"]]
    te_rows = [r for r in rows if r["split"] == "test" and not r["abstained"]]
    platt = R.fit_platt([r["confidence"] for r in tr_rows], [int(r["correct"]) for r in tr_rows])
    cal = {"raw_ece_test": None, "calibrated_ece_test": None}
    if te_rows:
        probs = [r["confidence"] for r in te_rows]
        outs = [int(r["correct"]) for r in te_rows]
        cal["raw_ece_test"] = round(M.ece(probs, outs)[0], 4)
        cal["raw_brier_test"] = round(M.brier(probs, outs), 4)
        if platt:
            cp = [calibrate(p, platt) for p in probs]
            cal["calibrated_ece_test"] = round(M.ece(cp, outs)[0], 4)
            cal["calibrated_brier_test"] = round(M.brier(cp, outs), 4)
            if cal["calibrated_brier_test"] <= cal["raw_brier_test"]:
                destination = Path(calibration_path or CALIBRATION_PATH)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(platt), encoding="utf-8")
                cal["adopted"] = True
            else:
                cal["adopted"] = False
    report["calibration"] = cal
    report["e2e_all"] = e2e["summary"]
    report["e2e_test"] = R.summarize_e2e([r for r in rows if r["split"] == "test"])
    report["e2e_rows"] = [{k: r[k] for k in ("id", "type", "split", "question", "correct", "abstained", "confidence", "faithfulness", "missing", "answer")} for r in rows]

    # 4. hallucination benchmark
    log("Hallucination-injection benchmark ...")
    report["hallucination"] = R.hallucination_benchmark(kb, rows, agent, questions, settings.eval_dir)

    report["seconds"] = round(time.time() - t0, 1)
    report = _clean(report)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")
    try:
        render_charts(report, out_dir)
    except Exception as exc:  # charts are a convenience, never fail the evaluation
        log(f"  (charts skipped: {exc})")
    log(f"Report written to {out_dir}")
    return report


def _f(x, pct=True):
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def render_markdown(r: dict) -> str:
    L = ["# Meridian evaluation report", "",
         f"{r['questions']} golden questions ({r['splits']['train']} train / {r['splits']['test']} held-out test) over {r['documents']} documents and {r['chunks']} chunks. Runtime {r['seconds']} s.",
         f"Dataset: `{r['dataset'].get('dataset', 'unknown')}`; synthetic: `{r['dataset'].get('synthetic', 'not specified')}`; vector store: `{r['vector_store']}`.",
         f"Split: {r['dataset'].get('split_strategy', 'not specified')}. {r['dataset'].get('limitations', '')}", ""]
    L += ["## 1. Retrieval ablation (held-out test split)", "",
          "| Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@10 | ms/query |", "|---|---|---|---|---|---|---|---|"]
    for name, s in r["ablation_test"].items():
        L.append(f"| {name} | {_f(s['hit@1']['mean'])} | {_f(s['hit@3']['mean'])} | {_f(s['hit@5']['mean'])} | {_f(s['recall@5']['mean'])} | "
                 f"{_f(s['mrr']['mean'], False)} | {_f(s['ndcg@10']['mean'], False)} | {s['ms']['mean']:.0f} |")
    L += ["", "Same ablation over all questions (train + test):", "", "| Configuration | Hit@1 | Hit@5 | MRR | nDCG@10 |", "|---|---|---|---|---|"]
    for name, s in r["ablation_all"].items():
        L.append(f"| {name} | {_f(s['hit@1']['mean'])} | {_f(s['hit@5']['mean'])} | {_f(s['mrr']['mean'], False)} | {_f(s['ndcg@10']['mean'], False)} |")
    if "ltr" in r:
        l = r["ltr"]
        L += ["", "## 2. Learned ranker (trained on train split, evaluated on test)", "",
              f"Adopted: **{l['adopted']}**. Hit@1 {_f(l['test_default']['hit@1']['mean'])} → {_f(l['test_learned']['hit@1']['mean'])}; "
              f"MRR {_f(l['test_default']['mrr']['mean'], False)} → {_f(l['test_learned']['mrr']['mean'], False)}.", "",
              "Top learned feature weights: " + ", ".join(f"`{k}` {v:+.2f}" for k, v in sorted(l["weights"].items(), key=lambda kv: -abs(kv[1]))[:6]), ""]
    e = r["e2e_all"]
    L += ["## 3. End-to-end answers", "", "| Metric | All | Test split |", "|---|---|---|"]
    t = r["e2e_test"]
    def m(s, k): return _f(s[k]["mean"] if isinstance(s[k], dict) else s[k])
    for label, k in (("Answer correctness", "answer_correctness"), ("Abstention recall (unanswerable declined)", "abstention_recall"),
                     ("Abstention precision", "abstention_precision"), ("False-abstention rate", "false_abstention_rate"),
                     ("Faithfulness (verified claims)", "faithfulness"), ("Citation recall", "citation_recall")):
        L.append(f"| {label} | {m(e, k)} | {m(t, k)} |")
    L += ["", f"Latency p50-ish mean {e['latency_ms']['mean']:.0f} ms. Errors: {e['errors'] or 'none'}.", "", "Correctness by question type:", "",
          "| Type | Correct |", "|---|---|"] + [f"| {k} | {_f(v)} |" for k, v in e["by_type"].items()]
    c = r["calibration"]
    L += ["", "## 4. Confidence calibration (test split)", "",
          f"ECE raw {_f(c.get('raw_ece_test'), False)} → calibrated {_f(c.get('calibrated_ece_test'), False)}; Brier raw {_f(c.get('raw_brier_test'), False)} → "
          f"calibrated {_f(c.get('calibrated_brier_test'), False)} (Platt parameters fitted on the train split; adopted: {c.get('adopted')}).", ""]
    h = r["hallucination"]
    L += ["## 5. Hallucination-injection benchmark", "",
          f"{h['clean_answers']} clean answers; false-positive rate on clean answers {_f(h['false_positive_rate'])}. "
          f"{h['total_injected']} injected hallucinations, overall detection {_f(h['overall_detection_rate'])}.", "",
          "| Injection | Count | Detected |", "|---|---|---|"]
    for k, n in h["injected"].items():
        L.append(f"| {k} | {n} | {_f(h['detection_rate'][k])} |")
    fails = [x for x in r["e2e_rows"] if not x["correct"]]
    L += ["", f"## 6. Failures ({len(fails)})", ""]
    L += [f"- `{x['id']}` ({x['type']}) {x['question']} → missing {x['missing']}" for x in fails] or ["None."]
    return "\n".join(L) + "\n"


def render_charts(r: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(r["ablation_test"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = range(len(names))
    for off, key, col in ((-0.27, "hit@1", "#8aa4c8"), (0.0, "hit@5", "#3b6ea8"), (0.27, "mrr", "#e0a030")):
        ax.bar([x + off for x in xs], [r["ablation_test"][n][key]["mean"] or 0 for n in names], 0.26, label=key, color=col)
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, rotation=20, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval ablation (held-out test split)"); ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "ablation.png", dpi=140); plt.close(fig)

    bins = r["e2e_all"]["reliability"]
    if bins:
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        ax.plot([0, 1], [0, 1], "--", color="#999")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#3b6ea8")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_dir / "reliability.png", dpi=140); plt.close(fig)
