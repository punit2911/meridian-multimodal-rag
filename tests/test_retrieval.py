from meridian.eval import runner as R
from meridian.eval import metrics as M
from meridian.retrieval.analyzer import Intent, analyze
from meridian.retrieval.grader import grade
from meridian.retrieval.pipeline import HybridRetriever, preset


def test_analyzer(kb):
    a = analyze("What torque for the HX-200 impeller bolt?", set(kb.known_models()))
    assert a.models == ["HX-200"] and a.intent == Intent.SPEC
    assert analyze("Specs of the HX-300", set(kb.known_models())).unknown_models == ["HX-300"]
    assert analyze("What does alarm E-47 mean?", set(kb.known_models())).codes == ["E-47"]


def test_retrieval_quality_floor(kb, questions, settings):
    rows = R.retrieval_rows(kb, questions, settings.eval_dir, preset("full"))
    s = R.summarize_retrieval(rows)
    assert s["hit@5"]["mean"] >= 0.95 and s["mrr"]["mean"] >= 0.75
    assert s["ndcg@10"]["mean"] <= 1.0


def test_full_beats_dense_only(kb, questions, settings):
    ab = R.ablation(kb, questions, settings.eval_dir, ["dense", "full"])
    assert ab["full"]["mrr"]["mean"] > ab["dense"]["mrr"]["mean"]


def test_grader_flags_out_of_corpus(kb):
    r = HybridRetriever(kb)
    for q, label in (("Who is the CEO of Helix Dynamics?", "incorrect"), ("What is the maximum flow rate of the HX-200?", "correct")):
        a = analyze(q, set(kb.known_models()))
        g = grade(r.retrieve(a).candidates, a, kb)
        assert g.label == label, (q, g)


def test_ndcg_bounded():
    assert M.ndcg_at_k([], [{"doc": "x", "contains": ["y"]}], 10) == 0.0
