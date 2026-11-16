from meridian.config import Settings
from meridian.eval.metrics import hit_at_k
from meridian.eval.runner import load_golden
from meridian.eval.stress import build_stress_corpus
from meridian.kb import KnowledgeBase
from meridian.retrieval.analyzer import analyze
from meridian.retrieval.pipeline import HybridRetriever


def test_stress_corpus_generation_and_retrieval(tmp_path):
    corpus_dir = tmp_path / "corpus"
    eval_dir = tmp_path / "eval"
    manifest = build_stress_corpus(corpus_dir, eval_dir, docs=5, seed=7)
    questions = load_golden(eval_dir)
    assert manifest["documents"] == 5
    assert manifest["answerable_questions"] == 30
    assert manifest["unanswerable_questions"] == 5
    assert manifest["test_questions"] > 0

    settings = Settings()
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    settings.demo_dir = corpus_dir
    settings.eval_dir = eval_dir
    kb = KnowledgeBase(settings)
    try:
        results = kb.add_directory(corpus_dir)
        assert sum(result.get("status") in {"added", "replaced"} for result in results) == 5
        assert len(kb.chunks) >= 5 * 6
        question = next(q for q in questions if q["id"] == "scale-000-0")
        analysis = analyze(question["question"], set(kb.known_models()))
        ranked = HybridRetriever(kb).retrieve(analysis).candidates
        assert hit_at_k([candidate.chunk for candidate in ranked], question["gold_evidence"], 5) == 1.0
    finally:
        kb.close()


def test_qdrant_pipeline_retrieves_generated_evidence(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: client)
    monkeypatch.setenv("MERIDIAN_VECTOR_STORE", "qdrant")
    corpus_dir = tmp_path / "corpus"
    eval_dir = tmp_path / "eval"
    build_stress_corpus(corpus_dir, eval_dir, docs=5, seed=9)

    settings = Settings()
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    settings.demo_dir = corpus_dir
    settings.eval_dir = eval_dir
    kb = KnowledgeBase(settings)
    try:
        kb.add_directory(corpus_dir)
        question = next(q for q in load_golden(eval_dir) if q["id"] == "scale-000-0")
        analysis = analyze(question["question"], set(kb.known_models()))
        ranked = HybridRetriever(kb).retrieve(analysis).candidates
        assert kb.ix.dense.vector_store == "qdrant"
        assert hit_at_k([candidate.chunk for candidate in ranked], question["gold_evidence"], 5) == 1.0
    finally:
        kb.close()
