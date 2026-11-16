import pytest

from meridian.ingest.pipeline import UnsupportedFileType, ingest_file
from meridian.schemas import Modality


def test_modalities_present(kb):
    mods = kb.stats()["modalities"]
    for m in ("text", "table", "table_row", "figure", "chart", "warning", "ocr_page"):
        assert mods.get(m, 0) > 0, m


def test_models_and_docs(kb):
    assert kb.known_models() == ["HX-200", "HX-500", "HX-900"]
    assert len(kb.docs) == 6


def test_table_rows_carry_bbox_and_facts(kb):
    rows = [c for c in kb.chunks if c.modality == Modality.TABLE_ROW and c.bbox]
    assert rows and rows[0].bbox.page_width > 0
    assert any(f.model == "HX-200" and f.display == "45 Nm" for f in kb.ix.facts.facts)


def test_chart_digitised(kb):
    vals = [f for f in kb.ix.facts.facts if f.attribute == "chart value" and f.model == "HX-500"]
    assert len(vals) >= 4 and all(f.approximate for f in vals)


def test_bulletin_title_has_subject(kb):
    assert any("Revised impeller bolt" in d.title for d in kb.docs.values())


def test_log_is_a_dataset(kb):
    assert kb.datasets() and len(kb.dataframe(kb.datasets()[0].id)) > 50


def test_unsupported_and_corrupt(settings, tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        ingest_file(bad, settings, doc_id="bad")
    with pytest.raises(UnsupportedFileType):
        from meridian.kb import KnowledgeBase
        x = tmp_path / "a.exe"; x.write_bytes(b"x")
        KnowledgeBase(settings).add_file(x)


def test_duplicate_upload_is_idempotent(kb, settings):
    p = next(settings.demo_dir.glob("*.csv"))
    assert kb.add_file(p)["status"] == "duplicate"
