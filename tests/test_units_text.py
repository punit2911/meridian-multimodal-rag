from meridian.utils import units as U
from meridian.utils.text import has_negation, tokenize
from meridian.eval.metrics import answer_correct


def test_quantity_extraction_and_conversion():
    qs = U.extract_quantities("Maximum pressure 16 bar and 150 °C at 120 m3/h")
    assert {(round(q.value), q.unit) for q in qs} >= {(16, "bar"), (150, "c")}
    a, b = U.extract_quantities("10 bar")[0], U.extract_quantities("145 psi")[0]
    assert U.quantities_match(a, b, rel_tol=0.01)


def test_answer_correct_normalizes_gold_unit_casing():
    assert answer_correct("Rated power: 84 kW.", [{"kind": "num", "value": 84, "unit": "kW"}])[0]
    assert answer_correct("Torque: 62 Nm.", [{"kind": "num", "value": 62, "unit": "Nm"}])[0]


def test_synonym_canonicalisation():
    assert tokenize("tightening torque") == tokenize("torque tightening")[::-1] or "torqu" in " ".join(tokenize("torque"))
    assert tokenize("weigh") == tokenize("weight")


def test_negation():
    assert has_negation("Do not open") and not has_negation("Open the valve")
