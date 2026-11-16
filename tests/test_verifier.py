from meridian.verify.verifier import EvidenceDoc, Verifier, faithfulness

EV = [EvidenceDoc(index=1, text="Table 2 (HX-200 fastener tightening torques), row: Impeller bolt: Fastener M12; Torque 45 Nm. Service Bulletin SB-2026-07 effective 2026-03-18.", doc_id="d")]


def v(text):
    return Verifier().verify(text, EV)


def test_supported():
    assert faithfulness(v("The impeller bolt torque is 45 Nm [1].")) == 1.0


def test_numeric_error():
    r = v("The impeller bolt torque is 60 Nm [1].")
    assert r[0].verdict in ("contradicted", "unsupported")


def test_fabrication_and_bad_citation():
    assert faithfulness(v("The warranty covers this component for seven years [1].")) < 1.0
    assert v("The impeller bolt torque is 45 Nm [9].")[0].reasons


def test_entity_and_negation():
    assert faithfulness(v("Bulletin SB-2031-01 sets the impeller bolt torque to 45 Nm [1].")) < 1.0
    assert faithfulness(v("The impeller bolt torque is not 45 Nm [1].")) < 1.0
