import pytest

from meridian.agent.agent import AgentError, Orchestrator
from meridian.agent.llm_agent import LLMOrchestrator
from meridian.llm.mock import MockLLM


def ask(agent, q, image=None):
    return agent.answer(q, image)


def test_conflict_resolution(agent):
    r = ask(agent, "What torque should I use for the HX-200 impeller bolt?")
    assert "52 Nm" in r.answer and "SB-2026-07" in r.answer and r.conflicts and r.faithfulness == 1.0


def test_false_premise(agent):
    r = ask(agent, "Is 45 Nm the right tightening torque for the HX-200 impeller bolt?")
    assert r.answer.startswith("No") and "52 Nm" in r.answer
    r = ask(agent, "Is the HX-200 rated for fluid at 150 °C?")
    assert "140" in r.answer and not r.abstained


def test_chart_reading(agent):
    r = ask(agent, "What is the HX-500 specific power at 50% load?")
    assert "6.6" in r.answer or "6.5" in r.answer


def test_analytics_is_exact(agent):
    r = ask(agent, "How many times has alarm E-47 been logged for HX-200 units?")
    assert "33" in r.answer and any(e.step == "tool" for e in r.trace)


def test_abstains_when_unknown(agent):
    for q in ("Who is the CEO of Helix Dynamics?", "What is the maximum flow of the HX-300?", "What is the warranty period for the HX-200?"):
        assert ask(agent, q).abstained, q


def test_abstains_when_requested_detail_is_absent(agent):
    r = ask(agent, "Which grease brand is recommended for the HX-900 bearings?")
    assert r.abstained


def test_image_alarm(agent, settings):
    img = (settings.eval_dir / "images" / "photo_alarm_E47_hx500.png").read_bytes()
    r = ask(agent, "What should I do about this alarm?", img)
    assert "separator" in r.answer.lower()


def test_bad_inputs(agent):
    with pytest.raises(AgentError):
        agent.answer("   ")
    with pytest.raises(AgentError):
        agent.answer("x" * 5000)
    with pytest.raises(AgentError):
        agent.answer("what is this", b"not an image")


def test_golden_regression(agent, questions, settings):
    from meridian.eval import runner as R
    out = R.end_to_end(None, agent, questions, settings.eval_dir)
    s = out["summary"]
    assert s["answer_correctness"]["mean"] >= 0.88 and s["abstention_recall"]["mean"] == 1.0 and not s["errors"]


# ---------------------------------------------------------------- LLM tool-calling path (scripted model, no network)
def test_llm_agent_grounded_and_fallbacks(kb, settings):
    q = "What torque should I use for the HX-200 impeller bolt?"
    good = LLMOrchestrator(kb, settings, MockLLM()).answer(q)
    assert not good.abstained and "52" in good.answer
    liar = LLMOrchestrator(kb, settings, MockLLM(responder=lambda q, e: "The HX-200 impeller bolt torque is 75 Nm [1]."))
    r = liar.answer(q)
    assert "75" not in r.answer and "52 Nm" in r.answer  # hallucination caught -> extractive fallback
    down = LLMOrchestrator(kb, settings, MockLLM(fail=True)).answer(q)
    assert "52 Nm" in down.answer
    refuse = LLMOrchestrator(kb, settings, MockLLM(responder=lambda q, e: "INSUFFICIENT_EVIDENCE")).answer(q)
    assert refuse.abstained
