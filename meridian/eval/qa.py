"""Golden QA set. Ground truth comes from the same dictionaries that rendered the corpus (see world.py),
so labels are exact. Questions are deliberately paraphrased (vocabulary mismatch) and cover every modality
plus conflict, false-premise, ambiguity, multi-hop and unanswerable cases."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..utils.text import slugify
from . import world as W


def doc_id(filename: str) -> str:
    return slugify(Path(filename).stem)


D200 = doc_id("HX-200_Service_Manual.pdf")
D500 = doc_id("HX-500_Service_Manual.pdf")
D900 = doc_id("HX-900_Service_Manual.pdf")
DSB = doc_id(W.BULLETIN["filename"])
DCARD = doc_id("HX-200_Field_Card_scan.pdf")
DLOG = doc_id("maintenance_log_2026.csv")


def num(value, unit=None, tol=0.01):
    return {"kind": "num", "value": value, "unit": unit, "tol": tol}


def txt(*values):
    return {"kind": "text", "values": list(values)}


def ev(doc, *contains):
    return {"doc": doc, "contains": [c.lower() for c in contains]}


def _log_stats(log_path: Path) -> dict:
    with open(log_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    e47_200 = sum(1 for r in rows if r["model"] == "HX-200" and r["error_code"] == "E-47")
    dt500 = round(sum(float(r["downtime_h"]) for r in rows if r["model"] == "HX-500"), 2)
    most = Counter(r["error_code"] for r in rows).most_common(1)[0][0]
    e52 = [float(r["downtime_h"]) for r in rows if r["error_code"] == "E-52"]
    return {"e47_200": e47_200, "dt500": dt500, "most": most, "e52_mean": round(sum(e52) / len(e52), 2), "n": len(rows)}


def build_questions(log_path: Path) -> list[dict]:
    st = _log_stats(log_path)
    Q: list[dict] = []

    def add(qtype, question, evidence, facts, abstain=False, image=None, note=""):
        Q.append({"type": qtype, "question": question, "gold_evidence": evidence, "answer_facts": facts,
                  "expect_abstain": abstain, "image": image, "note": note})

    # ---- spec lookups (table rows) ------------------------------------------------------------------
    S = "spec"
    add(S, "What is the maximum flow rate of the HX-200?", [ev(D200, "maximum flow")], [num(120, "m3/h")])
    add(S, "How much head can the HX-200 pump generate at most?", [ev(D200, "maximum head")], [num(58, "m")])
    add(S, "What motor power does the HX-200 use?", [ev(D200, "rated motor power")], [num(22, "kw")])
    add(S, "What's the top fluid temperature the HX-200 can handle?", [ev(D200, "maximum fluid temperature")], [num(140, "c")])
    add(S, "How heavy is the HX-200?", [ev(D200, "pump weight")], [num(185, "kg")])
    add(S, "What is the rated speed of the HX-200?", [ev(D200, "rated speed")], [num(2950, "rpm")])
    add(S, "How noisy is the HX-200 pump?", [ev(D200, "noise level")], [num(78, "dba")])
    add(S, "What is the maximum working pressure of the HX-200?", [ev(D200, "maximum working pressure")], [num(16, "bar")])
    add(S, "What is the free air delivery of the HX-500 compressor?", [ev(D500, "free air delivery")], [num(18.5, "m3/min")])
    add(S, "How many litres of oil does the HX-500 hold?", [ev(D500, "oil capacity")], [num(38, "l")])
    add(S, "At what discharge temperature does the HX-500 trip?", [ev(D500, "discharge temperature trip")], [num(105, "c")])
    add(S, "What is the weight of the HX-500?", [ev(D500, "weight")], [num(1450, "kg")])
    add(S, "What is the HX-500 motor rating?", [ev(D500, "rated motor power")], [num(110, "kw")])
    add(S, "What is the maximum working pressure of the HX-500?", [ev(D500, "maximum working pressure")], [num(10, "bar")])
    add(S, "What pressure can the HX-900 hydraulic unit reach in normal operation?", [ev(D900, "maximum working pressure")], [num(350, "bar")])
    add(S, "What is the relief valve setting on the HX-900?", [ev(D900, "relief valve setting")], [num(360, "bar")])
    add(S, "What is the tank capacity of the HX-900 power unit?", [ev(D900, "tank capacity")], [num(400, "l")])
    add(S, "Which oil grade should the HX-900 use?", [ev(D900, "oil grade")], [txt("ISO VG 46")])
    add(S, "What filtration rating does the HX-900 have?", [ev(D900, "filtration rating")], [num(10, "um")])
    add(S, "What is the rated flow of the HX-900?", [ev(D900, "rated flow")], [num(90, "l/min")])
    add(S, "How much does the HX-900 weigh?", [ev(D900, "weight")], [num(920, "kg")])
    add(S, "What's the maximum oil temperature for the HX-900?", [ev(D900, "maximum oil temperature")], [num(65, "c")])
    add("ambiguous", "What is the maximum working pressure?", [ev(D200, "maximum working pressure"), ev(D500, "maximum working pressure"), ev(D900, "maximum working pressure")],
        [num(16, "bar"), num(10, "bar"), num(350, "bar")], note="no model given: must answer per product")

    # ---- torque / maintenance tables --------------------------------------------------------------------
    T = "table"
    add(T, "What torque should the HX-200 casing bolts be tightened to?", [ev(D200, "casing bolt")], [num(110, "nm")])
    add(T, "What is the tightening torque for the HX-500 oil drain plug?", [ev(D500, "oil drain plug")], [num(50, "nm")])
    add(T, "What torque applies to the HX-500 airend mounting bolts?", [ev(D500, "airend mounting bolt")], [num(210, "nm")])
    add(T, "What is the HX-900 pump mounting bolt torque?", [ev(D900, "pump mounting bolt")], [num(200, "nm")])
    add(T, "How often should the HX-500 oil filter be replaced?", [ev(D500, "replace oil filter")], [num(2000, "h")])
    add(T, "What is the replacement interval for the HX-200 mechanical seal?", [ev(D200, "replace mechanical seal")], [num(8000, "h")])
    add(T, "How frequently should HX-900 oil be sampled for contamination analysis?", [ev(D900, "sample oil")], [num(500, "h")])
    add(T, "When should the HX-500 oil separator element be replaced?", [ev(D500, "separator element")], [num(6000, "h")])
    add(T, "How often do the HX-200 bearings need lubrication?", [ev(D200, "lubricate bearings")], [num(2000, "h")])
    add(T, "What is the torque for the HX-900 relief valve cartridge?", [ev(D900, "relief valve cartridge")], [num(80, "nm")])

    # ---- alarm codes / troubleshooting ------------------------------------------------------------------
    E = "troubleshooting"
    add(E, "What does error E-47 mean on the HX-200?", [ev(D200, "e-47", "low suction pressure")], [txt("low suction pressure")])
    add(E, "My HX-500 shows E-47. What should I do?", [ev(D500, "e-47", "separator")], [txt("oil separator element")])
    add(E, "The HX-900 displays E-47, what is wrong?", [ev(D900, "e-47", "return filter")], [txt("return filter")])
    add(E, "HX-200 motor overload alarm E-12: what is the likely cause?", [ev(D200, "e-12")], [txt("blocked impeller")])
    add(E, "The HX-500 shut down because the air got too hot. What should I check?", [ev(D500, "e-05")], [txt("oil level", "cooler")])
    add(E, "My HX-200 is leaking at the seal and shows a fault. What is the remedy?", [ev(D200, "e-52")], [txt("replace the mechanical seal")])
    add(E, "The HX-200 is vibrating heavily. What could be causing it?", [ev(D200, "e-63")], [txt("imbalance", "misaligned")])
    add(E, "HX-900 oil level warning: which alarm is that and what is the remedy?", [ev(D900, "e-09")], [txt("refill")])
    add(E, "The HX-200 keeps losing prime and the pump is cavitating. Which alarm applies?", [ev(D200, "e-47")], [txt("suction")])
    add(E, "How do I clear alarm E-71 on the HX-500?", [ev(D500, "e-71")], [txt("refill")])

    # ---- procedures / text ----------------------------------------------------------------------------------
    PR = "procedure"
    add(PR, "How do I replace the mechanical seal on the HX-200?", [ev(D200, "gland bolts")], [txt("gland bolts"), txt("lock out")])
    add(PR, "What are the steps for changing the compressor oil on the HX-500?", [ev(D500, "container of at least 40")], [txt("drain plug"), txt("sight glass")])
    add(PR, "How do I replace the impeller on the HX-200?", [ev(D200, "puller tool")], [txt("puller", "pt-200")])
    add(PR, "How do I start the HX-200 for the first time?", [ev(D200, "direction of rotation")], [txt("direction of rotation"), txt("prime")])
    add(PR, "How do I replace the return filter on the HX-900?", [ev(D900, "drip tray")], [txt("drip tray"), txt("o-ring")])
    add(PR, "What safety precautions apply before servicing the HX-500?", [ev(D500, "depressurise the system")], [txt("lock out"), txt("depressurise")])
    add(PR, "How do I bleed air out of the HX-900 hydraulic circuit?", [ev(D900, "bleed screw")], [txt("bleed screw")])
    add(PR, "How should the HX-200 suction piping be arranged?", [ev(D200, "five times the pipe diameter")], [txt("five times")])
    add("text", "How should the HX-200 maintenance intervals change for abrasive fluids?", [ev(D200, "abrasive")], [txt("halve")])
    add("text", "How level does the HX-200 baseplate have to be?", [ev(D200, "0.2 mm per metre")], [num(0.2, "mm")])
    add("text", "At what leakage rate should the HX-200 mechanical seal be replaced?", [ev(D200, "10 drops per minute")], [num(10)])

    # ---- charts and figures -------------------------------------------------------------------------------------
    C = "chart"
    add(C, "According to the performance curve, what head does the HX-200 deliver at 80 m3/h?", [ev(D200, "performance curve")], [num(39, None, 0.07)])
    add(C, "What is the head of the HX-200 at a flow of 40 m3/h?", [ev(D200, "performance curve")], [num(53, None, 0.07)])
    add(C, "Read the HX-200 curve: what head at 100 m3/h?", [ev(D200, "performance curve")], [num(28, None, 0.08)])
    add(C, "What head does the HX-200 give at 70 m3/h?", [ev(D200, "performance curve")], [num(43, None, 0.08)], note="needs interpolation")
    add(C, "What is the HX-500 specific power at 50% load?", [ev(D500, "specific power")], [num(6.6, None, 0.07)])
    add(C, "What is the specific power of the HX-500 at full load?", [ev(D500, "specific power")], [num(6.0, None, 0.07)])
    add(C, "How hot is the HX-900 oil after 60 minutes of warm-up?", [ev(D900, "oil temperature")], [num(56, None, 0.06)])
    add("figure", "Show me the exploded view of the HX-200 pump.", [ev(D200, "exploded view")], [txt("figure 1")])
    add("figure", "Where can I find the hydraulic circuit schematic for the HX-900?", [ev(D900, "hydraulic circuit")], [txt("figure 1")])
    add("figure", "Is there a diagram showing the air and oil flow of the HX-500?", [ev(D500, "flow diagram")], [txt("figure 1")])

    # ---- multi-hop (cross-reference following) ---------------------------------------------------------------------
    M = "multihop"
    add(M, "What tightening torque is specified for the fasteners in the HX-200 mechanical seal replacement procedure?",
        [ev(D200, "gland bolts"), ev(D200, "gland bolt")], [num(25, "nm")])
    add(M, "Which torque applies to the plug that is refitted in the HX-500 oil change procedure?",
        [ev(D500, "container of at least 40"), ev(D500, "oil drain plug")], [num(50, "nm")])
    add(M, "In the HX-200 impeller replacement procedure, what torque goes on the casing bolts?",
        [ev(D200, "casing bolt")], [num(110, "nm")])
    add(M, "Which alarm code appears when the HX-200 seal fails, and which section describes the fix?",
        [ev(D200, "e-52")], [txt("e-52"), txt("6.1")])
    add(M, "Alarm E-31 on the HX-200 points to lubrication. How often is that task due?",
        [ev(D200, "e-31"), ev(D200, "lubricate bearings")], [num(2000, "h")])

    # ---- supersession / conflict ------------------------------------------------------------------------------------------
    K = "conflict"
    add(K, "What torque should I use for the HX-200 impeller bolt?", [ev(D200, "impeller bolt"), ev(DSB, "impeller bolt")],
        [num(52, "nm"), txt("sb-2026-07")], note="manual says 45, bulletin supersedes with 52")
    add(K, "Is 45 Nm the right tightening torque for the HX-200 impeller bolt?", [ev(DSB, "impeller bolt")],
        [num(52, "nm"), txt("sb-2026-07")], note="false premise caused by superseded value")
    add(K, "Do I need threadlocker on the HX-200 impeller bolt?", [ev(DSB, "threadlocker")], [txt("loctite 243")])

    # ---- false premises ----------------------------------------------------------------------------------------------------------
    F = "false_premise"
    add(F, "The HX-500 maximum working pressure is 16 bar, correct?", [ev(D500, "maximum working pressure")], [num(10, "bar")])
    add(F, "I read that the HX-900 tank holds 250 L. Is that right?", [ev(D900, "tank capacity")], [num(400, "l")])
    add(F, "Is the HX-200 rated for fluid at 150 °C?", [ev(D200, "maximum fluid temperature")], [num(140, "c")])

    # ---- analytics over the CSV log -----------------------------------------------------------------------------------------------
    A = "analytics"
    add(A, "How many times has alarm E-47 been logged for HX-200 units?", [], [num(st["e47_200"])])
    add(A, "What is the total downtime recorded for HX-500 units?", [], [num(st["dt500"], None, 0.005)])
    add(A, "Which alarm code occurs most often in the maintenance log?", [], [txt(st["most"].lower())])
    add(A, "What is the average downtime for E-52 events?", [], [num(st["e52_mean"], None, 0.01)])

    # ---- scanned page (OCR) -----------------------------------------------------------------------------------------------------------
    O = "scanned"
    add(O, "How many starts per hour are permitted for the HX-200?", [ev(DCARD, "starts per hour")], [num(6)])
    add(O, "How much liquid is needed to prime the HX-200 casing?", [ev(DCARD, "priming volume")], [num(14, "l")])
    add(O, "How much seal flush water does the HX-200 need?", [ev(DCARD, "seal flush")], [num(2, "l/min")])
    add(O, "At what suction pressure should the HX-200 be stopped?", [ev(DCARD, "suction pressure")], [num(0.5, "bar")])

    # ---- image queries -----------------------------------------------------------------------------------------------------------------
    I = "image"
    add(I, "What does this alarm mean and what should I do?", [ev(D200, "e-47", "low suction pressure")],
        [txt("suction")], image="images/photo_alarm_E47_hx200.png")
    add(I, "What should I do about this alarm?", [ev(D500, "e-47", "separator")],
        [txt("oil separator element")], image="images/photo_alarm_E47_hx500.png", note="same code, different machine")
    add(I, "What oil capacity does this machine have?", [ev(D500, "oil capacity")], [num(38, "l")],
        image="images/photo_dataplate_hx500.png")
    add(I, "Which figure in the manuals matches this picture?", [ev(D200, "exploded view")], [txt("exploded")],
        image="images/query_diagram_exploded.jpg")

    # ---- unanswerable ----------------------------------------------------------------------------------------------------------------------
    U = "unanswerable"
    add(U, "What is the impeller bolt torque for the HX-700?", [], [], abstain=True)
    add(U, "What is the warranty period for the HX-200?", [], [], abstain=True)
    add(U, "How much does an HX-500 compressor cost?", [], [], abstain=True)
    add(U, "Which grease brand is recommended for the HX-900 bearings?", [], [], abstain=True)
    add(U, "Who is the CEO of Helix Dynamics?", [], [], abstain=True)
    add(U, "What is the maximum flow of the HX-300?", [], [], abstain=True)

    # ids and deterministic train/test split (alternating within each type, so both splits cover all types)
    counters: dict[str, int] = defaultdict(int)
    for i, q in enumerate(Q, 1):
        q["id"] = f"q{i:03d}"
        q["split"] = "train" if counters[q["type"]] % 2 == 0 else "test"
        counters[q["type"]] += 1
    return Q


def write_golden(eval_dir: Path, corpus_dir: Path) -> Path:
    qs = build_questions(corpus_dir / "maintenance_log_2026.csv")
    out = eval_dir / "golden_qa.json"
    out.write_text(json.dumps({"version": 1, "questions": qs}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
