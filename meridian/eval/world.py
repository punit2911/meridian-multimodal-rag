"""The fictional 'Helix Dynamics' product world used for the demo corpus and the golden QA set.

Everything here is data. `datagen.py` renders it into PDFs / DOCX / CSV / images and `qa.py` derives
questions whose ground truth is exact by construction (the same dictionaries produce both).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MANUFACTURER = "Helix Dynamics"


# ----------------------------------------------------------------------------- block helpers
def H1(t): return ("h1", t)
def H2(t): return ("h2", t)
def P(t): return ("p", t)
def WARN(t): return ("warn", t)
def STEPS(items): return ("steps", items)
def TABLE(caption, header, rows, widths): return ("table", caption, header, rows, widths)
def FIG(key, caption, width_cm=11.0): return ("fig", key, caption, width_cm)
def PB(): return ("pb",)


@dataclass
class ManualSpec:
    key: str
    filename: str
    model: str
    product: str
    revision: str
    date: str
    blocks: list[tuple] = field(default_factory=list)


# ----------------------------------------------------------------------------- chart data (ground truth)
HX200_CURVE = {"x": [0, 20, 40, 60, 80, 100, 120], "y": [58.0, 56.5, 53.0, 47.0, 39.0, 28.0, 14.0],
               "xlabel": "Flow (m3/h)", "ylabel": "Head (m)", "title": "HX-200 head vs flow at 2950 rpm"}
HX500_BARS = {"x": ["25%", "50%", "75%", "100%"], "y": [7.4, 6.6, 6.2, 6.0],
              "xlabel": "Load step", "ylabel": "Specific power (kW per m3/min)", "title": "HX-500 specific power by load"}
HX900_TEMP = {"x": [0, 20, 40, 60, 80, 100, 120], "y": [30.0, 41.0, 50.0, 56.0, 60.0, 62.0, 63.0],
              "xlabel": "Run time (min)", "ylabel": "Oil temperature (degC)", "title": "HX-900 oil temperature during warm-up"}

# ----------------------------------------------------------------------------- table data (ground truth)
HX200_SPECS = [
    ["Maximum flow", "120", "m³/h"], ["Maximum head", "58", "m"], ["Rated motor power", "22", "kW"],
    ["Rated speed", "2950", "rpm"], ["Maximum working pressure", "16", "bar"], ["Maximum fluid temperature", "140", "°C"],
    ["Pump weight", "185", "kg"], ["Noise level", "78", "dB(A)"], ["Mechanical seal type", "MS-4", "-"],
    ["Drive end bearing", "6310-C3", "-"],
]
HX200_TORQUE = [
    ["Impeller bolt", "M12", "45"], ["Casing bolt", "M16", "110"], ["Gland bolt", "M10", "25"],
    ["Bearing housing bolt", "M12", "60"], ["Coupling guard bolt", "M8", "18"], ["Drain plug", "G1/2", "30"],
]
HX200_MAINT = [
    ["Visual inspection", "Weekly", "Operator"], ["Check mechanical seal for leakage", "250 h", "Technician"],
    ["Lubricate bearings", "2000 h", "Technician"], ["Inspect impeller wear", "4000 h", "Technician"],
    ["Replace mechanical seal", "8000 h", "Technician"], ["Replace bearings", "20000 h", "Technician"],
]
HX200_ERRORS = [
    ["E-12", "Motor overload", "Blocked impeller or high fluid density",
     "Isolate power, clear the obstruction and check the fluid density against Table 1"],
    ["E-31", "High bearing temperature", "Insufficient lubrication or worn bearing",
     "Lubricate the bearings per Table 3 or replace the bearings"],
    ["E-47", "Low suction pressure", "Closed suction valve, clogged strainer or air ingress",
     "Open the suction valve fully, clean the strainer and check the suction line for air leaks"],
    ["E-52", "Mechanical seal leakage", "Worn seal faces or dry running",
     "Replace the mechanical seal (see Section 6.1)"],
    ["E-63", "Excessive vibration", "Impeller imbalance or misaligned coupling",
     "Inspect the impeller and realign the coupling"],
]

HX500_SPECS = [
    ["Free air delivery", "18.5", "m³/min"], ["Maximum working pressure", "10", "bar"], ["Rated motor power", "110", "kW"],
    ["Oil capacity", "38", "L"], ["Discharge temperature trip", "105", "°C"], ["Noise level", "72", "dB(A)"],
    ["Weight", "1450", "kg"], ["Rated speed", "2980", "rpm"], ["Maximum separator pressure drop", "1.0", "bar"],
]
HX500_TORQUE = [
    ["Oil drain plug", "G3/4", "50"], ["Oil filler plug", "G1", "40"], ["Separator cover bolt", "M16", "120"],
    ["Airend mounting bolt", "M20", "210"], ["Cooler mounting bolt", "M10", "45"], ["Coupling bolt", "M12", "85"],
]
HX500_MAINT = [
    ["Check oil level", "Daily", "Operator"], ["Clean cooler fins", "1000 h", "Technician"],
    ["Replace oil filter", "2000 h", "Technician"], ["Replace air filter element", "2000 h", "Technician"],
    ["Replace oil", "4000 h", "Technician"], ["Replace oil separator element", "6000 h", "Technician"],
]
HX500_ERRORS = [
    ["E-05", "Discharge temperature high", "Low oil level, blocked cooler or failed thermostatic valve",
     "Check the oil level, clean the cooler fins and test the thermostatic valve"],
    ["E-47", "Separator differential pressure high", "Clogged oil separator element",
     "Replace the oil separator element (see Table 3)"],
    ["E-58", "Motor overload", "Excessive load or unbalanced supply voltage",
     "Check the supply voltage on all three phases and reduce the load"],
    ["E-71", "Oil level low", "Oil leak or insufficient fill",
     "Inspect the unit for leaks and refill the oil to the sight-glass mark (capacity in Table 1)"],
]

HX900_SPECS = [
    ["Maximum working pressure", "350", "bar"], ["Relief valve setting", "360", "bar"], ["Tank capacity", "400", "L"],
    ["Rated flow", "90", "L/min"], ["Rated motor power", "75", "kW"], ["Oil grade", "ISO VG 46", "-"],
    ["Filtration rating", "10", "µm"], ["Maximum oil temperature", "65", "°C"], ["Weight", "920", "kg"], ["Noise level", "74", "dB(A)"],
]
HX900_TORQUE = [
    ["Pump mounting bolt", "M16", "200"], ["Tank cover bolt", "M10", "40"], ["Filter housing", "M12", "70"],
    ["Relief valve cartridge", "M27", "80"],
]
HX900_MAINT = [
    ["Check oil level", "Daily", "Operator"], ["Sample oil for contamination analysis", "500 h", "Technician"],
    ["Replace return filter element", "1000 h", "Technician"], ["Check accumulator pre-charge", "2000 h", "Technician"],
    ["Replace oil", "6000 h", "Technician"],
]
HX900_FLUIDS = [
    ["Mineral oil ISO VG 46", "Yes", "Standard fill"], ["Water-glycol", "No", "Seals are incompatible"],
    ["Phosphate ester", "No", "Requires FKM seals which are not fitted"],
    ["Bio-oil (HEES)", "Yes", "Reduce all service intervals by 50 percent"],
]
HX900_ERRORS = [
    ["E-09", "Low oil level", "Tank level below the lower mark", "Inspect for leaks and refill to the upper mark"],
    ["E-20", "Oil over-temperature", "Fouled cooler or excessive duty cycle", "Clean the cooler and reduce the duty cycle"],
    ["E-47", "Return filter clogged", "Differential pressure across the return filter above 3 bar",
     "Replace the return filter element (see Section 6.1)"],
    ["E-88", "System over-pressure", "Relief valve stuck or set too high", "Check and reset the relief valve to the value in Table 1"],
]


def _manuals() -> list[ManualSpec]:
    hx200 = ManualSpec("hx200", "HX-200_Service_Manual.pdf", "HX-200", "Centrifugal Process Pump", "C", "2025-09-01", [
        H1("1 Safety"),
        P("This manual is intended for qualified maintenance technicians. Read this chapter completely before installing, "
          "operating or servicing the HX-200 centrifugal process pump."),
        WARN("WARNING: Lock out and tag out the electrical supply before any maintenance work. Stored energy in the coupling and "
             "residual pressure in the casing can cause serious injury."),
        WARN("WARNING: Never run the pump dry. Dry running destroys the mechanical seal within seconds and can cause fluid release."),
        WARN("CAUTION: Fluid temperatures above 60 °C can cause burns. Wear heat-resistant gloves and eye protection when working "
             "on the casing."),
        H1("2 Technical specifications"),
        P("Table 1 lists the rated performance data of the HX-200. Do not operate the pump outside these limits."),
        TABLE("Table 1. HX-200 technical specifications", ["Parameter", "Value", "Unit"], HX200_SPECS, [6.5, 4.0, 3.5]),
        PB(),
        H1("3 Installation and start-up"),
        H2("3.1 Installation"),
        P("Mount the pump on a rigid baseplate and level it to within 0.2 mm per metre. Keep the suction pipe as short and straight "
          "as possible, with a straight length of at least five times the pipe diameter upstream of the suction flange. Support the "
          "pipework independently so that no piping loads act on the pump casing."),
        H2("3.2 First start-up"),
        STEPS(["Check that the direction of rotation matches the arrow on the casing.",
               "Fill the casing completely with the process fluid to prime the pump and vent the air.",
               "Open the suction valve fully.",
               "Start the motor with the discharge valve closed, then open the discharge valve slowly until the duty point is reached.",
               "Check the mechanical seal for leakage and the bearings for abnormal temperature during the first hour."]),
        P("Figure 1 shows the main components referenced in this manual."),
        FIG("hx200_exploded", "Figure 1. Exploded view of the HX-200 pump assembly.", 11.5),
        PB(),
        H1("4 Tightening torques"),
        P("Tighten all fasteners with a calibrated torque wrench to the values in Table 2. Tighten in a cross pattern in at least two passes."),
        TABLE("Table 2. HX-200 fastener tightening torques", ["Component", "Fastener", "Torque (Nm)"], HX200_TORQUE, [6.5, 4.0, 3.5]),
        H1("5 Maintenance schedule"),
        P("Carry out the tasks in Table 3 at the stated intervals. The intervals apply to continuous duty with clean water; halve "
          "the intervals when pumping abrasive fluids."),
        TABLE("Table 3. HX-200 maintenance schedule", ["Task", "Interval", "Performed by"], HX200_MAINT, [8.0, 3.0, 3.5]),
        PB(),
        H1("6 Service procedures"),
        H2("6.1 Mechanical seal replacement"),
        P("Replace the mechanical seal when leakage exceeds 10 drops per minute or when alarm E-52 is shown."),
        STEPS(["Lock out and tag out the power supply and close the suction and discharge valves.",
               "Drain the casing through the drain plug and allow the pump to cool below 40 °C.",
               "Remove the coupling guard and the spacer coupling.",
               "Remove the gland bolts and slide the old mechanical seal off the shaft.",
               "Clean the shaft sleeve and inspect it for scoring deeper than 0.1 mm.",
               "Fit the new mechanical seal MS-4 using the supplied assembly lubricant.",
               "Tighten the gland bolts evenly to the torque listed in Table 2.",
               "Refit the coupling guard, open the valves, vent the casing and run a leak test for 10 minutes."]),
        WARN("WARNING: Seal faces are brittle. Never touch the lapped faces with bare hands and never drop the seal."),
        H2("6.2 Impeller replacement"),
        STEPS(["Perform steps 1 to 4 of Section 6.1 to isolate and open the pump.",
               "Remove the casing bolts and withdraw the back pull-out assembly.",
               "Remove the impeller bolt and pull the impeller from the shaft with puller tool PT-200.",
               "Inspect the impeller for erosion and replace it if the vane thickness is below 3 mm.",
               "Fit the new impeller and tighten the impeller bolt to the torque listed in Table 2.",
               "Refit the assembly and tighten the casing bolts to the torque listed in Table 2 in a cross pattern."]),
        PB(),
        H1("7 Troubleshooting"),
        P("Table 4 lists the alarm codes shown on the controller display together with probable causes and remedies."),
        TABLE("Table 4. HX-200 alarm codes", ["Code", "Description", "Probable cause", "Remedy"], HX200_ERRORS, [1.6, 3.6, 5.0, 6.3]),
        P("Figure 2 shows the performance curve of the HX-200 at rated speed. Use it to verify the duty point after any repair."),
        FIG("hx200_curve", "Figure 2. HX-200 performance curve: head versus flow at rated speed.", 10.5),
    ])

    hx500 = ManualSpec("hx500", "HX-500_Service_Manual.pdf", "HX-500", "Rotary Screw Compressor", "B", "2025-11-12", [
        H1("1 Safety"),
        P("This manual is intended for qualified maintenance technicians. Read this chapter before installing, operating or "
          "servicing the HX-500 rotary screw compressor."),
        WARN("WARNING: Lock out and tag out the electrical supply before any maintenance work."),
        WARN("WARNING: The compressor contains pressurised air and hot oil. Fully depressurise the system and wait 10 minutes for the "
             "oil to cool before opening any plug or cover."),
        WARN("WARNING: Never operate the compressor with the guards removed."),
        H1("2 Technical specifications"),
        P("Table 1 lists the rated data of the HX-500. Do not exceed the stated limits."),
        TABLE("Table 1. HX-500 technical specifications", ["Parameter", "Value", "Unit"], HX500_SPECS, [6.5, 4.0, 3.5]),
        PB(),
        H1("3 Installation"),
        H2("3.1 Site requirements"),
        P("Install the compressor in a ventilated room with at least one metre of clearance around the unit. The ambient "
          "temperature must stay between 5 and 40 °C. Route the cooling air outlet so that hot air cannot be drawn back into the intake."),
        H2("3.2 Air and oil flow"),
        P("Ambient air passes through the intake filter into the compressor element where oil is injected for cooling and sealing. "
          "The air-oil mixture flows to the oil separator vessel. Separated oil returns through the cooler and the oil filter while "
          "clean air leaves through the aftercooler to the outlet. Figure 1 shows the flow path."),
        FIG("hx500_flow", "Figure 1. HX-500 air and oil flow diagram.", 12.0),
        PB(),
        H1("4 Tightening torques"),
        P("Tighten all fasteners and plugs with a calibrated torque wrench to the values in Table 2."),
        TABLE("Table 2. HX-500 fastener tightening torques", ["Component", "Fastener", "Torque (Nm)"], HX500_TORQUE, [6.5, 4.0, 3.5]),
        H1("5 Maintenance schedule"),
        P("Carry out the tasks in Table 3 at the stated intervals. Shorten the intervals in dusty or very hot environments."),
        TABLE("Table 3. HX-500 maintenance schedule", ["Task", "Interval", "Performed by"], HX500_MAINT, [8.0, 3.0, 3.5]),
        PB(),
        H1("6 Service procedures"),
        H2("6.1 Oil change"),
        P("Change the oil every 4000 hours or once a year, whichever comes first. Use Helix CompOil 46 only."),
        STEPS(["Run the compressor until it reaches operating temperature, then stop it and lock out the power supply.",
               "Fully depressurise the system and wait 10 minutes.",
               "Place a container of at least 40 L under the drain and remove the oil drain plug.",
               "Allow the oil to drain completely and replace the oil filter (see Table 3).",
               "Refit the oil drain plug and tighten it to the torque listed in Table 2.",
               "Fill with new oil through the filler plug until the level reaches the mark on the sight glass (capacity in Table 1).",
               "Run the compressor for 5 minutes and re-check the oil level."]),
        WARN("WARNING: Hot oil can cause severe burns. Wear heat-resistant gloves and never open the drain plug while the system is pressurised."),
        H2("6.2 Oil separator element replacement"),
        STEPS(["Stop the compressor, lock out the power supply and depressurise the system.",
               "Remove the separator cover bolts and lift off the cover.",
               "Remove the old separator element and clean the vessel.",
               "Fit the new element with new gaskets and refit the cover.",
               "Tighten the separator cover bolts to the torque listed in Table 2 in a cross pattern."]),
        PB(),
        H1("7 Troubleshooting"),
        P("Table 4 lists the alarm codes shown on the controller display together with probable causes and remedies."),
        TABLE("Table 4. HX-500 alarm codes", ["Code", "Description", "Probable cause", "Remedy"], HX500_ERRORS, [1.6, 3.8, 5.0, 6.1]),
        P("Figure 2 shows how the specific power of the HX-500 changes with the load step. The compressor is most efficient at full load."),
        FIG("hx500_bars", "Figure 2. HX-500 specific power versus load step (lower is better).", 10.5),
    ])

    hx900 = ManualSpec("hx900", "HX-900_Service_Manual.pdf", "HX-900", "Hydraulic Power Unit", "A", "2026-01-20", [
        H1("1 Safety"),
        P("This manual is intended for qualified hydraulic technicians. Read this chapter before installing, operating or "
          "servicing the HX-900 hydraulic power unit."),
        WARN("WARNING: Hydraulic fluid under pressure can penetrate skin and cause severe injury. Never check for leaks with your hands. "
             "Release all pressure, including the accumulator pre-charge, before opening any line."),
        WARN("WARNING: Lock out and tag out the electrical supply before any maintenance work."),
        WARN("CAUTION: Oil temperatures above 55 °C can cause burns."),
        H1("2 Technical specifications"),
        P("Table 1 lists the rated data of the HX-900. The relief valve must never be set above the stated value."),
        TABLE("Table 1. HX-900 technical specifications", ["Parameter", "Value", "Unit"], HX900_SPECS, [6.5, 4.0, 3.5]),
        PB(),
        H1("3 Installation and commissioning"),
        H2("3.1 Hydraulic circuit"),
        P("The circuit consists of the tank, the pump, the relief valve, the return filter and the actuator lines. Figure 1 shows the schematic."),
        FIG("hx900_circuit", "Figure 1. HX-900 hydraulic circuit schematic.", 12.0),
        H2("3.2 Bleeding the circuit"),
        STEPS(["Fill the tank to the upper mark with the specified oil.",
               "Open the bleed screw on the pump housing until oil flows free of bubbles.",
               "Start the motor at idle with zero load and cycle each actuator slowly ten times.",
               "Re-check the tank level and top up to the upper mark."]),
        PB(),
        H1("4 Tightening torques"),
        P("Tighten all fasteners with a calibrated torque wrench to the values in Table 2."),
        TABLE("Table 2. HX-900 fastener tightening torques", ["Component", "Fastener", "Torque (Nm)"], HX900_TORQUE, [6.5, 4.0, 3.5]),
        H1("5 Maintenance schedule"),
        P("Carry out the tasks in Table 3 at the stated intervals."),
        TABLE("Table 3. HX-900 maintenance schedule", ["Task", "Interval", "Performed by"], HX900_MAINT, [8.0, 3.0, 3.5]),
        PB(),
        H1("6 Service procedures"),
        H2("6.1 Return filter element replacement"),
        P("Replace the return filter element every 1000 hours or when alarm E-47 is shown."),
        STEPS(["Stop the unit, lock out the power supply and release all system pressure.",
               "Place a drip tray under the filter housing and remove the housing.",
               "Remove the old element and inspect it for metal particles.",
               "Fit a new element with a new O-ring lubricated with clean oil.",
               "Tighten the filter housing to the torque listed in Table 2.",
               "Bleed the circuit as described in Section 3.2 and check for leaks."]),
        H2("6.2 Fluid compatibility"),
        P("Only use fluids that Table 4 marks as compatible."),
        TABLE("Table 4. HX-900 fluid compatibility", ["Fluid", "Compatible", "Note"], HX900_FLUIDS, [6.0, 3.0, 7.0]),
        PB(),
        H1("7 Troubleshooting"),
        P("Table 5 lists the alarm codes shown on the controller display together with probable causes and remedies."),
        TABLE("Table 5. HX-900 alarm codes", ["Code", "Description", "Probable cause", "Remedy"], HX900_ERRORS, [1.6, 3.4, 5.2, 6.3]),
        P("Figure 2 shows how the oil temperature rises during warm-up at rated duty."),
        FIG("hx900_temp", "Figure 2. HX-900 oil temperature versus run time during warm-up.", 10.5),
    ])
    return [hx200, hx500, hx900]


MANUALS = _manuals()
MANUAL_BY_KEY = {m.key: m for m in MANUALS}

# ----------------------------------------------------------------------------- service bulletin
BULLETIN = {
    "filename": "SB-2026-07_HX-200_Impeller_Bolt_Torque.docx",
    "id": "SB-2026-07",
    "date": "2026-03-18",
    "title": "Service Bulletin SB-2026-07",
    "subject": "HX-200: Revised impeller bolt tightening torque",
    "applies": "HX-200 pumps with serial numbers 260400 and higher (production from March 2026).",
    "paragraphs": [
        "Field reports showed impeller bolt loosening under repeated thermal cycling. Engineering has increased the specified "
        "tightening torque for the impeller bolt and added a threadlocker requirement.",
    ],
    "table": {
        "header": ["Component", "Fastener", "Previous torque (Nm)", "Revised torque (Nm)"],
        "rows": [["Impeller bolt", "M12", "45", "52"]],
    },
    "action": "Action required: Tighten the impeller bolt to 52 Nm and apply medium-strength threadlocker (Loctite 243) to the threads. "
              "Pumps with serial numbers below 260400 should be retrofitted at the next scheduled impeller inspection.",
    "supersedes": "This bulletin supersedes the impeller bolt value in Table 2 of the HX-200 Service Manual, revision C.",
    "serial_table": {
        "header": ["Serial range", "Status"],
        "rows": [["260400 and higher", "Revised value applies at delivery"], ["Below 260400", "Retrofit at next impeller inspection"]],
    },
    "warning": "WARNING: Do not exceed 52 Nm. Over-torquing can crack the impeller hub.",
}

# ----------------------------------------------------------------------------- scanned field card
FIELD_CARD_LINES = [
    "HELIX DYNAMICS - HX-200 QUICK FIELD CARD",
    "",
    "Maximum starts per hour: 6",
    "Priming volume (casing fill): 14 L",
    "Minimum continuous flow: 12 m3/h",
    "Seal flush water: 2 L/min",
    "STOP the pump if suction pressure drops below 0.5 bar (alarm E-47)",
    "Never run dry. Never start against a closed suction valve.",
    "Record every alarm in the maintenance log.",
]

# ----------------------------------------------------------------------------- maintenance log
LOG_UNITS = {
    "HX-200": ["HX200-01", "HX200-02", "HX200-03", "HX200-04"],
    "HX-500": ["HX500-01", "HX500-02", "HX500-03"],
    "HX-900": ["HX900-01", "HX900-02"],
}
LOG_CODES = {
    "HX-200": [("E-12", "Cleared blocked impeller"), ("E-31", "Lubricated bearings"), ("E-47", "Cleaned suction strainer"),
               ("E-52", "Replaced mechanical seal"), ("E-63", "Realigned coupling")],
    "HX-500": [("E-05", "Cleaned cooler fins"), ("E-47", "Replaced oil separator element"), ("E-58", "Checked supply voltage"),
               ("E-71", "Refilled oil")],
    "HX-900": [("E-09", "Refilled tank"), ("E-20", "Cleaned cooler"), ("E-47", "Replaced return filter element"),
               ("E-88", "Reset relief valve")],
}
TECHNICIANS = ["A. Rao", "M. Fischer", "S. Okafor", "L. Chen", "P. Novak"]

# ----------------------------------------------------------------------------- page-level text snippets for fixture rendering
DISPLAY_ALARMS = [
    {"file": "photo_alarm_E47_hx200.png", "lines": ["HELIX HX-200", "ALARM  E-47", "LOW SUCTION PRESSURE"]},
    {"file": "photo_alarm_E47_hx500.png", "lines": ["HELIX HX-500", "ALARM  E-47", "SEPARATOR DP HIGH"]},
]
DATA_PLATE = {"file": "photo_dataplate_hx500.png",
              "lines": ["HELIX DYNAMICS", "MODEL HX-500", "SERIAL 480112", "MAX PRESSURE 10 BAR", "YEAR 2025"]}
