"""Quantity parsing and unit conversion (engineering units).

The verifier, calculator tool, table-fact miner and chart reader all share this module so that
"45 Nm", "45 N·m" and "33.2 lb-ft" are recognised as comparable quantities.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

from ..schemas import Quantity

# canonical unit -> (dimension, factor to the dimension's base unit, display)
UNITS: dict[str, tuple[str, float, str]] = {
    # torque (base: Nm)
    "nm": ("torque", 1.0, "Nm"),
    "lbft": ("torque", 1.3558179483314004, "lb-ft"),
    "lbin": ("torque", 0.1129848290276167, "lb-in"),
    # pressure (base: bar)
    "bar": ("pressure", 1.0, "bar"),
    "psi": ("pressure", 0.0689475729317831, "psi"),
    "mpa": ("pressure", 10.0, "MPa"),
    "kpa": ("pressure", 0.01, "kPa"),
    "pa": ("pressure", 1e-5, "Pa"),
    # power (base: kW)
    "kw": ("power", 1.0, "kW"),
    "w": ("power", 0.001, "W"),
    "hp": ("power", 0.745699872, "hp"),
    # flow (base: m3/h)
    "m3/h": ("flow", 1.0, "m³/h"),
    "m3/min": ("flow", 60.0, "m³/min"),
    "l/min": ("flow", 0.06, "L/min"),
    "l/s": ("flow", 3.6, "L/s"),
    "gpm": ("flow", 0.227124707, "gpm"),
    # volume (base: L)
    "l": ("volume", 1.0, "L"),
    "ml": ("volume", 0.001, "mL"),
    "m3": ("volume", 1000.0, "m³"),
    "gal": ("volume", 3.785411784, "gal"),
    # mass (base: kg)
    "kg": ("mass", 1.0, "kg"),
    "g": ("mass", 0.001, "g"),
    "lb": ("mass", 0.45359237, "lb"),
    "t": ("mass", 1000.0, "t"),
    # length (base: mm)
    "mm": ("length", 1.0, "mm"),
    "cm": ("length", 10.0, "cm"),
    "m": ("length", 1000.0, "m"),
    "in": ("length", 25.4, "in"),
    "ft": ("length", 304.8, "ft"),
    "um": ("length", 0.001, "µm"),
    # temperature (affine; base: °C)
    "c": ("temperature", 1.0, "°C"),
    "f": ("temperature", 1.0, "°F"),
    "k": ("temperature", 1.0, "K"),
    # misc
    "rpm": ("speed", 1.0, "rpm"),
    "dba": ("noise", 1.0, "dB(A)"),
    "db": ("noise", 1.0, "dB"),
    "hz": ("frequency", 1.0, "Hz"),
    "v": ("voltage", 1.0, "V"),
    "a": ("current", 1.0, "A"),
    "%": ("percent", 1.0, "%"),
    "h": ("time", 1.0, "h"),
    "min": ("time", 1 / 60, "min"),
    "s": ("time", 1 / 3600, "s"),
    "cst": ("viscosity", 1.0, "cSt"),
    "ppm": ("ratio", 1.0, "ppm"),
}

# surface form (lowercase, after normalisation) -> canonical key
_SURFACE: dict[str, str] = {
    "nm": "nm", "n·m": "nm", "n.m": "nm", "n-m": "nm", "newton-metre": "nm", "newton-meter": "nm",
    "lb-ft": "lbft", "lb·ft": "lbft", "lbf-ft": "lbft", "lbf·ft": "lbft", "ft-lb": "lbft", "ft-lbs": "lbft",
    "ft·lb": "lbft", "ft-lbf": "lbft", "lbft": "lbft", "lb-in": "lbin", "in-lb": "lbin", "lbf-in": "lbin",
    "bar": "bar", "bara": "bar", "barg": "bar", "psi": "psi", "psig": "psi", "mpa": "mpa", "kpa": "kpa", "pa": "pa",
    "kw": "kw", "w": "w", "hp": "hp",
    "m3/h": "m3/h", "m3/hr": "m3/h", "m3h": "m3/h", "m3/min": "m3/min", "l/min": "l/min", "lpm": "l/min",
    "l/s": "l/s", "gpm": "gpm",
    "l": "l", "ml": "ml", "m3": "m3", "gal": "gal", "litre": "l", "liter": "l", "litres": "l", "liters": "l",
    "kg": "kg", "g": "g", "lb": "lb", "lbs": "lb",
    "mm": "mm", "cm": "cm", "m": "m", "in": "in", "ft": "ft", "um": "um",
    "°c": "c", "degc": "c", "°f": "f", "degf": "f", "k": "k",
    "rpm": "rpm", "db(a)": "dba", "dba": "dba", "db": "db", "hz": "hz", "v": "v", "a": "a", "%": "%",
    "h": "h", "hr": "h", "hrs": "h", "hours": "h", "hour": "h", "min": "min", "mins": "min", "minutes": "min",
    "s": "s", "sec": "s", "cst": "cst", "ppm": "ppm",
}

_SINGLE_LETTER = {"a", "s", "k", "f", "c", "t"}  # ambiguous in prose ("5 a day"); h, m, g, w, v, l are accepted

_UNIT_ALTS = sorted(
    {re.escape(s) for s in _SURFACE if s not in {"°c", "°f"}} | {r"°\s?c", r"°\s?f"},
    key=len,
    reverse=True,
)
_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
_UNIT_RE = r"(?P<unit>" + "|".join(_UNIT_ALTS) + r")"

_RANGE_RE = re.compile(
    rf"(?<![\w\-./])(?P<a>{_NUM})\s*[-–—]\s*(?P<b>{_NUM})(?:\s*{_UNIT_RE}(?![A-Za-z0-9]))?", re.IGNORECASE
)
_QTY_RE = re.compile(
    rf"(?<![\w\-./])(?P<sign>[-−+])?(?P<num>{_NUM})(?:(?P<space>\s*){_UNIT_RE}(?![A-Za-z0-9]))?", re.IGNORECASE
)
_LABEL_BEFORE = re.compile(
    r"(?:\b(?:table|figure|fig\.?|section|sec\.?|step|page|pg\.?|item|no\.?|rev(?:ision)?|sb|chapter|e|f|hx|pn)|§|#)\s*[-]?\s*$",
    re.IGNORECASE,
)


def normalize_unit_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("µ", "u").replace("μ", "u").replace("−", "-")


def _to_float(s: str) -> float:
    s = s.strip()
    if "," in s and "." not in s:
        parts = s.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    return float(s)


def canonical_unit(surface: Optional[str]) -> Optional[str]:
    if not surface:
        return None
    s = normalize_unit_text(surface).lower().replace(" ", "")
    s = s.replace("°", "°")
    return _SURFACE.get(s)


def display_unit(canon: Optional[str]) -> str:
    if not canon:
        return ""
    return UNITS.get(canon, ("", 1.0, canon))[2]


def unit_dimension(canon: Optional[str]) -> Optional[str]:
    if not canon or canon not in UNITS:
        return None
    return UNITS[canon][0]


def to_base(value: float, canon: Optional[str]) -> tuple[Optional[str], float]:
    """Return (dimension, value in the dimension's base unit)."""
    if not canon or canon not in UNITS:
        return None, value
    dim, factor, _ = UNITS[canon]
    if dim == "temperature":
        if canon == "f":
            return dim, (value - 32.0) * 5.0 / 9.0
        if canon == "k":
            return dim, value - 273.15
        return dim, value
    return dim, value * factor


def from_base(value: float, canon: str) -> float:
    dim, factor, _ = UNITS[canon]
    if dim == "temperature":
        if canon == "f":
            return value * 9.0 / 5.0 + 32.0
        if canon == "k":
            return value + 273.15
        return value
    return value / factor


def convert(value: float, src: str, dst: str) -> float:
    s, d = canonical_unit(src) or src, canonical_unit(dst) or dst
    if s not in UNITS or d not in UNITS:
        raise ValueError(f"Unknown unit: {src!r} or {dst!r}")
    if UNITS[s][0] != UNITS[d][0]:
        raise ValueError(f"Cannot convert {UNITS[s][0]} to {UNITS[d][0]}")
    _, base = to_base(value, s)
    return from_base(base, d)


def format_number(value: float, max_decimals: int = 2) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return text


def format_quantity(value: float, unit: Optional[str]) -> str:
    u = display_unit(unit)
    return f"{format_number(value)} {u}".strip() if u else format_number(value)


def extract_quantities(text: str, include_unitless: bool = True) -> list[Quantity]:
    """Extract numbers with optional units. Identifier-like numbers (HX-200, E-47, SB-2026-07) are skipped."""
    text = normalize_unit_text(text)
    out: list[Quantity] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= cs or s >= ce) for cs, ce in consumed)

    for m in _RANGE_RE.finditer(text):
        raw_unit = m.group("unit")
        if not raw_unit:
            continue
        unit = canonical_unit(raw_unit)
        if unit is None:
            continue
        a, b = _to_float(m.group("a")), _to_float(m.group("b"))
        out.append(Quantity(value=a, unit=unit, raw=m.group(0), start=m.start(), end=m.end()))
        out.append(Quantity(value=b, unit=unit, raw=m.group(0), start=m.start(), end=m.end()))
        consumed.append((m.start(), m.end()))

    for m in _QTY_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        raw_unit = m.group("unit")
        unit = canonical_unit(raw_unit) if raw_unit else None
        if raw_unit and unit is None:
            raw_unit = None
        if raw_unit and unit in _SINGLE_LETTER and m.group("space") and raw_unit.islower():
            unit = None  # "5 a" / "12 m" in prose is not a quantity
        if unit is None:
            prefix = text[max(0, m.start() - 12): m.start()]
            if _LABEL_BEFORE.search(prefix):
                continue
            if not include_unitless:
                continue
        try:
            value = _to_float(m.group("num"))
        except ValueError:
            continue
        if m.group("sign") in ("-", "−"):
            value = -value
        end = m.end() if unit else m.start() + len(m.group("sign") or "") + len(m.group("num"))
        out.append(Quantity(value=value, unit=unit, raw=text[m.start():end], start=m.start(), end=end))
    out.sort(key=lambda q: q.start)
    return out


def quantities_match(a: Quantity, b: Quantity, rel_tol: float = 1e-3, abs_tol: float = 1e-9) -> bool:
    """True when two quantities describe the same magnitude (unit-aware)."""
    da, va = to_base(a.value, a.unit)
    db, vb = to_base(b.value, b.unit)
    if a.unit and b.unit:
        if da != db:
            return False
        tol = max(rel_tol, 0.0)
        return abs(va - vb) <= max(abs_tol, tol * max(abs(va), abs(vb)))
    # at least one side is unit-less: compare raw numbers
    return abs(a.value - b.value) <= max(abs_tol, rel_tol * max(abs(a.value), abs(b.value)))


def iter_numbers(qs: Iterable[Quantity]) -> list[float]:
    return [q.value for q in qs]
