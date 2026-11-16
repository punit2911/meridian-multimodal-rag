#!/usr/bin/env python3
"""Run the full Meridian benchmark. Usage: python scripts/run_eval.py [--no-train]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.config import Settings  # noqa: E402
from meridian.eval.report import run_full_evaluation  # noqa: E402
from scripts_common import ensure_demo_kb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true", help="skip LTR training (use existing weights)")
    ap.add_argument("--calibration-path", type=Path, help="write adopted calibration here instead of changing the shared default")
    args = ap.parse_args()
    s = Settings()
    kb = ensure_demo_kb(s)
    run_full_evaluation(kb, s, train_models=not args.no_train, calibration_path=args.calibration_path)


if __name__ == "__main__":
    main()
