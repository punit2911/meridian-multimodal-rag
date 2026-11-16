#!/usr/bin/env python3
"""Regenerate the synthetic demo corpus (PDF/DOCX/CSV/scans/photos) and the golden question set. Needs reportlab (requirements-dev.txt)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.config import Settings  # noqa: E402
from meridian.eval.datagen import build_demo_corpus  # noqa: E402

s = Settings()
build_demo_corpus(s.demo_dir, s.eval_dir, s.data_dir / "_work")
print("corpus written to", s.demo_dir, "and golden questions to", s.eval_dir)
