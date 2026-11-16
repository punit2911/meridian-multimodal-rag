"""Helpers shared by the CLI scripts and the server for first-run setup."""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .kb import KnowledgeBase


def ensure_demo_corpus(s: Settings) -> None:
    """Generate the demo corpus and golden question set if they are missing."""
    if not (s.eval_dir / "golden_qa.json").exists() or not (s.demo_dir.exists() and any(s.demo_dir.iterdir())):
        from .eval.datagen import build_demo_corpus
        build_demo_corpus(s.demo_dir, s.eval_dir, s.data_dir / "_work")


def load_demo(kb: KnowledgeBase, s: Settings) -> list[dict]:
    ensure_demo_corpus(s)
    return kb.add_directory(Path(s.demo_dir))
