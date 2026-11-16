"""Shared helpers for the command-line scripts."""
from meridian.bootstrap import load_demo
from meridian.config import Settings
from meridian.kb import KnowledgeBase


def ensure_demo_kb(s: Settings) -> KnowledgeBase:
    kb = KnowledgeBase(s)
    if not kb.chunks:
        print("Ingesting the demo corpus ...")
        load_demo(kb, s)
    return kb
