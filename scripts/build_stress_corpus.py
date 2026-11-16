#!/usr/bin/env python3
"""Build a deterministic synthetic scale corpus and held-out benchmark."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.config import PROJECT_ROOT  # noqa: E402
from meridian.eval.stress import build_stress_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=100, help="number of generated model-specific documents")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "demo_corpus_stress")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "eval_data_stress")
    args = parser.parse_args()
    manifest = build_stress_corpus(args.corpus_dir, args.eval_dir, docs=args.docs, seed=args.seed)
    print(
        f"Generated {manifest['documents']} documents and {manifest['total_questions']} questions "
        f"({manifest['train_questions']} train, {manifest['test_questions']} test)."
    )
    print("Corpus:", args.corpus_dir)
    print("Evaluation data:", args.eval_dir)


if __name__ == "__main__":
    main()
