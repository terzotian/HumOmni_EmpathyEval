#!/usr/bin/env python3
"""Extract Layer-1 features for test questions and cache to outputs/features/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.data import TEST_DATASETS, load_questions  # noqa: E402
from empathy_eval.pipeline.predictor import Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features for EmpathyEval test sets")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/pipeline.yaml",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(TEST_DATASETS),
        choices=list(TEST_DATASETS),
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions per dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite cached features")
    args = parser.parse_args()

    predictor = Predictor(PROJECT_ROOT, args.config)

    for dataset in args.datasets:
        questions = list(load_questions(PROJECT_ROOT, [dataset]))
        if args.limit:
            questions = questions[: args.limit]

        print(f"\n[{dataset}] extracting {len(questions)} questions...")
        for q in tqdm(questions):
            predictor.extract_and_cache(q, force=args.force)

        print(f"Features saved under: {predictor.features_dir / dataset}")


if __name__ == "__main__":
    main()
