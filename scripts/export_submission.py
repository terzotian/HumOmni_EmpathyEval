#!/usr/bin/env python3
"""Export official Track 1 submission jsonl from pipeline predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.data import TEST_DATASETS  # noqa: E402
from empathy_eval.submission import export_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Track 1 submission jsonl")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/predictions",
        help="Directory containing {dataset}/predictions.jsonl",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(TEST_DATASETS),
        choices=list(TEST_DATASETS),
        help="Datasets to include (default: all test subsets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/submission/track1_submission.jsonl",
    )
    args = parser.parse_args()

    pred_files = [args.predictions_dir / ds / "predictions.jsonl" for ds in args.datasets]
    counts = export_submission(pred_files, args.output)

    total = sum(counts.values())
    print(f"Wrote {total} rows -> {args.output}")
    for ds, n in sorted(counts.items()):
        print(f"  {ds}: {n}")
    print("\nPreview (first 3 lines):")
    for i, line in enumerate(args.output.read_text(encoding="utf-8").splitlines()):
        if i >= 3:
            break
        print(f"  {line}")


if __name__ == "__main__":
    main()
