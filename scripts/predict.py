#!/usr/bin/env python3
"""Run EmpathyEval inference: rules baseline or rules + Qwen API judge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.data import TEST_DATASETS, load_single_question  # noqa: E402
from empathy_eval.pipeline.predictor import Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict best empathy response option")
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
    parser.add_argument("--question-id", type=str, default=None, help="Run a single question")
    parser.add_argument("--dataset", type=str, default=None, help="Required with --question-id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-qwen", action="store_true", help="Enable Qwen API final judge")
    parser.add_argument("--no-qwen", action="store_true", help="Force rules-only even if config enables Qwen")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="Print predictions to stdout")
    args = parser.parse_args()

    use_qwen = None
    if args.use_qwen:
        use_qwen = True
    if args.no_qwen:
        use_qwen = False

    predictor = Predictor(PROJECT_ROOT, args.config)

    if args.question_id:
        if not args.dataset:
            raise SystemExit("--dataset is required when using --question-id")
        q = load_single_question(PROJECT_ROOT, args.dataset, args.question_id)
        pred = predictor.predict_one(q, use_qwen=use_qwen, force_features=args.force_features)
        print(json.dumps(pred.to_dict(), indent=2, ensure_ascii=False))
        return

    for dataset in args.datasets:
        preds = predictor.predict_dataset(
            dataset,
            limit=args.limit,
            use_qwen=use_qwen,
            force_features=args.force_features,
        )
        out_path = predictor.save_predictions(dataset, preds)
        print(f"[{dataset}] {len(preds)} predictions -> {out_path}")
        if args.print_json:
            for p in preds:
                print(json.dumps(p.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
