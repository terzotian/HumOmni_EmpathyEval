#!/usr/bin/env python3
"""
Local Qwen2.5-Omni inference on TEST sets (competition-compliant).

No API calls. Use this for submission predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.compliance import assert_local_inference_only  # noqa: E402
from empathy_eval.data import TEST_DATASETS, load_questions  # noqa: E402
from empathy_eval.omni.inference import OmniEmpathyJudge  # noqa: E402
from empathy_eval.pipeline.predictor import Prediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Omni inference (test data, no API)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/omni.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(TEST_DATASETS),
    )
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    datasets = args.datasets or cfg.get("datasets", list(TEST_DATASETS))
    judge = OmniEmpathyJudge(cfg, PROJECT_ROOT)

    predictions: list[Prediction] = []

    if args.question_id:
        if not args.dataset:
            raise SystemExit("--dataset required with --question-id")
        assert_local_inference_only(dataset=args.dataset)
        from empathy_eval.data import load_single_question

        q = load_single_question(PROJECT_ROOT, args.dataset, args.question_id)
        result = judge.predict_question(q)
        pred = Prediction(
            question_id=q.question_id,
            dataset=q.dataset,
            group_id=q.group_id,
            choice=result["choice"],
            source=result["source"],
            reason=result.get("reason", ""),
            rule_winner=result["choice"],
            rule_scores=[],
        )
        print(json.dumps(pred.to_dict(), indent=2, ensure_ascii=False))
        return

    for ds in datasets:
        assert_local_inference_only(dataset=ds)
        questions = list(load_questions(PROJECT_ROOT, [ds]))
        if args.limit:
            questions = questions[: args.limit]

        out_dir = PROJECT_ROOT / "outputs/predictions" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for q in tqdm(questions, desc=f"omni:{ds}"):
                result = judge.predict_question(q)
                pred = Prediction(
                    question_id=q.question_id,
                    dataset=q.dataset,
                    group_id=q.group_id,
                    choice=result["choice"],
                    source=result["source"],
                    reason=result.get("reason", ""),
                    rule_winner=result["choice"],
                    rule_scores=[],
                )
                predictions.append(pred)
                f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")

        print(f"[{ds}] {len(questions)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
