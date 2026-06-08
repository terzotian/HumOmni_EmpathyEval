#!/usr/bin/env python3
"""Demo inference on the Task 1 doc example audio folder (Context 1/2, A/B)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.data import Question  # noqa: E402
from empathy_eval.pipeline.predictor import Predictor  # noqa: E402


CONTEXTS = {
    "Context 1": (
        "During a bachelorette weekend, their friends surprised them with a spa package "
        "that included a manicure, and they've never been to a nail salon before."
    ),
    "Context 2": (
        "After losing a fantasy football bet, the agreed penalty was to get a bright, "
        "glittery manicure and keep it for a week."
    ),
}


def build_question(ctx_name: str, audio_dir: Path) -> Question:
    prefix = ctx_name
    return Question(
        question_id=f"demo_{ctx_name.replace(' ', '_').lower()}",
        dataset="demo",
        context=CONTEXTS[ctx_name],
        utterance="I can't believe I'm getting my nails done.",
        utterance_audio=audio_dir / "Utterance Audio.wav",
        reference_response="",
        options={
            "opt-A": audio_dir / f"{prefix}_Candidate Responses_A.wav",
            "opt-B": audio_dir / f"{prefix}_Candidate Responses_B.wav",
        },
        base_dir=audio_dir,
        group_id="demo_task1",
        task="context_variant",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo on Task 1 doc example audios")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PROJECT_ROOT / "Task 1: Audio Supplementation",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/pipeline.yaml",
    )
    parser.add_argument("--use-qwen", action="store_true")
    args = parser.parse_args()

    predictor = Predictor(PROJECT_ROOT, args.config)
    results = []

    for ctx_name in CONTEXTS:
        q = build_question(ctx_name, args.audio_dir)
        pred = predictor.predict_one(q, use_qwen=args.use_qwen, force_features=True)
        results.append(pred.to_dict())
        print(f"\n=== {ctx_name} ===")
        print(f"choice: {pred.choice} (source={pred.source})")
        print(f"rule_winner: {pred.rule_winner}")
        print(f"reason: {pred.reason}")

    out = args.audio_dir / "demo_predictions.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
