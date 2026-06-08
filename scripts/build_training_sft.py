#!/usr/bin/env python3
"""Build SFT jsonl from training goodPara/badPara pairs (no API, no test data)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from empathy_eval.training_data import load_training_examples, save_training_examples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training SFT dataset")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/omni.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    tasks = train_cfg.get("tasks", ["context_variant", "tone_variant"])
    limit = args.limit or train_cfg.get("limit")

    examples = load_training_examples(PROJECT_ROOT, tasks=tasks, limit=limit)
    out_raw = PROJECT_ROOT / "outputs/training/training_choices.jsonl"
    out_sft = PROJECT_ROOT / train_cfg.get("sft_dataset", "outputs/training/sft_train.jsonl")
    save_training_examples(examples, out_raw)

    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with out_sft.open("w", encoding="utf-8") as f:
        for ex in examples:
            d = ex.to_dict()
            f.write(
                json.dumps(
                    {
                        "id": d["example_id"],
                        "task": d["task"],
                        "messages": [
                            {
                                "role": "user",
                                "instruction": "Select the most empathetic response (A or B).",
                                "context": d["context"],
                                "utterance_text": d["utterance_text"],
                                "utterance_audio": d["utterance_audio"],
                                "option_a_audio": d["options"]["opt-A"],
                                "option_b_audio": d["options"]["opt-B"],
                            },
                            {"role": "assistant", "content": d["answer"]},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Training choice examples: {len(examples)}")
    print(f"  raw: {out_raw}")
    print(f"  sft: {out_sft}")


if __name__ == "__main__":
    main()
