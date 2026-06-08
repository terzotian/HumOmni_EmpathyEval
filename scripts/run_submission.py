#!/usr/bin/env python3
"""
Compliant end-to-end submission pipeline (LOCAL inference only).

Stages:
  1) optional: train para classifier
  2) optional: build training SFT / API labels / LoRA (training only)
  3) local inference on all test subsets
  4) export official submission jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compliant submission pipeline")
    parser.add_argument(
        "--backend",
        choices=["omni", "rules"],
        default="omni",
        help="omni=local Qwen2.5-Omni (recommended); rules=feature baseline",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit per dataset (debug)")
    parser.add_argument("--skip-para-train", action="store_true")
    parser.add_argument("--train-para", action="store_true", help="Train para classifier first")
    parser.add_argument("--build-sft", action="store_true", help="Build training SFT jsonl")
    parser.add_argument("--train-lora", action="store_true", help="LoRA fine-tune Omni on training data")
    args = parser.parse_args()

    py = sys.executable

    if args.train_para:
        _run([py, "scripts/train_para_classifier.py"], "Train goodPara classifier (local)")

    if args.build_sft:
        _run([py, "scripts/build_training_sft.py"], "Build training SFT dataset (local)")

    if args.train_lora:
        _run([py, "scripts/train_omni_lora.py"], "LoRA fine-tune Omni (training data only)")

    if args.backend == "omni":
        cmd = [py, "scripts/run_omni_inference.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        _run(cmd, "Local Omni inference on TEST sets (no API)")
    else:
        cmd = [py, "scripts/predict.py", "--no-qwen"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        _run(cmd, "Rules baseline on TEST sets (no API)")

    _run([py, "scripts/export_submission.py"], "Export official submission jsonl")

    out = PROJECT_ROOT / "outputs/submission/track1_submission.jsonl"
    print(f"\nDone. Upload this file to Google Drive:\n  {out}")
    print("Reminder: never use API on test data for submission.")


if __name__ == "__main__":
    main()
