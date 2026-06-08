#!/usr/bin/env python3
"""
Use Qwen API to enrich TRAINING examples only (competition-compliant).

Never pass test-set paths to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from empathy_eval.compliance import assert_api_training_only  # noqa: E402
from empathy_eval.training_data import load_training_examples  # noqa: E402


def _call_qwen(prompt: str, model: str, temperature: float) -> str:
    from dashscope import Generation
    import dashscope

    dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    resp = Generation.call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        temperature=temperature,
    )
    if getattr(resp, "status_code", None) != 200:
        raise RuntimeError(getattr(resp, "message", resp))
    return resp.output.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(description="API labels for training data only")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/omni.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY not set in environment/.env")

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["api_training"]

    if not cfg.get("enabled", False):
        print("api_training.enabled is false in configs/omni.yaml")
        print("Set enabled: true to run API enrichment on training data.")
        raise SystemExit(0)

    limit = args.limit or cfg.get("limit", 50)
    examples = load_training_examples(PROJECT_ROOT, limit=limit)

    out_path = PROJECT_ROOT / cfg.get("output", "outputs/training/api_labels.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            paths = [
                str(ex.utterance_audio),
                str(ex.options["opt-A"]),
                str(ex.options["opt-B"]),
            ]
            assert_api_training_only(paths=paths, operation="generate_training_labels")

            prompt = (
                "Training annotation task (NOT test data).\n"
                f"Context: {ex.context}\n"
                f"User utterance: {ex.utterance_text}\n"
                f"Reference response: {ex.reference_response}\n"
                "Two candidate spoken responses exist: A and B.\n"
                f"Known label: {ex.answer} is the more empathetic (goodPara) option.\n"
                "Explain in 2-3 sentences WHY the good option fits context and tone better."
            )
            rationale = _call_qwen(prompt, cfg.get("model", "qwen-plus"), cfg.get("temperature", 0.1))
            row = {
                "example_id": ex.example_id,
                "answer": ex.answer,
                "rationale": rationale,
                "paths": paths,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"labeled {ex.example_id}")

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
