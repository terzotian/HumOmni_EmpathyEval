#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-Omni-3B on TRAINING goodPara/badPara pairs.

Heavy on Mac 48G — start with train.limit in configs/omni.yaml (e.g. 100).
Test inference uses run_omni_inference.py with lora_adapter path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.omni.prompts import build_training_instruction  # noqa: E402
from empathy_eval.training_data import load_training_examples  # noqa: E402


class OmniSFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx].to_dict()


def _build_conversation(example: dict) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": build_training_instruction(example)},
        {"type": "audio", "audio": example["utterance_audio"]},
        {"type": "text", "text": "Candidate opt-A:"},
        {"type": "audio", "audio": example["options"]["opt-A"]},
        {"type": "text", "text": "Candidate opt-B:"},
        {"type": "audio", "audio": example["options"]["opt-B"]},
    ]
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-Omni on training data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/omni.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg.get("model_id", "Qwen/Qwen2.5-Omni-3B")
    train_cfg = cfg.get("train", {})
    limit = args.limit or train_cfg.get("limit")
    tasks = train_cfg.get("tasks", ["context_variant", "tone_variant"])

    examples = load_training_examples(PROJECT_ROOT, tasks=tasks, limit=limit)
    if not examples:
        raise SystemExit("No training examples found.")

    print(f"Training examples: {len(examples)}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    dtype = torch.float16 if device == "mps" else torch.float32
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        enable_audio_output=False,
    ).to(device)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    lora = LoraConfig(
        r=int(train_cfg.get("lora_r", 8)),
        lora_alpha=int(train_cfg.get("lora_alpha", 16)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 2e-4)))

    epochs = int(train_cfg.get("epochs", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, ex in enumerate(tqdm(examples, desc=f"epoch {epoch}"), start=1):
            d = ex.to_dict()
            conv = _build_conversation(d)
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            ).to(device)

            labels = inputs["input_ids"].clone()
            answer = d["answer"]
            answer_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids
            labels[:] = -100
            if len(answer_ids) <= labels.shape[1]:
                labels[0, -len(answer_ids) :] = torch.tensor(answer_ids, device=device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_loss += loss.item()

            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(examples) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"epoch {epoch} avg_loss={total_loss / max(len(examples), 1):.4f}")

    out_dir = PROJECT_ROOT / train_cfg.get("output_dir", "checkpoints/omni_lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    meta = {
        "model_id": model_id,
        "num_examples": len(examples),
        "tasks": tasks,
        "device": device,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter -> {out_dir}")
    print("Set configs/omni.yaml lora_adapter to this path for inference.")


if __name__ == "__main__":
    main()
