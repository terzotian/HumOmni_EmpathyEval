"""Local Qwen2.5-Omni inference for empathy option selection."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from empathy_eval.compliance import assert_local_inference_only
from empathy_eval.data import Question
from empathy_eval.omni.prompts import build_selection_instruction


def _parse_answer(text: str, valid_letters: set[str]) -> str | None:
    text = text.strip()
    for letter in valid_letters:
        if re.search(rf"\b{letter}\b", text):
            return letter
    m = re.search(r"\b([ABC])\b", text)
    return m.group(1) if m else None


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


class OmniEmpathyJudge:
    """Load Qwen2.5-Omni locally and choose the best response option."""

    def __init__(self, cfg: dict[str, Any], project_root: Path):
        self.cfg = cfg
        self.project_root = project_root
        self.model_id = cfg.get("model_id", "Qwen/Qwen2.5-Omni-3B")
        self.device = _resolve_device(cfg.get("device", "auto"))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 16))
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return

        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "enable_audio_output": False,  # text-only; saves ~2GB
        }
        if self.device == "mps":
            kwargs["device_map"] = None
        else:
            kwargs["device_map"] = "auto"

        lora_path = self.cfg.get("lora_adapter")
        if lora_path:
            lora_path = self.project_root / lora_path
            if lora_path.exists():
                from peft import PeftModel

                base = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **kwargs,
                )
                self._model = PeftModel.from_pretrained(base, str(lora_path))
            else:
                print(f"[omni] LoRA path missing: {lora_path}; using base model.")
                self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **kwargs,
                )
        else:
            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_id,
                **kwargs,
            )

        if self.device == "mps":
            self._model = self._model.to("mps")

        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()

        self._processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        self._model.eval()

    def _build_conversation(self, question: Question) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": build_selection_instruction(question)},
            {"type": "audio", "audio": str(question.utterance_audio)},
        ]
        for key in sorted(question.options):
            content.append({"type": "text", "text": f"Now listen to {key}:"})
            content.append({"type": "audio", "audio": str(question.options[key])})

        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def predict_question(self, question: Question) -> dict[str, Any]:
        assert_local_inference_only(
            dataset=question.dataset,
            paths=[str(question.utterance_audio), *[str(p) for p in question.options.values()]],
        )

        self._lazy_load()
        from qwen_omni_utils import process_mm_info

        conversation = self._build_conversation(question)
        text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        device = self.device if self.device != "auto" else next(self._model.parameters()).device
        inputs = inputs.to(device)
        if hasattr(self._model, "dtype"):
            inputs = inputs.to(self._model.dtype)

        output = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_audio_in_video=False,
            return_audio=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        decoded = self._processor.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        valid = {k.replace("opt-", "") for k in question.options}
        letter = _parse_answer(decoded, valid) or sorted(valid)[0]
        choice = f"opt-{letter}"

        return {
            "choice": choice,
            "source": "omni_local",
            "reason": decoded,
            "model_id": self.model_id,
        }
