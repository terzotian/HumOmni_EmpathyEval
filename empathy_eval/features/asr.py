"""Whisper-based speech recognition."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import whisper

from empathy_eval.audio import load_mono


@lru_cache(maxsize=1)
def _load_whisper(model_name: str):
    return whisper.load_model(model_name)


def transcribe(path: str | Path, model_name: str = "base", language: str = "en") -> str:
    audio, _ = load_mono(path)
    model = _load_whisper(model_name)
    result = model.transcribe(audio, language=language, fp16=False)
    return result["text"].strip()
