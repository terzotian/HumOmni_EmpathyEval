"""Audio loading utilities (no ffmpeg required)."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def load_mono(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio as mono float32, resampled to target_sr."""
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return mono, sr


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
