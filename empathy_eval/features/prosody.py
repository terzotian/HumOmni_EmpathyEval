"""Lightweight prosody features via librosa."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from empathy_eval.audio import load_mono


def extract_prosody(path: str | Path) -> dict[str, float]:
    y, sr = load_mono(path)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) < sr * 0.05:
        yt = y

    f0, _, _ = librosa.pyin(yt, fmin=80, fmax=400, sr=sr)
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    rms = librosa.feature.rms(y=yt)[0]
    zcr = librosa.feature.zero_crossing_rate(yt)[0]

    tempo_raw, _ = librosa.beat.beat_track(y=yt, sr=sr)
    tempo = float(np.asarray(tempo_raw).item())

    return {
        "duration_sec": float(len(yt) / sr),
        "f0_mean_hz": float(np.mean(f0_valid)) if len(f0_valid) else 0.0,
        "f0_std_hz": float(np.std(f0_valid)) if len(f0_valid) else 0.0,
        "f0_range_hz": float(np.ptp(f0_valid)) if len(f0_valid) else 0.0,
        "voiced_ratio": float(np.mean(~np.isnan(f0))) if f0 is not None else 0.0,
        "rms_std": float(np.std(rms)),
        "rms_mean": float(np.mean(rms)),
        "zcr_std": float(np.std(zcr)),
        "tempo_bpm": tempo,
    }


def expressiveness_score(prosody: dict[str, float]) -> float:
    """
    Heuristic 0-1 score: more pitch/energy variation suggests human expressiveness.
    Not reliable alone; combine with para classifier or LLM judge.
    """
    parts = [
        min(prosody.get("f0_std_hz", 0) / 80.0, 1.0),
        min(prosody.get("f0_range_hz", 0) / 250.0, 1.0),
        min(prosody.get("rms_std", 0) / 0.08, 1.0),
        min(prosody.get("zcr_std", 0) / 0.15, 1.0),
        prosody.get("voiced_ratio", 0),
    ]
    return float(np.mean(parts))
