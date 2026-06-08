"""Paralinguistic naturalness scoring (heuristic + optional classifier checkpoint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from empathy_eval.features.prosody import expressiveness_score, extract_prosody

_CHECKPOINT_CACHE: dict[str, tuple[nn.Module, str]] = {}


class ParaClassifier(nn.Module):
    """Small classifier head on frozen wav2vec2 features."""

    def __init__(self, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(768, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _load_para_checkpoint(checkpoint_path: Path) -> tuple[nn.Module, str]:
    key = str(checkpoint_path.resolve())
    if key in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[key]

    from transformers import Wav2Vec2Model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone", "facebook/wav2vec2-base")
    backbone = Wav2Vec2Model.from_pretrained(backbone_name, use_safetensors=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    clf = ParaClassifier()
    clf.load_state_dict(ckpt["classifier"])
    clf.eval()

    bundle = (nn.ModuleDict({"backbone": backbone, "clf": clf}), backbone_name)
    _CHECKPOINT_CACHE[key] = bundle
    return bundle


@torch.no_grad()
def para_classifier_score(path: str | Path, checkpoint_path: str | Path) -> float | None:
    """
    Return P(goodPara) in [0,1], or None if checkpoint missing.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    from empathy_eval.audio import load_mono

    bundle, _ = _load_para_checkpoint(checkpoint_path)
    backbone: nn.Module = bundle["backbone"]
    clf: nn.Module = bundle["clf"]

    audio, sr = load_mono(path)
    if len(audio) < sr * 0.05:
        return 0.0

    inputs = torch.tensor(audio).unsqueeze(0)
    hidden = backbone(inputs).last_hidden_state.mean(dim=1)
    logits = clf(hidden)
    probs = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(probs)


def naturalness_score(
    path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    prosody = extract_prosody(path)
    heuristic = expressiveness_score(prosody)
    para_prob = None
    if checkpoint_path:
        para_prob = para_classifier_score(path, checkpoint_path)

    if para_prob is not None:
        combined = 0.7 * para_prob + 0.3 * heuristic
    else:
        combined = heuristic

    return {
        "naturalness_heuristic": heuristic,
        "naturalness_para_prob": para_prob if para_prob is not None else -1.0,
        "naturalness_combined": combined,
        **{f"prosody_{k}": v for k, v in prosody.items()},
    }
