#!/usr/bin/env python3
"""Train goodPara vs badPara classifier on training response_audio."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from empathy_eval.audio import load_mono  # noqa: E402
from empathy_eval.features.naturalness import ParaClassifier  # noqa: E402


class ParaAudioDataset(Dataset):
    def __init__(self, files: list[tuple[Path, int]], max_len: int = 160_000):
        self.files = files
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path, label = self.files[idx]
        audio, _ = load_mono(path)
        if len(audio) > self.max_len:
            start = random.randint(0, len(audio) - self.max_len)
            audio = audio[start : start + self.max_len]
        return torch.tensor(audio), torch.tensor(label, dtype=torch.long)


def collect_files(audio_dirs: list[Path], max_per_class: int | None) -> list[tuple[Path, int]]:
    good, bad = [], []
    for d in audio_dirs:
        for p in d.glob("*goodPara.wav"):
            good.append((p, 1))
        for p in d.glob("*badPara.wav"):
            bad.append((p, 0))

    random.shuffle(good)
    random.shuffle(bad)
    if max_per_class:
        good = good[:max_per_class]
        bad = bad[:max_per_class]

    files = good + bad
    random.shuffle(files)
    print(f"Collected goodPara={len(good)}, badPara={len(bad)}")
    return files


def split_files(files: list[tuple[Path, int]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train goodPara vs badPara classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/pipeline.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train_para"]

    audio_dirs = [PROJECT_ROOT / p for p in cfg["audio_dirs"]]
    for d in audio_dirs:
        if not d.exists():
            raise SystemExit(f"Missing training audio dir: {d}")

    files = collect_files(audio_dirs, cfg.get("max_samples_per_class"))
    train_files, val_files = split_files(files, cfg["val_ratio"], cfg["seed"])

    from transformers import Wav2Vec2Model

    backbone_name = cfg["backbone"]
    backbone = Wav2Vec2Model.from_pretrained(backbone_name, use_safetensors=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    clf = ParaClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    clf.to(device)

    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    optimizer = torch.optim.AdamW(clf.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(ParaAudioDataset(train_files), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ParaAudioDataset(val_files), batch_size=batch_size)

    best_acc = 0.0
    ckpt_path = PROJECT_ROOT / "checkpoints/para_classifier.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        clf.train()
        total_loss = 0.0
        for audio, labels in tqdm(train_loader, desc=f"epoch {epoch} train"):
            audio, labels = audio.to(device), labels.to(device)
            with torch.no_grad():
                hidden = backbone(audio).last_hidden_state.mean(dim=1)
            logits = clf(hidden)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        clf.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                hidden = backbone(audio).last_hidden_state.mean(dim=1)
                preds = clf(hidden).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(total, 1)
        print(f"epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "backbone": backbone_name,
                    "classifier": clf.state_dict(),
                    "val_acc": acc,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint -> {ckpt_path}")

    print(f"Done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
