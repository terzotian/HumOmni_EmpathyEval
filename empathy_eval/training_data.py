"""Build supervised training examples from empatheticDialogue training audio."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingChoiceExample:
    example_id: str
    task: str  # context_variant | tone_variant
    context: str
    utterance_text: str
    utterance_audio: Path
    reference_response: str
    options: dict[str, Path]  # opt-A, opt-B
    answer: str  # A or B
    good_option: str
    bad_option: str

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "task": self.task,
            "context": self.context,
            "utterance_text": self.utterance_text,
            "utterance_audio": str(self.utterance_audio),
            "reference_response": self.reference_response,
            "options": {k: str(v) for k, v in self.options.items()},
            "answer": self.answer,
            "good_option": self.good_option,
            "bad_option": self.bad_option,
        }


_PARA_CTX_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<ctx_idx>[12])_(?P<label>goodPara|badPara)\.wav$"
)
_PARA_EMO_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<emotion>[a-z]+)_(?P<label>goodPara|badPara)\.wav$"
)


def _load_jsonl_map(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[row["id"]] = row
    return mapping


def _context_variant_examples(
    project_root: Path,
    *,
    shuffle_bad_good: bool = True,
    seed: int = 42,
) -> list[TrainingChoiceExample]:
    base = project_root / "data/extracted/empatheticDialogue_t_multi-context"
    audio_dir = base / "response_audio"
    user_dir = base / "user_audio"
    text_map = _load_jsonl_map(
        project_root / "data/raw/empatheticDialogue_t_multi-context_flat.jsonl"
    )

    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for wav in audio_dir.glob("*.wav"):
        m = _PARA_CTX_RE.match(wav.name)
        if not m:
            continue
        key = (m.group("prefix"), m.group("ctx_idx"))
        grouped.setdefault(key, {})[m.group("label")] = wav

    rng = random.Random(seed)
    examples: list[TrainingChoiceExample] = []

    for (prefix, ctx_idx), labels in sorted(grouped.items()):
        if "goodPara" not in labels or "badPara" not in labels:
            continue

        row = text_map.get(prefix)
        if not row:
            continue

        ctx_i = int(ctx_idx) - 1
        contexts = row.get("contexts") or []
        if ctx_i >= len(contexts):
            continue

        ctx_obj = contexts[ctx_i]
        emotion = row.get("emotion", "neutral")
        utterance_audio = user_dir / f"{prefix}_{ctx_idx}_{emotion}.wav"
        if not utterance_audio.exists():
            # fallback: any user audio with same prefix+idx
            candidates = list(user_dir.glob(f"{prefix}_{ctx_idx}_*.wav"))
            if not candidates:
                continue
            utterance_audio = candidates[0]

        good_path = labels["goodPara"]
        bad_path = labels["badPara"]

        if shuffle_bad_good and rng.random() < 0.5:
            options = {"opt-A": bad_path, "opt-B": good_path}
            answer = "B"
            good_option, bad_option = "B", "A"
        else:
            options = {"opt-A": good_path, "opt-B": bad_path}
            answer = "A"
            good_option, bad_option = "A", "B"

        examples.append(
            TrainingChoiceExample(
                example_id=f"{prefix}_{ctx_idx}",
                task="context_variant",
                context=ctx_obj["context"],
                utterance_text=row["text"],
                utterance_audio=utterance_audio,
                reference_response=ctx_obj["response"],
                options=options,
                answer=answer,
                good_option=good_option,
                bad_option=bad_option,
            )
        )

    return examples


def _tone_variant_examples(
    project_root: Path,
    *,
    shuffle_bad_good: bool = True,
    seed: int = 42,
) -> list[TrainingChoiceExample]:
    base = project_root / "data/extracted/empatheticDialogue_n_multi-emotion"
    audio_dir = base / "response_audio"
    user_dir = base / "user_audio"
    text_map = _load_jsonl_map(
        project_root / "data/raw/empatheticDialogue_n_multi-emotion_flat.jsonl"
    )

    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for wav in audio_dir.glob("*.wav"):
        m = _PARA_EMO_RE.match(wav.name)
        if not m:
            continue
        key = (m.group("prefix"), m.group("emotion"))
        grouped.setdefault(key, {})[m.group("label")] = wav

    rng = random.Random(seed)
    examples: list[TrainingChoiceExample] = []

    for (prefix, emotion), labels in sorted(grouped.items()):
        if "goodPara" not in labels or "badPara" not in labels:
            continue

        row = text_map.get(prefix)
        if not row:
            continue

        ctx_blob = row.get("contexts") or {}
        shared_context = ctx_blob.get("Context", "")
        if not shared_context:
            continue

        utterance_audio = user_dir / f"{prefix}_{emotion}.wav"
        if not utterance_audio.exists():
            continue

        response_key = f"{emotion}_response"
        reference = ctx_blob.get(response_key, "")
        if not reference:
            continue

        good_path = labels["goodPara"]
        bad_path = labels["badPara"]

        if shuffle_bad_good and rng.random() < 0.5:
            options = {"opt-A": bad_path, "opt-B": good_path}
            answer = "B"
            good_option, bad_option = "B", "A"
        else:
            options = {"opt-A": good_path, "opt-B": bad_path}
            answer = "A"
            good_option, bad_option = "A", "B"

        examples.append(
            TrainingChoiceExample(
                example_id=f"{prefix}_{emotion}",
                task="tone_variant",
                context=shared_context,
                utterance_text=row["text"],
                utterance_audio=utterance_audio,
                reference_response=reference,
                options=options,
                answer=answer,
                good_option=good_option,
                bad_option=bad_option,
            )
        )

    return examples


def load_training_examples(
    project_root: Path,
    tasks: list[str] | None = None,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[TrainingChoiceExample]:
    selected = tasks or ["context_variant", "tone_variant"]
    examples: list[TrainingChoiceExample] = []
    if "context_variant" in selected:
        examples.extend(_context_variant_examples(project_root, seed=seed))
    if "tone_variant" in selected:
        examples.extend(_tone_variant_examples(project_root, seed=seed))

    if limit is not None and len(examples) > limit:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:limit]
    return examples


def save_training_examples(examples: list[TrainingChoiceExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
