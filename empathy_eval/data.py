"""Dataset loaders for test JSON releases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Question:
    question_id: str
    dataset: str
    context: str
    utterance: str
    utterance_audio: Path
    reference_response: str
    options: dict[str, Path]
    base_dir: Path
    group_id: str = ""
    task: str = ""  # context_variant | tone_variant; inferred from dataset if empty

    def resolve(self, rel: str) -> Path:
        rel_path = rel.lstrip("./")
        return self.base_dir / rel_path


TEST_DATASETS: dict[str, dict[str, str | int]] = {
    "gigaspeech": {
        "task": "context_variant",
        "json": "data/extracted/phase1-test_multi-context_gigaspeech/phase1-test_gigaspeech_release.json",
        "base_dir": "data/extracted/phase1-test_multi-context_gigaspeech",
    },
    "meld": {
        "task": "context_variant",
        "json": "data/extracted/phase1-test_multi-context_meld/phase1-test_meld_release.json",
        "base_dir": "data/extracted/phase1-test_multi-context_meld",
    },
    "emovdb": {
        "task": "tone_variant",
        "json": "data/extracted/phase1-test_multi-emotion_emovdb/phase1-test_emovdb_release.json",
        "base_dir": "data/extracted/phase1-test_multi-emotion_emovdb",
    },
}


def _group_id(dataset: str, question_id: str, utterance_audio: str) -> str:
    import re

    if dataset == "meld":
        return re.sub(r"_\d+$", "", question_id)
    if dataset == "emovdb":
        return re.sub(r"_e\d+$", "", question_id)
    return Path(utterance_audio).stem


def load_questions(
    project_root: Path,
    datasets: list[str] | None = None,
) -> Iterator[Question]:
    """Yield Question objects from configured test releases."""
    names = datasets or list(TEST_DATASETS)
    for name in names:
        if name not in TEST_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Choose from {list(TEST_DATASETS)}")
        meta = TEST_DATASETS[name]
        base_dir = project_root / str(meta["base_dir"])
        json_path = project_root / str(meta["json"])
        items = json.loads(json_path.read_text(encoding="utf-8"))
        for item in items:
            options = {
                key: base_dir / rel.lstrip("./")
                for key, rel in item["options"].items()
            }
            yield Question(
                question_id=item["question_id"],
                dataset=name,
                task=str(meta["task"]),
                context=item["context"],
                utterance=item["utterance"],
                utterance_audio=base_dir / item["utterance_audio"].lstrip("./"),
                reference_response=item["response"],
                options=options,
                base_dir=base_dir,
                group_id=_group_id(name, item["question_id"], item["utterance_audio"]),
            )


def load_single_question(project_root: Path, dataset: str, question_id: str) -> Question:
    for q in load_questions(project_root, [dataset]):
        if q.question_id == question_id:
            return q
    raise KeyError(f"Question {question_id} not found in {dataset}")
