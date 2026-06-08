"""Convert internal predictions to official Track 1 submission format."""

from __future__ import annotations

import json
import re
from pathlib import Path


def choice_to_answer(choice: str) -> str:
    """
    Official format uses A/B/C, not opt-A/opt-B/opt-C.
    """
    choice = choice.strip()
    if re.fullmatch(r"[ABC]", choice):
        return choice
    m = re.search(r"opt-([ABC])", choice, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"([ABC])$", choice, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    raise ValueError(f"Cannot parse option label: {choice!r}")


def load_predictions_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_submission(
    prediction_files: list[Path],
    output_path: Path,
    *,
    sort_by_id: bool = True,
) -> dict[str, int]:
    """
    Merge per-dataset prediction files into one official submission jsonl.

    Returns counts per dataset for logging.
    """
    merged: dict[str, dict] = {}
    counts: dict[str, int] = {}

    for pred_file in prediction_files:
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing predictions: {pred_file}")

        for row in load_predictions_jsonl(pred_file):
            qid = row["question_id"]
            if qid in merged:
                raise ValueError(f"Duplicate question_id across files: {qid}")

            answer = choice_to_answer(row["choice"])
            merged[qid] = {"question_id": qid, "answer": answer}
            dataset = row.get("dataset", "unknown")
            counts[dataset] = counts.get(dataset, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(merged.values())
    if sort_by_id:
        items.sort(key=lambda x: x["question_id"])

    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return counts
