#!/usr/bin/env python3
"""Resume EmpathyEval download with retries (run inside humomni env)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "gracehuggingface/EmpathyEval"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = [
    "phase1-test_multi-emotion_emovdb.zip",
    "empatheticDialogue_n_multi-emotion.zip",
    "empatheticDialogue_t_multi-context.zip",
]
MAX_RETRIES = 20
RETRY_WAIT_SEC = 60


def file_ready(name: str) -> bool:
    path = LOCAL_DIR / name
    return path.is_file() and path.stat().st_size > 0


def download_file(name: str) -> None:
    cmd = [
        sys.executable.replace("python", "hf") if False else "hf",
        "download",
        REPO_ID,
        name,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(LOCAL_DIR),
    ]
    # Use python -m huggingface_hub if hf is not on PATH
    import shutil

    hf_bin = shutil.which("hf")
    if hf_bin:
        cmd[0] = hf_bin
    else:
        raise RuntimeError("hf CLI not found in PATH")

    print(f"Downloading {name} ...", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    pending = [f for f in FILES if not file_ready(f)]
    if not pending:
        print("All target files already present.")
        return

    print(f"Pending: {pending}")
    for name in pending:
        for attempt in range(1, MAX_RETRIES + 1):
            if file_ready(name):
                print(f"OK: {name}")
                break
            try:
                download_file(name)
            except subprocess.CalledProcessError as exc:
                print(f"Attempt {attempt}/{MAX_RETRIES} failed for {name}: {exc}")
                if attempt == MAX_RETRIES:
                    raise SystemExit(f"Giving up on {name}") from exc
                time.sleep(RETRY_WAIT_SEC)
        else:
            continue

    remaining = [f for f in FILES if not file_ready(f)]
    if remaining:
        raise SystemExit(f"Still missing: {remaining}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
