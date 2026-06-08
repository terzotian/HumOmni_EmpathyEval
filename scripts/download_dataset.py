#!/usr/bin/env python3
"""Download a Hugging Face dataset repo into the project data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face Hub")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset config YAML",
    )
    parser.add_argument("--repo-id", type=str, help="Override repo_id from config")
    parser.add_argument("--local-dir", type=Path, help="Override local_dir from config")
    parser.add_argument(
        "--login",
        action="store_true",
        help="Run interactive Hugging Face login before download",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / args.config)

    repo_id = args.repo_id or config["repo_id"]
    if repo_id.startswith("YOUR_"):
        raise SystemExit(
            "Please set repo_id in configs/dataset.yaml "
            "(format: username/dataset-name)."
        )

    local_dir = args.local_dir or project_root / config["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.login or config.get("private", False):
        login()

    include_patterns = config.get("include_patterns") or None

    print(f"Downloading dataset: {repo_id}")
    print(f"Saving to: {local_dir}")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=include_patterns,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            "Download failed. If the dataset is private/gated, run:\n"
            "  hf auth login\n"
            "or re-run with --login\n\n"
            f"Original error: {exc}"
        ) from exc

    print(f"Done. Files saved under: {path}")


if __name__ == "__main__":
    main()
