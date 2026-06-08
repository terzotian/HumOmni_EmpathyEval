"""Competition compliance guards.

Official rule: test data must NOT be used in any API-related process.
APIs may only be used for model training on training data.
"""

from __future__ import annotations

# Datasets that belong to the public TEST split (local inference only).
TEST_DATASET_NAMES: frozenset[str] = frozenset({"gigaspeech", "meld", "emovdb"})

# Path fragments that indicate test assets.
TEST_PATH_MARKERS: tuple[str, ...] = (
    "phase1-test",
    "phase1-test_",
    "/gigaspeech_",
    "/meld_",
    "/emovdb_",
)


class ComplianceError(RuntimeError):
    """Raised when an operation would violate competition rules."""


def is_test_dataset(name: str) -> bool:
    return name in TEST_DATASET_NAMES


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(marker in normalized for marker in TEST_PATH_MARKERS)


def assert_local_inference_only(*, dataset: str | None = None, paths: list[str] | None = None) -> None:
    """Ensure an inference path only touches test data locally (no API)."""
    if dataset and is_test_dataset(dataset):
        return
    if paths:
        for p in paths:
            if is_test_path(p):
                return
    raise ComplianceError(
        "Could not verify test-data context for local inference. "
        f"dataset={dataset!r}, paths={paths!r}"
    )


def assert_api_training_only(
    *,
    dataset: str | None = None,
    paths: list[str] | None = None,
    operation: str = "API call",
) -> None:
    """Block API usage when test data is involved."""
    if dataset and is_test_dataset(dataset):
        raise ComplianceError(
            f"{operation} blocked: dataset {dataset!r} is TEST data. "
            "Competition rules forbid API on test data."
        )

    if paths:
        for p in paths:
            if is_test_path(p):
                raise ComplianceError(
                    f"{operation} blocked: path {p!r} looks like TEST data. "
                    "Competition rules forbid API on test data."
                )

    # Training paths should live under empatheticDialogue_* or data/raw jsonl.
    if paths:
        for p in paths:
            if "empatheticDialogue" not in p and "data/raw" not in p:
                raise ComplianceError(
                    f"{operation} blocked: path {p!r} is not a recognized TRAINING path."
                )


def assert_not_demo_with_api(dataset: str, operation: str = "API call") -> None:
    if dataset == "demo":
        raise ComplianceError(f"{operation} blocked on demo dataset.")
