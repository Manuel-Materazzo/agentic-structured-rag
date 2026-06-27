"""
submission.py — Dish mapping loader and CSV submission exporter.

Responsibilities:
- Load Dataset/ground_truth/dish_mapping.json at startup
- Map dish name strings → integer IDs
- Export output/submission.csv with row_id and result columns
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.app.config import (
    DISH_MAPPING_PATH,
    DOMANDE_CSV_PATH,
    SUBMISSION_CSV_PATH,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dish mapping — loaded once at module level
# ---------------------------------------------------------------------------

_DISH_MAPPING: dict[str, int] = {}
_DISH_MAPPING_LOWER: dict[str, int] = {}  # case-insensitive lookup


def load_dish_mapping(path: Path = DISH_MAPPING_PATH) -> dict[str, int]:
    """
    Load dish_mapping.json into memory.
    Returns {dish_name: dish_id} dict.
    """
    global _DISH_MAPPING, _DISH_MAPPING_LOWER
    with open(path, encoding="utf-8") as f:
        _DISH_MAPPING = json.load(f)
    _DISH_MAPPING_LOWER = {k.lower().strip(): v for k, v in _DISH_MAPPING.items()}
    log.info(f"Loaded {len(_DISH_MAPPING)} dishes from {path}")
    return _DISH_MAPPING


def get_dish_mapping() -> dict[str, int]:
    """Return the cached dish mapping, loading it if necessary."""
    if not _DISH_MAPPING:
        load_dish_mapping()
    return _DISH_MAPPING


def dish_name_to_id(name: str) -> int | None:
    """
    Map a dish name string to its integer ID.
    Tries exact match first, then case-insensitive.
    Returns None if no match found.
    """
    if not _DISH_MAPPING:
        load_dish_mapping()
    # Exact match
    if name in _DISH_MAPPING:
        return _DISH_MAPPING[name]
    # Case-insensitive match
    normalized = name.lower().strip()
    return _DISH_MAPPING_LOWER.get(normalized)


def dish_ids_to_result_string(ids: list[int]) -> str:
    """Convert a sorted list of dish IDs to a comma-separated string."""
    sorted_ids = sorted(set(ids))
    return ",".join(str(i) for i in sorted_ids)


# ---------------------------------------------------------------------------
# Questions loader
# ---------------------------------------------------------------------------

def load_questions(path: Path = DOMANDE_CSV_PATH) -> pd.DataFrame:
    """Load domande.csv and return a DataFrame with row_id and domanda columns."""
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} questions from {path}")
    return df


# ---------------------------------------------------------------------------
# Submission export
# ---------------------------------------------------------------------------

def export_empty_submission(
    output_path: Path = SUBMISSION_CSV_PATH,
    questions_path: Path = DOMANDE_CSV_PATH,
) -> Path:
    """
    Generate an empty submission CSV with all row_ids present but result=''.
    Useful for verifying the format before the full pipeline is ready.
    """
    df = load_questions(questions_path)
    # Find the row_id column (handles both 'row_id' and index-based)
    if "row_id" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={"index": "row_id"})
        df["row_id"] = df["row_id"] + 1

    submission = pd.DataFrame({
        "row_id": df["row_id"].astype(int),
        "result": [""] * len(df),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    log.info(f"Exported empty submission ({len(submission)} rows) → {output_path}")
    return output_path


def export_submission(
    answers: dict[int, list[int]],
    output_path: Path = SUBMISSION_CSV_PATH,
    questions_path: Path = DOMANDE_CSV_PATH,
) -> Path:
    """
    Export the final submission CSV.

    Args:
        answers: {row_id: [dish_id, ...]} mapping
        output_path: destination CSV path
        questions_path: domande.csv path (to ensure all row_ids are present)

    Returns:
        Path to the written CSV.
    """
    df = load_questions(questions_path)
    if "row_id" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={"index": "row_id"})
        df["row_id"] = df["row_id"] + 1

    rows = []
    for _, row in df.iterrows():
        rid = int(row["row_id"])
        ids = answers.get(rid, [])
        result = dish_ids_to_result_string(ids) if ids else ""
        rows.append({"row_id": rid, "result": result})

    submission = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    log.info(f"Exported submission ({len(submission)} rows) → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dish_mapping()
    out = export_empty_submission()
    print(f"✅ Empty submission written to {out}")
    # Verify format
    df = pd.read_csv(out)
    assert list(df.columns) == ["row_id", "result"], "Unexpected columns"
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
