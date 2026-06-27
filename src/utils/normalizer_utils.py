"""
normalizer_utils.py — Normalize raw answer strings before dish ID mapping.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from difflib import get_close_matches

from src.evaluation.generate_submission_file import dish_name_to_id, get_dish_mapping


def normalize_text(text: str) -> str:
    """Canonicalize whitespace, punctuation, and accents for comparisons."""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def normalize_answers(values: list[str]) -> list[str]:
    """Deduplicate and normalize a list of candidate dish names."""
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def map_dish_names_to_ids(
        names: list[str],
        allow_fuzzy: bool = True,
) -> list[int]:
    """Map dish names to IDs using the curated mapping file."""
    mapping = get_dish_mapping()
    mapping_keys = list(mapping.keys())
    ids: list[int] = []
    for name in names:
        dish_id = dish_name_to_id(name)
        if dish_id is not None:
            ids.append(dish_id)
            continue
        if not allow_fuzzy:
            continue
        matches = get_close_matches(name, mapping_keys, n=1, cutoff=0.9)
        if matches:
            ids.append(mapping[matches[0]])
    return sorted(set(ids))


def extract_dish_from_row(row: tuple, dish_mappings: dict[str, int], cutoff: float = 0.6) -> tuple[str, int] | None:
    """Extract the best matching dish name and ID from a row of cells.

    Args:
        row: A tuple of cells to search for dish names.
        dish_mappings: Dictionary mapping dish names to IDs.
        cutoff: Minimum similarity score threshold for matches.

    Returns:
        A tuple of (dish_name, dish_id) if a match is found, otherwise None.
    """
    best_name, best_score = None, 0.0

    for cell in row:
        cell_str = str(cell)
        matches = difflib.get_close_matches(cell_str, dish_mappings.keys(), n=1, cutoff=cutoff)
        if matches:
            score = difflib.SequenceMatcher(None, cell_str, matches[0]).ratio()
            if score > best_score:
                best_score, best_name = score, matches[0]

    if best_name is None:
        return None
    return best_name, dish_mappings[best_name]


def extract_dishes_from_rows(rows: list[tuple], dish_mappings: dict[str, int]) -> list[tuple[str, int] | None]:
    """Extract dish names and IDs from multiple rows of cells.

    Args:
        rows: List of tuples, where each tuple represents a row of cells.
        dish_mappings: Dictionary mapping dish names to IDs.

    Returns:
        List of tuples (dish_name, dish_id) or None for each row.
    """
    return [extract_dish_from_row(row, dish_mappings) for row in rows]
