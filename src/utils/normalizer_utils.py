"""
normalizer_utils.py — Normalize raw answer strings before dish ID mapping.
"""

from __future__ import annotations

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

