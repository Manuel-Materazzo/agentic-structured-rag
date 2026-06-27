"""
validation_utils.py — Candidate validation against the curated dish mapping.
"""

from __future__ import annotations

from src.app.submission import get_dish_mapping, dish_name_to_id


def validate_candidate_dishes(names: list[str]) -> list[str]:
    """
    Keep only candidates present in the dish mapping.

    Unknown candidates are removed TODO: normalize and fix them.
    """
    mapping = get_dish_mapping()
    valid: list[str] = []
    for name in names:
        if dish_name_to_id(name) is not None or name in mapping:
            valid.append(name)
    return valid

