"""
distances_ingestion.py — Loads the planet distances CSV directly into DuckDB.

NOT indexed in Qdrant (distances are deterministic, SQL is the right tool).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.app.config import KB_DIR

log = logging.getLogger(__name__)

# TODO: configurable
DISTANCES_CSV: Path = KB_DIR / "misc" / "Distanze.csv"


def ingest_distances(
    facts_con: duckdb.DuckDBPyConnection,
    csv_path: Path = DISTANCES_CSV,
) -> int:
    """
    Load the planet distances CSV into the planet_distances table.

    The CSV is expected to have planets as both row index and column headers
    (a symmetric distance matrix).

    Returns:
        Number of rows inserted.
    """
    if not csv_path.exists():
        log.error(f"Distances CSV not found: {csv_path}")
        return 0

    # Load and pivot the matrix
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()

    rows_inserted = 0
    for planet_a in df.index:
        for planet_b in df.columns:
            if planet_a == planet_b:
                continue
            try:
                distance = float(df.loc[planet_a, planet_b])
            except (ValueError, KeyError):
                continue

            try:
                facts_con.execute(
                    "INSERT OR IGNORE INTO planet_distances (planet_a, planet_b, distance_ly) "
                    "VALUES (?, ?, ?)",
                    [planet_a, planet_b, distance],
                )
                rows_inserted += 1
            except Exception as exc:
                log.warning(f"Failed to insert distance {planet_a}↔{planet_b}: {exc}")

    log.info(f"Inserted {rows_inserted} planet distance rows from {csv_path.name}")
    return rows_inserted
