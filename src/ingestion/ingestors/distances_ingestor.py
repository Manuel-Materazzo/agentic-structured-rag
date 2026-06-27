"""
distances_ingestor.py — Ingestion pipeline for the planet distances CSV.
Implements BaseIngestor to handle DuckDB population without LLM or Qdrant indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import duckdb
import pandas as pd

from ingestion.ingestors.base_ingestor import BaseIngestor
from src.app.config import KB_DIR
from src.ingestion.ingestion_manager import IngestionManager

log = logging.getLogger(__name__)

# TODO: configurable
DISTANCES_DIR: Path = KB_DIR / "misc"


class DistancesIngestor(BaseIngestor):
    """
    Ingestor for deterministic CSV data.
    Bypasses LLM extraction and Qdrant indexing by overriding the pipeline phases.
    """

    @property
    def source_type(self) -> str:
        return "distanze"

    @property
    def system_prompt(self) -> str:
        # Unused
        return ""

    @property
    def collection_name(self) -> str:
        # Unused
        return ""

    @property
    def chunk_max_char(self) -> int:
        return 0

    @property
    def directory(self) -> Path:
        return DISTANCES_DIR

    @property
    def file_glob(self) -> str:
        return "Distanze.csv"

    @property
    def use_vision_fallback(self) -> bool:
        return False

    def parse_document(self, source_path: Path) -> str:
        """
        Override of parsing: we don't use DoclingParser for CSV files.
        We simply return the file path as a string,
        which will be passed to subsequent phases.
        """
        return str(source_path)

    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str,
                         facts_con: duckdb.DuckDBPyConnection) -> None:
        """
        Populates the planet_distances table by reading the CSV using Pandas.
        """
        csv_path = Path(extraction_result.get("csv_path", ""))
        if not csv_path.exists():
            log.error("CSV file not found during write phase: %s", csv_path)
            return

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
                        """
                        INSERT INTO planet_distances (planet_a, planet_b, distance_ly) 
                        VALUES (?, ?, ?)
                        ON CONFLICT(planet_a, planet_b) DO UPDATE SET distance_ly = excluded.distance_ly
                        """,
                        [planet_a, planet_b, distance],
                    )
                    rows_inserted += 1
                except Exception as exc:
                    log.warning("Failed to insert distance %s↔%s: %s", planet_a, planet_b, exc)

        log.info("Wrote %d planet distance rows for doc %s", rows_inserted, doc_id[:8])

    def make_vector_indexer(self, ingestion_manager: IngestionManager, source_path: Path, raw_text: str) -> Callable[
        [str, str, dict], None]:
        """
                Distances are not indexed in Qdrant. Returning a no-op callback.
        """
        def _noop_indexer(doc_id: str, source_path: str, extraction_result: dict) -> None:
            log.info("Skipping Qdrant indexing for distances (deterministic data).")
            pass

        return _noop_indexer

    def ingest_all(self, ingestion_manager: IngestionManager) -> list[str]:
        """
        Override of the BaseIngestor's 3-phase pipeline.
        For CSV files, we skip the disk cache as .txt files and use a simplified workflow.
        """
        import hashlib

        def _sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        files = sorted(self.directory.glob(self.file_glob))
        if not files:
            log.warning("No CSV files found in %s", self.directory)
            return []

        log.info("Found %d CSV files to ingest for %s", len(files), self.source_type)
        successful_doc_ids = []

        for file_path in files:
            doc_id = _sha256_file(file_path)
            source_path = str(file_path)

            existing = ingestion_manager.km.log_get(source_path)
            if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
                log.info("Skip (already complete): %s", file_path.name)
                continue

            ingestion_manager.km.log_upsert(doc_id, source_path, "extracting")
            try:
                # Create a fake extraction_result that carries the file path
                extraction_result = {"csv_path": source_path}

                ingestion_manager.km.write_document(
                    extraction_result,
                    doc_id,
                    source_path,
                    self.source_type,
                    self.write_db_entries
                )
                ingestion_manager.km.log_set_status(doc_id, "extracted")

                # Skip phase 3 (embedding) and go directly to complete
                successful_doc_ids.append(doc_id)
                log.info("Ingestion complete: %s (%s)", source_path, doc_id)
            except Exception as exc:
                log.error("Failed to extract/write %s: %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        return successful_doc_ids