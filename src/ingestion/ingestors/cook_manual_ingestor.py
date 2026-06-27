"""
cook_manual_ingestor.py — Ingestion pipeline for the Manuale di Cucina.
Implements BaseIngestor to handle technique taxonomy extraction and text indexing.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import duckdb

from ingestion.ingestors.base_ingestor import BaseIngestor
from src.app.config import CHUNK_MAX_CHAR_MANUAL, COLLECTION_MANUAL, KB_DIR, OPENAI_API_KEY
from src.ingestion.ingestion_manager import IngestionManager
from src.ingestion.shared import build_payload, normalize_whitespace

log = logging.getLogger(__name__)

# TODO: configurable
MANUAL_DIR: Path = KB_DIR / "misc"

# TODO: extract to config
# TODO: Replace manual JSON schema declaration with a Pydantic model.
SYSTEM_PROMPT = """You are a structured entity extractor for a fictionary culinary manual (Manuale di Cucina).

Extract ALL culinary techniques, their macro categories, and required license levels. Return a JSON object with this EXACT schema:

{
  "techniques": [
    {
      "name": "<technique name>",
      "macro_category": "<macro category name>",
      "required_license_level": "<license level (e.g. 'Level A', 'Level B') or null>"
    }
  ],
  "parsing_confidence": "high" or "low",
  "parsing_issues": "<description or null>"
}

CRITICAL RULES:
- name: must be the exact name of the culinary technique.
- macro_category: the broad category the technique belongs to.
- required_license_level: the professional license required to perform the technique, or null if not specified.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
"""


def _build_datapizza_pipeline(vs, collection_name: str, chunk_max_char: int):
    """Build a datapizza-ai IngestionPipeline."""
    from datapizza.embedders import ChunkEmbedder
    from datapizza.embedders.openai import OpenAIEmbedder
    from datapizza.modules.parsers.docling import DoclingParser
    from datapizza.modules.splitters import NodeSplitter
    from datapizza.pipeline import IngestionPipeline as DatapizzaPipeline
    from src.app.config import EMBEDDING_MODEL

    embedder_client = OpenAIEmbedder(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)
    return DatapizzaPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=chunk_max_char),
            ChunkEmbedder(client=embedder_client),
        ],
        vector_store=vs,
        collection_name=collection_name,
    )


class CookManualIngestor(BaseIngestor):
    @property
    def source_type(self) -> str:
        return "manual"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def collection_name(self) -> str:
        return COLLECTION_MANUAL

    @property
    def chunk_max_char(self) -> int:
        return CHUNK_MAX_CHAR_MANUAL

    @property
    def directory(self) -> Path:
        return MANUAL_DIR

    @property
    def file_glob(self) -> str:
        # set punctual file name
        return "Manuale di Cucina.pdf"

    @property
    def use_vision_fallback(self) -> bool:
        return True

    def parse_document(self, source_path: Path) -> str:
        """Adds manual-specific text normalization."""
        raw_text = super().parse_document(source_path)
        return normalize_whitespace(raw_text)

    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str,
                         facts_con: duckdb.DuckDBPyConnection) -> None:
        """Write extracted techniques to the technique_taxonomy table."""
        techniques = extraction_result.get("techniques", [])
        for tech in techniques:
            name = tech.get("name")
            macro_category = tech.get("macro_category")
            required_license = tech.get("required_license_level")

            if not name or not macro_category:
                log.warning("Skipping incomplete technique: %s", tech)
                continue

            facts_con.execute(
                """
                INSERT INTO technique_taxonomy (technique, macro_category, required_license_level)
                VALUES (?, ?, ?)
                ON CONFLICT(technique) DO UPDATE SET
                    macro_category = excluded.macro_category,
                    required_license_level = excluded.required_license_level
                """,
                [name, macro_category, required_license],
            )
        log.info("Wrote %d techniques for doc %s", len(techniques), doc_id[:8])

    def make_vector_indexer(self, ingestion_manager: IngestionManager, source_path: Path, raw_text: str) -> Callable[
        [str, str, dict], None]:
        """Return a vector_indexer that chunks raw_text and upserts into Qdrant."""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(raw_text)
            tmp_path = tmp.name

        def _index(doc_id: str, source_path: str, extraction_result: dict) -> None:
            try:
                vs = ingestion_manager.km.get_qdrant_vectorstore()
                datapizza_pipeline = _build_datapizza_pipeline(vs, self.collection_name, self.chunk_max_char)

                datapizza_pipeline.run(
                    file_path=tmp_path,
                    metadata=build_payload(
                        doc_id=doc_id,
                        source_path=source_path,
                        source_type=self.source_type,
                        text="",
                    ),
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return _index