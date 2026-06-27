"""
galactic_code_ingestion.py — Ingestion pipeline for the Codice Galattico.
Implements BaseIngestor to handle compliance rules and text indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import duckdb

from ingestion.ingestors.base_ingestor import BaseIngestor
from src.app.config import CHUNK_MAX_CHAR_CODE, COLLECTION_CODE, KB_DIR, OPENAI_API_KEY
from src.ingestion.ingestion_manager import IngestionManager
from src.ingestion.shared import build_payload, normalize_whitespace, read_text_fallback

log = logging.getLogger(__name__)

CODE_DIR: Path = KB_DIR / "codice_galattico"

# TODO: extract to config
# TODO: Replace manual JSON schema declaration with a Pydantic model.
SYSTEM_PROMPT = """You are a structured entity extractor for a regulatory compliance document (Codice Galattico).

Extract ALL compliance rules regarding ingredient limits or technique license requirements. Return a JSON object with this EXACT schema:

{
  "compliance_rules": [
    {
      "rule_type": "ingredient_limit" or "technique_license",
      "subject": "<name of ingredient or technique>",
      "constraint_value": "<max quantity (e.g. '500g') or required license level (e.g. 'Level B')>",
      "scope": "<context of the rule, e.g. 'per dish' or null>"
    }
  ],
  "parsing_confidence": "high" or "low",
  "parsing_issues": "<description or null>"
}

CRITICAL RULES:
- rule_type: must be exactly 'ingredient_limit' or 'technique_license'.
- constraint_value: string representing the limit or the required license.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
"""


class GalacticCodeIngestor(BaseIngestor):
    @property
    def source_type(self) -> str:
        return "codice"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def collection_name(self) -> str:
        return COLLECTION_CODE

    @property
    def chunk_max_char(self) -> int:
        return CHUNK_MAX_CHAR_CODE

    @property
    def directory(self) -> Path:
        return CODE_DIR

    @property
    def file_glob(self) -> str:
        # Original code used * to glob all files in the dir
        return "*"

    @property
    def use_vision_fallback(self) -> bool:
        # Regulatory text is assumed to be machine-readable
        return False

    def _parse_code_text(self, source_path: str) -> str:
        """Parse a Codice Galattico document to plain text."""
        try:
            from datapizza.modules.parsers.docling import DoclingParser

            parser = DoclingParser()
            nodes = parser([source_path])
            text = "\n\n".join(getattr(n, "text", "") for n in nodes if getattr(n, "text", ""))
            if text.strip():
                return normalize_whitespace(text)
        except Exception as exc:
            log.warning("DoclingParser failed for %s: %s", source_path, exc)

        return normalize_whitespace(read_text_fallback(Path(source_path)))

    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> None:
        """Write extracted compliance rules to the DuckDB database."""
        rules = extraction_result.get("compliance_rules", [])
        for rule in rules:
            rule_type = rule.get("rule_type")
            subject = rule.get("subject")
            constraint_value = rule.get("constraint_value")
            scope = rule.get("scope")

            if not rule_type or not subject or not constraint_value:
                log.warning("Skipping incomplete compliance rule: %s", rule)
                continue

            rule_id = facts_con.execute("SELECT nextval('compliance_rules_id_seq')").fetchone()[0]
            facts_con.execute(
                """
                INSERT INTO compliance_rules
                    (id, rule_type, subject, constraint_value, scope, doc_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [rule_id, rule_type, subject, constraint_value, scope, doc_id],
            )
        log.info("Wrote %d compliance rules for doc %s", len(rules), doc_id[:8])

    def make_vector_indexer(self, ingestion_manager: IngestionManager, source_path: Path) -> Callable[[str, str, dict], None]:
        """Return a vector_indexer that chunks raw_text and upserts into Qdrant."""
        # Pre-parse the text to bypass DoclingParser inside datapizza pipeline
        raw_text = self._parse_code_text(str(source_path))

        def _index(doc_id: str, source_path: str, extraction_result: dict) -> None:
            vs = ingestion_manager.km.get_qdrant_vectorstore()

            from datapizza.embedders import ChunkEmbedder
            from datapizza.embedders.openai import OpenAIEmbedder
            from datapizza.modules.splitters import NodeSplitter
            from datapizza.pipeline import IngestionPipeline as DatapizzaPipeline
            from src.app.config import EMBEDDING_MODEL

            embedder_client = OpenAIEmbedder(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)
            datapizza_pipeline = DatapizzaPipeline(
                modules=[
                    NodeSplitter(max_char=self.chunk_max_char),
                    ChunkEmbedder(client=embedder_client),
                ],
                vector_store=vs,
                collection_name=self.collection_name,
            )

            # Codice files are plain text — pass raw_text directly
            datapizza_pipeline.run(
                text=raw_text,
                metadata=build_payload(
                    doc_id=doc_id,
                    source_path=source_path,
                    source_type=self.source_type,
                    text="",
                ),
            )
        return _index