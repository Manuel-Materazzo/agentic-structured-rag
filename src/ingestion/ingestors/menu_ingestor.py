"""
menu_ingestor.py — Ingestion pipeline for restaurant menu PDFs.
Implements BaseIngestor to handle menu-specific parsing, extraction, and indexing.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import duckdb

from ingestion.ingestors.base_ingestor import BaseIngestor
from model.menu import RestaurantData
from src.app.config import CHUNK_MAX_CHAR_MENU, COLLECTION_MENU, KB_DIR, OPENAI_EMBEDDER_API_KEY, \
    OPENAI_EMBEDDER_BASE_URL
from src.ingestion.ingestion_manager import IngestionManager
from src.ingestion.shared import build_payload

log = logging.getLogger(__name__)

MENU_DIR: Path = KB_DIR / "menu"

# TODO: extract to config
SYSTEM_PROMPT = """You are a structured entity extractor for a fictionary restaurant menu database.

Extract ALL dishes from the provided menu text and return a JSON object with this EXACT schema:
{output_format}

CRITICAL RULES:
- quantity_grams: FLOAT with decimal point (e.g. 200.0) or null. NEVER use 0.0 for "as much as needed", "to taste", "traces", or unquantifiable amounts.
- quantity_raw: ALWAYS populate with the original text (e.g. "200g", "as much as needed", "3 leaves").
- parsing_confidence: Use "low" ONLY if you could NOT extract coherent entity structure. NOT just because the text was messy or sounds fictional.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
"""


def _build_datapizza_pipeline(vs, collection_name: str, chunk_max_char: int):
    """Build a datapizza-ai IngestionPipeline. Usiamo DoclingParser perché leggerà un .txt istantaneamente."""
    from datapizza.embedders import ChunkEmbedder
    from datapizza.embedders.openai import OpenAIEmbedder
    from datapizza.modules.parsers.docling import DoclingParser
    from datapizza.modules.splitters import NodeSplitter
    from datapizza.pipeline import IngestionPipeline as DatapizzaPipeline
    from src.app.config import EMBEDDING_MODEL

    embedder_client = OpenAIEmbedder(
        api_key=OPENAI_EMBEDDER_API_KEY,
        base_url=OPENAI_EMBEDDER_BASE_URL,
        model_name=EMBEDDING_MODEL
    )
    return DatapizzaPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=chunk_max_char),
            ChunkEmbedder(client=embedder_client),
        ],
        vector_store=vs,
        collection_name=collection_name,
    )


class MenuIngestor(BaseIngestor):
    @property
    def source_type(self) -> str:
        return "menu"

    @property
    def system_prompt(self) -> str:
        output_format = json.dumps(RestaurantData.model_json_schema(), indent=2)
        return SYSTEM_PROMPT.format(output_format=output_format)

    @property
    def collection_name(self) -> str:
        return COLLECTION_MENU

    @property
    def chunk_max_char(self) -> int:
        return CHUNK_MAX_CHAR_MENU

    @property
    def directory(self) -> Path:
        return MENU_DIR

    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str,
                         facts_con: duckdb.DuckDBPyConnection) -> None:
        """Write extracted menu data to DuckDB tables."""
        rest_data = extraction_result.get("restaurant", {})
        rest_name = rest_data.get("name") or "Unknown"

        existing_rest = facts_con.execute(
            "SELECT id FROM restaurants WHERE doc_id = ? AND name = ?",
            [doc_id, rest_name],
        ).fetchone()

        if existing_rest:
            rest_id = existing_rest[0]
        else:
            rest_id = facts_con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM restaurants").fetchone()[0]
            facts_con.execute(
                """
                INSERT INTO restaurants
                    (id, name, chef, planet, chef_license, professional_orders, doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rest_id, rest_name, rest_data.get("chef"), rest_data.get("planet"),
                    rest_data.get("chef_license"), rest_data.get("professional_orders", []), doc_id,
                ],
            )

        for dish in extraction_result.get("dishes", []):
            dish_name = (dish.get("name") or "").strip()
            if not dish_name:
                continue

            existing_dish = facts_con.execute(
                "SELECT id FROM dishes WHERE name = ? AND restaurant_id = ?",
                [dish_name, rest_id],
            ).fetchone()

            if existing_dish:
                dish_id = existing_dish[0]
            else:
                dish_id = facts_con.execute("SELECT nextval('dishes_id_seq')").fetchone()[0]
                facts_con.execute(
                    """
                    INSERT INTO dishes (id, name, restaurant_id, preparation_notes, doc_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [dish_id, dish_name, rest_id, dish.get("preparation_notes"), doc_id],
                )

            for ing in dish.get("ingredients", []):
                ing_name = (ing.get("name") or "").strip()
                if not ing_name: continue

                # Force quantity_raw to empty string if None or missing
                qty_raw = ing.get("quantity_raw")
                if not qty_raw:
                    qty_raw = ""

                try:
                    facts_con.execute(
                        """
                        INSERT INTO dish_ingredients
                            (dish_id, ingredient, quantity_grams, quantity_raw, is_regulated)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (dish_id, ingredient) DO NOTHING
                        """,
                        [dish_id, ing_name, ing.get("quantity_grams"), qty_raw, False],
                    )
                except Exception as exc:
                    log.warning("Failed to insert ingredient '%s': %s", ing_name, exc)

            for tech in dish.get("techniques", []):
                tech_name = tech.strip() if isinstance(tech, str) else str(tech)
                if not tech_name: continue
                try:
                    facts_con.execute(
                        """
                        INSERT INTO dish_techniques (dish_id, technique)
                        VALUES (?, ?)
                        ON CONFLICT (dish_id, technique) DO NOTHING
                        """,
                        [dish_id, tech_name],
                    )
                except Exception as exc:
                    log.warning("Failed to insert technique '%s': %s", tech_name, exc)

    def make_vector_indexer(self, ingestion_manager: IngestionManager, source_path: Path, raw_text: str) -> Callable[
        [str, str, dict], None]:
        """Return a vector_indexer bound to the Qdrant connection."""

        # Salva il testo già parsato in un file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(raw_text)
            tmp_path = tmp.name

        def _index(doc_id: str, source_path: str, extraction_result: dict) -> None:
            try:
                vs = ingestion_manager.km.get_qdrant_vectorstore()
                datapizza_pipeline = _build_datapizza_pipeline(vs, self.collection_name, self.chunk_max_char)
                restaurant_name = extraction_result.get("restaurant", {}).get("name", "unknown")

                # Passa il file temporaneo alla pipeline
                datapizza_pipeline.run(
                    file_path=tmp_path,
                    metadata=build_payload(
                        doc_id=doc_id,
                        source_path=source_path,
                        source_type=self.source_type,
                        text="",
                        restaurant=restaurant_name,
                    ),
                )
            finally:
                # Assicurati di eliminare il file temporaneo dopo l'uso
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return _index

    def read_db_entries_for_embedding(self, doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        """Recover DB Metadata"""
        row = facts_con.execute(
            "SELECT name FROM restaurants WHERE doc_id = ? LIMIT 1", [doc_id]
        ).fetchone()

        restaurant_name = row[0] if row else "unknown"
        return {"restaurant": {"name": restaurant_name}}