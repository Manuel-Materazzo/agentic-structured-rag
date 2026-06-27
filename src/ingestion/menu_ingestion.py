"""
menu_ingestion.py — Ingestion pipeline for restaurant menu PDFs.

Parses every PDF under Dataset/knowledge_base/menu/ using DoclingParser,
runs LLM entity extraction, writes structured data to DuckDB,
and indexes chunks in the menu_index Qdrant collection.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import duckdb

from src.app.config import (
    CHUNK_MAX_CHAR_MENU,
    COLLECTION_MENU,
    EMBEDDING_MODEL,
    KB_DIR,
    OPENAI_API_KEY,
    PARSED_DIR,
)
from src.ingestion.runner import (
    _get_qdrant_client,
    _log_get,
    _log_set_status,
    _log_upsert,
    sha256_file,
)
from src.ingestion.structured_extraction import extract_entities, write_to_duckdb

log = logging.getLogger(__name__)

# TODO: extract to config
MENU_DIR: Path = KB_DIR / "menu"


def _build_ingestion_pipeline(vs, collection_name: str):
    """Build a datapizza-ai IngestionPipeline for menu PDFs."""
    from datapizza.embedders import ChunkEmbedder
    from datapizza.embedders.openai import OpenAIEmbedder
    from datapizza.modules.parsers.docling import DoclingParser
    from datapizza.modules.splitters import NodeSplitter
    from datapizza.pipeline import IngestionPipeline

    embedder_client = OpenAIEmbedder(api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL)

    pipeline = IngestionPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=CHUNK_MAX_CHAR_MENU),
            ChunkEmbedder(client=embedder_client),
        ],
        vector_store=vs,
        collection_name=collection_name,
    )
    return pipeline


def _enrich_chunk_metadata(
    chunks,
    doc_id: str,
    source_path: str,
    restaurant_name: str,
) -> list:
    """
    Add the Qdrant payload contract fields to each chunk:
    chunk_id, doc_id, source_path, source_type, restaurant, etc.
    """
    enriched = []
    for chunk in chunks:
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "source_path": source_path,
            "source_type": "menu",
            "restaurant": restaurant_name,
        })
        enriched.append(chunk)
    return enriched


def ingest_menu(
    pdf_path: Path,
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    skip_if_complete: bool = True,
) -> str:
    """
    Ingest a single menu PDF through the full pipeline.

    Returns the doc_id.
    """
    source_path = str(pdf_path)
    doc_id = sha256_file(pdf_path)

    # Skip if already complete
    if skip_if_complete:
        existing = _log_get(log_con, source_path)
        if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
            log.info(f"Skip (already complete): {pdf_path.name}")
            return doc_id

    _log_upsert(log_con, doc_id, source_path, "pending")

    try:
        # 1. Entity extraction (parsing + LLM)
        _log_set_status(log_con, doc_id, "parsing")
        extraction_result = extract_entities(
            source_path=source_path,
            source_type="menu",
            doc_id=doc_id,
        )

        # Vision fallback if confidence is low
        # TODO: implement

        # Save parsed JSON
        import json
        parsed_out = PARSED_DIR / f"{doc_id}.json"
        parsed_out.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed_out, "w", encoding="utf-8") as f:
            json.dump(extraction_result, f, ensure_ascii=False, indent=2)

        _log_set_status(log_con, doc_id, "parsed")

        # 2. Write to DuckDB
        _log_set_status(log_con, doc_id, "extracting")
        write_to_duckdb(extraction_result, doc_id, source_path, "menu", facts_con)
        _log_set_status(log_con, doc_id, "extracted")

        # 3. Index in Qdrant via IngestionPipeline
        _log_set_status(log_con, doc_id, "embedding")
        vs = _get_qdrant_client()
        pipeline = _build_ingestion_pipeline(vs, COLLECTION_MENU)

        restaurant_name = extraction_result.get("restaurant", {}).get("name", "unknown")

        # Run pipeline: DoclingParser → NodeSplitter → ChunkEmbedder → Qdrant upsert
        pipeline.run(
            file_path=source_path,
            metadata={
                "chunk_id_fn": lambda: str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_path": source_path,
                "source_type": "menu",
                "restaurant": restaurant_name,
            },
        )
        _log_set_status(log_con, doc_id, "indexed")

        _log_set_status(log_con, doc_id, "complete")
        log.info(f"✅ Menu ingested: {pdf_path.name} ({doc_id[:8]}…)")
        return doc_id

    except Exception as exc:
        _log_set_status(log_con, doc_id, "failed", error_message=str(exc))
        log.error(f"Menu ingestion failed for {pdf_path.name}: {exc}", exc_info=True)
        raise


def ingest_all_menus(
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    menu_dir: Path = MENU_DIR,
) -> list[str]:
    """
    Ingest all PDF menus found in menu_dir.

    Returns list of doc_ids successfully ingested.
    """
    pdf_files = sorted(menu_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in {menu_dir}")
        return []

    log.info(f"Found {len(pdf_files)} menu PDFs to ingest")
    doc_ids: list[str] = []

    for pdf_path in pdf_files:
        try:
            doc_id = ingest_menu(pdf_path, facts_con, log_con)
            doc_ids.append(doc_id)
        except Exception as exc:
            log.error(f"Failed to ingest {pdf_path.name}: {exc}")

    log.info(f"Menu ingestion complete: {len(doc_ids)}/{len(pdf_files)} succeeded")
    return doc_ids
