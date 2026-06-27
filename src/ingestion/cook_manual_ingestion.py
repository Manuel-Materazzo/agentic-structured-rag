"""
cook_manual_ingestion.py - Ingestion pipeline for the Manuale di Cucina.

Populates technique_taxonomy in DuckDB and indexes semantic chunks in Qdrant.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import duckdb

from src.app.config import CHUNK_MAX_CHAR_MANUAL, COLLECTION_MANUAL, EMBEDDING_DIM, KB_DIR, PARSED_DIR
from src.ingestion.runner import _get_qdrant_client, _log_get, _log_set_status, _log_upsert, sha256_file
from src.ingestion.structured_extraction import extract_entities
from src.ingestion.shared import build_payload, normalize_whitespace, read_text_fallback, split_text

log = logging.getLogger(__name__)

# TODO: configurable
MANUAL_PDF: Path = KB_DIR / "misc" / "Manuale di Cucina.pdf"


def _parse_manual_text(source_path: str) -> str:
    try:
        from datapizza.modules.parsers.docling import DoclingParser

        parser = DoclingParser()
        nodes = parser([source_path])
        text = "\n\n".join(getattr(node, "text", "") for node in nodes if getattr(node, "text", ""))
        if text.strip():
            return normalize_whitespace(text)
    except Exception as exc:
        log.warning("DoclingParser failed for manual %s: %s", source_path, exc)
    return normalize_whitespace(read_text_fallback(Path(source_path)))


def _extract_taxonomy(text: str) -> list[tuple[str, str, str | None]]:
    """
    Heuristic extraction of technique taxonomy rows from the manual.

    This is intentionally conservative: it creates entries only when a line
    looks like 'Tecnica - Macro Categoria - Licenza'.
    """
    rows: list[tuple[str, str, str | None]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in re.split(r"\s{2,}| \| | - ", line) if part.strip()]
        if len(parts) >= 2 and len(parts[0]) > 2:
            technique = parts[0][:120]
            macro_category = parts[1][:120]
            required_license = parts[2][:120] if len(parts) > 2 else None
            rows.append((technique, macro_category, required_license))
    return rows


def _write_taxonomy(rows: list[tuple[str, str, str | None]], doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> int:
    inserted = 0
    for technique, macro_category, required_license in rows:
        if not technique or not macro_category:
            continue
        facts_con.execute(
            """
            INSERT INTO technique_taxonomy (technique, macro_category, required_license_level)
            VALUES (?, ?, ?)
            ON CONFLICT (technique) DO UPDATE SET
                macro_category = excluded.macro_category,
                required_license_level = excluded.required_license_level
            """,
            [technique, macro_category, required_license],
        )
        inserted += 1
    return inserted


def _upsert_chunks(vs, chunks, collection_name: str):
    client = vs.get_client()
    from qdrant_client.models import PointStruct

    points = []
    for chunk in chunks:
        payload = chunk.metadata
        points.append(
            PointStruct(
                id=payload["chunk_id"],
                vector=[0.0] * EMBEDDING_DIM,
                payload=payload,
            )
        )
    if points:
        client.upsert(collection_name=collection_name, points=points)


def ingest_manual(
    source_path: str | Path = MANUAL_PDF,
    facts_con: duckdb.DuckDBPyConnection | None = None,
    log_con: duckdb.DuckDBPyConnection | None = None,
) -> str:
    path = Path(source_path)
    doc_id = sha256_file(path)
    source_path_str = str(path)

    if facts_con is None or log_con is None:
        from src.ingestion.runner import init_all
        facts_con, log_con = init_all()

    existing = _log_get(log_con, source_path_str)
    if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
        return doc_id

    _log_upsert(log_con, doc_id, source_path_str, "pending")
    try:
        _log_set_status(log_con, doc_id, "parsing")
        raw_text = _parse_manual_text(source_path_str)
        extraction_result = extract_entities(source_path_str, "manual", doc_id, raw_text=raw_text)
        parsed_out = PARSED_DIR / f"{doc_id}.json"
        parsed_out.parent.mkdir(parents=True, exist_ok=True)
        parsed_out.write_text(json.dumps(extraction_result, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_set_status(log_con, doc_id, "parsed")

        _log_set_status(log_con, doc_id, "extracting")
        facts_con.execute(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [doc_id, source_path_str, "manual"],
        )
        taxonomy_rows = _extract_taxonomy(raw_text)
        _write_taxonomy(taxonomy_rows, doc_id, facts_con)
        _log_set_status(log_con, doc_id, "extracted")

        _log_set_status(log_con, doc_id, "embedding")
        chunks = []
        for idx, chunk_text in enumerate(split_text(raw_text, CHUNK_MAX_CHAR_MANUAL), start=1):
            chunks.append(
                type("Chunk", (), {
                    "metadata": build_payload(
                        doc_id=doc_id,
                        source_path=source_path_str,
                        source_type="manual",
                        text=chunk_text,
                        section=f"manual_chunk_{idx}",
                    )
                })()
            )
        _upsert_chunks(_get_qdrant_client(), chunks, COLLECTION_MANUAL)
        _log_set_status(log_con, doc_id, "indexed")

        _log_set_status(log_con, doc_id, "complete")
        return doc_id
    except Exception as exc:
        _log_set_status(log_con, doc_id, "failed", error_message=str(exc))
        raise
