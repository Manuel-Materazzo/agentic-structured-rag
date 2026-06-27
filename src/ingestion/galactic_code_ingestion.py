"""
galactic_code_ingestion.py - Ingestion pipeline for the Codice Galattico.

Populates compliance_rules in DuckDB and indexes normative chunks in Qdrant.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import duckdb

from src.app.config import CHUNK_MAX_CHAR_CODE, COLLECTION_CODE, EMBEDDING_DIM, KB_DIR, PARSED_DIR
from src.ingestion.runner import _get_qdrant_client, _log_get, _log_set_status, _log_upsert, sha256_file
from src.ingestion.structured_extraction import extract_entities
from src.ingestion.shared import build_payload, normalize_whitespace, read_text_fallback, split_text

log = logging.getLogger(__name__)

# TODO: configurable
CODE_DIR: Path = KB_DIR / "codice_galattico"


def _discover_code_files() -> list[Path]:
    return sorted(CODE_DIR.glob("*"))


def _parse_code_text(source_path: str) -> str:
    try:
        from datapizza.modules.parsers.docling import DoclingParser

        parser = DoclingParser()
        nodes = parser([source_path])
        text = "\n\n".join(getattr(node, "text", "") for node in nodes if getattr(node, "text", ""))
        if text.strip():
            return normalize_whitespace(text)
    except Exception as exc:
        log.warning("DoclingParser failed for code %s: %s", source_path, exc)
    return normalize_whitespace(read_text_fallback(Path(source_path)))


def _extract_compliance_rules(text: str) -> list[tuple[str, str, str, str | None]]:
    """
    Conservative extraction of compliance rules from the code text.
    """
    rules: list[tuple[str, str, str, str | None]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if re.search(r"\b(limite|massimo|minimo|vietato|consentito|licenza)\b", line, re.I):
            subject = line[:120]
            rules.append(("norm_rule", subject, line[:240], None))
    return rules


def _write_rules(rows: list[tuple[str, str, str, str | None]], doc_id: str,
                 facts_con: duckdb.DuckDBPyConnection) -> int:
    inserted = 0
    for rule_type, subject, constraint_value, scope in rows:
        facts_con.execute(
            """
            INSERT INTO compliance_rules (id, rule_type, subject, constraint_value, scope, doc_id)
            VALUES (
                COALESCE((SELECT MAX(id) + 1 FROM compliance_rules), 1),
                ?, ?, ?, ?, ?
            )
            """,
            [rule_type, subject, constraint_value, scope, doc_id],
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


def ingest_code(
        source_path: str | Path,
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
        raw_text = _parse_code_text(source_path_str)
        extraction_result = extract_entities(source_path_str, "codice", doc_id, raw_text=raw_text)
        parsed_out = PARSED_DIR / f"{doc_id}.json"
        parsed_out.parent.mkdir(parents=True, exist_ok=True)
        parsed_out.write_text(json.dumps(extraction_result, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_set_status(log_con, doc_id, "parsed")

        _log_set_status(log_con, doc_id, "extracting")
        facts_con.execute(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [doc_id, source_path_str, "codice"],
        )
        _write_rules(_extract_compliance_rules(raw_text), doc_id, facts_con)
        _log_set_status(log_con, doc_id, "extracted")

        _log_set_status(log_con, doc_id, "embedding")
        chunks = []
        for idx, chunk_text in enumerate(split_text(raw_text, CHUNK_MAX_CHAR_CODE), start=1):
            chunks.append(
                type("Chunk", (), {
                    "metadata": build_payload(
                        doc_id=doc_id,
                        source_path=source_path_str,
                        source_type="codice",
                        text=chunk_text,
                        section=f"code_chunk_{idx}",
                    )
                })()
            )
        _upsert_chunks(_get_qdrant_client(), chunks, COLLECTION_CODE)
        _log_set_status(log_con, doc_id, "indexed")
        _log_set_status(log_con, doc_id, "complete")
        return doc_id
    except Exception as exc:
        _log_set_status(log_con, doc_id, "failed", error_message=str(exc))
        raise
