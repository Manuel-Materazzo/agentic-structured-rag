"""
blog_ingestion.py - Ingestion pipeline for blog HTML posts.

Parses HTML, chunks semantically, and indexes into the blog_index collection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

from src.app.config import CHUNK_MAX_CHAR_BLOG, COLLECTION_BLOG, EMBEDDING_DIM, KB_DIR, PARSED_DIR
from src.ingestion.runner import _get_qdrant_client, _log_get, _log_set_status, _log_upsert, sha256_file
from src.ingestion.shared import build_payload, normalize_whitespace, sectionize_html, split_text

log = logging.getLogger(__name__)

# TODO: configurable
BLOG_DIR: Path = KB_DIR / "blogpost"


def _discover_blog_files() -> list[Path]:
    return [p for p in sorted(BLOG_DIR.rglob("*")) if p.is_file()]


def _read_html(source_path: str) -> str:
    path = Path(source_path)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _parse_blog_sections(source_path: str) -> list[tuple[str, str]]:
    html_text = _read_html(source_path)
    sections = sectionize_html(html_text)
    if sections:
        return sections
    return [("blog", normalize_whitespace(html_text))]


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


def ingest_blog(
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
        sections = _parse_blog_sections(source_path_str)
        parsed_out = PARSED_DIR / f"{doc_id}.json"
        parsed_out.parent.mkdir(parents=True, exist_ok=True)
        parsed_out.write_text(
            json.dumps({"sections": [{"title": title, "text": text} for title, text in sections]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _log_set_status(log_con, doc_id, "parsed")

        facts_con.execute(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [doc_id, source_path_str, "blog"],
        )

        _log_set_status(log_con, doc_id, "extracting")
        _log_set_status(log_con, doc_id, "extracted")

        _log_set_status(log_con, doc_id, "embedding")
        chunks = []
        for section_title, section_text in sections:
            for idx, chunk_text in enumerate(split_text(section_text, CHUNK_MAX_CHAR_BLOG), start=1):
                chunks.append(
                    type("Chunk", (), {
                        "metadata": build_payload(
                            doc_id=doc_id,
                            source_path=source_path_str,
                            source_type="blog",
                            text=chunk_text,
                            section=f"{section_title}::{idx}",
                        )
                    })()
                )
        _upsert_chunks(_get_qdrant_client(), chunks, COLLECTION_BLOG)
        _log_set_status(log_con, doc_id, "indexed")
        _log_set_status(log_con, doc_id, "complete")
        return doc_id
    except Exception as exc:
        _log_set_status(log_con, doc_id, "failed", error_message=str(exc))
        raise
