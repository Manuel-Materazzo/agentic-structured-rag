"""
runner.py — Entry point for the ingestion pipeline.

Responsibilities:
- Create data/database/facts.db with the full schema (§7.3)
- Create data/database/ingestion_log.db with the ingestion_log table
- Create the four Qdrant collections (menu_index, manual_index, code_index, blog_index)
- Expose ingest_document(), update_document(), delete_document() lifecycle functions
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from src.app.config import (
    ALL_COLLECTIONS,
    EMBEDDING_DIM,
    FACTS_DB_PATH,
    INGESTION_LOG_DB_PATH,
    PARSED_DIR,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_LOCATION,
    QDRANT_PORT,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's binary content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# DuckDB — facts.db
# ---------------------------------------------------------------------------

FACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source_path  TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    ingested_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS restaurants (
    id                  INTEGER PRIMARY KEY,
    name                TEXT NOT NULL,
    chef                TEXT,
    planet              TEXT,
    chef_license        TEXT,
    professional_orders TEXT[],
    doc_id              TEXT REFERENCES documents(doc_id)
);

CREATE SEQUENCE IF NOT EXISTS restaurants_id_seq START 1;

CREATE TABLE IF NOT EXISTS dishes (
    id                INTEGER PRIMARY KEY,
    name              TEXT NOT NULL,
    restaurant_id     INTEGER REFERENCES restaurants(id),
    preparation_notes TEXT,
    doc_id            TEXT REFERENCES documents(doc_id)
);

CREATE SEQUENCE IF NOT EXISTS dishes_id_seq START 1;

CREATE TABLE IF NOT EXISTS dish_ingredients (
    dish_id         INTEGER REFERENCES dishes(id),
    ingredient      TEXT NOT NULL,
    quantity_grams  FLOAT,
    quantity_raw    TEXT NOT NULL,
    is_regulated    BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (dish_id, ingredient)
);

CREATE TABLE IF NOT EXISTS dish_techniques (
    dish_id   INTEGER REFERENCES dishes(id),
    technique TEXT NOT NULL,
    PRIMARY KEY (dish_id, technique)
);

CREATE TABLE IF NOT EXISTS technique_taxonomy (
    technique              TEXT PRIMARY KEY,
    macro_category         TEXT NOT NULL,
    required_license_level TEXT
);

CREATE TABLE IF NOT EXISTS planet_distances (
    planet_a    TEXT NOT NULL,
    planet_b    TEXT NOT NULL,
    distance_ly FLOAT NOT NULL,
    PRIMARY KEY (planet_a, planet_b)
);

CREATE TABLE IF NOT EXISTS compliance_rules (
    id               INTEGER PRIMARY KEY,
    rule_type        TEXT NOT NULL,
    subject          TEXT NOT NULL,
    constraint_value TEXT NOT NULL,
    scope            TEXT,
    doc_id           TEXT REFERENCES documents(doc_id)
);

CREATE SEQUENCE IF NOT EXISTS compliance_rules_id_seq START 1;
"""


def init_facts_db() -> duckdb.DuckDBPyConnection:
    """Create/open facts.db and ensure the schema exists."""
    con = duckdb.connect(str(FACTS_DB_PATH))
    con.execute(FACTS_SCHEMA_SQL)
    log.info(f"facts.db ready at {FACTS_DB_PATH}")
    return con


# ---------------------------------------------------------------------------
# DuckDB — ingestion_log.db
# ---------------------------------------------------------------------------

INGESTION_LOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    doc_id        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    last_updated  TIMESTAMP NOT NULL,
    error_message TEXT
);
"""

VALID_STATUSES = frozenset(
    ["pending", "parsing", "parsed", "extracting", "extracted",
     "embedding", "indexed", "complete", "failed"]
)


def init_ingestion_log_db() -> duckdb.DuckDBPyConnection:
    """Create/open ingestion_log.db and ensure the schema exists."""
    con = duckdb.connect(str(INGESTION_LOG_DB_PATH))
    con.execute(INGESTION_LOG_SCHEMA_SQL)
    log.info(f"ingestion_log.db ready at {INGESTION_LOG_DB_PATH}")
    return con


# ---------------------------------------------------------------------------
# Ingestion log helpers
# ---------------------------------------------------------------------------

def _log_upsert(
    log_con: duckdb.DuckDBPyConnection,
    doc_id: str,
    source_path: str,
    status: str,
    error_message: Optional[str] = None,
):
    now = datetime.now(timezone.utc)
    log_con.execute(
        """
        INSERT INTO ingestion_log (doc_id, source_path, content_hash, status, last_updated, error_message)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (doc_id) DO UPDATE SET
            status = excluded.status,
            last_updated = excluded.last_updated,
            error_message = excluded.error_message
        """,
        [doc_id, source_path, doc_id, status, now, error_message],
    )


def _log_set_status(
    log_con: duckdb.DuckDBPyConnection,
    doc_id: str,
    status: str,
    error_message: Optional[str] = None,
):
    now = datetime.now(timezone.utc)
    log_con.execute(
        """
        UPDATE ingestion_log
        SET status = ?, last_updated = ?, error_message = ?
        WHERE doc_id = ?
        """,
        [status, now, error_message, doc_id],
    )


def _log_get(
    log_con: duckdb.DuckDBPyConnection,
    source_path: str,
) -> Optional[dict]:
    row = log_con.execute(
        "SELECT doc_id, source_path, content_hash, status, last_updated, error_message "
        "FROM ingestion_log WHERE source_path = ?",
        [source_path],
    ).fetchone()
    if row is None:
        return None
    return dict(zip(
        ["doc_id", "source_path", "content_hash", "status", "last_updated", "error_message"],
        row,
    ))


# ---------------------------------------------------------------------------
# Qdrant — collection initialisation
# ---------------------------------------------------------------------------

def _get_qdrant_client():
    """Return a QdrantVectorstore instance using the configured connection."""
    from datapizza.vectorstores.qdrant import QdrantVectorstore

    if QDRANT_HOST:
        return QdrantVectorstore(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
    else:
        # Windows local paths (e.g. C:\...) must be passed as 'path',
        # not 'location', otherwise QdrantClient mistakes 'c' as a URL scheme.
        # The wrapper validation requires 'location' in kwargs, so passing only 'path' does not work.
        # Passing both 'location' and 'path' throws a qdrant error.
        # TODO: open an issue on datapizza-ai to fix this.
        # in the meantime, i'll monkeypatch the class to handle this case.
        is_local_path = QDRANT_LOCATION and QDRANT_LOCATION != ":memory:" and os.path.splitdrive(QDRANT_LOCATION)[0]

        if is_local_path:
            vs = QdrantVectorstore.__new__(QdrantVectorstore)
            vs.host = None
            vs.port = 6333
            vs.api_key = None
            vs.kwargs = {"path": QDRANT_LOCATION}
            return vs
        else:
            return QdrantVectorstore(location=QDRANT_LOCATION)


def init_qdrant_collections() -> None:
    """Ensure the four Qdrant collections exist with the correct vector config."""
    from qdrant_client.models import Distance, VectorParams

    vs = _get_qdrant_client()
    client = vs.get_client()

    for collection_name in ALL_COLLECTIONS:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            log.info(f"Created Qdrant collection: {collection_name}")
        else:
            log.info(f"Qdrant collection already exists: {collection_name}")


# ---------------------------------------------------------------------------
# Document lifecycle: ingest / update / delete
# ---------------------------------------------------------------------------

def _cascade_delete_duckdb(facts_con: duckdb.DuckDBPyConnection, doc_id: str):
    """Remove all DuckDB rows associated with a doc_id (cascade order)."""
    facts_con.execute(
        "DELETE FROM dish_techniques WHERE dish_id IN "
        "(SELECT id FROM dishes WHERE doc_id = ?)", [doc_id]
    )
    facts_con.execute(
        "DELETE FROM dish_ingredients WHERE dish_id IN "
        "(SELECT id FROM dishes WHERE doc_id = ?)", [doc_id]
    )
    facts_con.execute("DELETE FROM dishes WHERE doc_id = ?", [doc_id])
    facts_con.execute("DELETE FROM restaurants WHERE doc_id = ?", [doc_id])
    facts_con.execute("DELETE FROM compliance_rules WHERE doc_id = ?", [doc_id])
    facts_con.execute("DELETE FROM documents WHERE doc_id = ?", [doc_id])


def ingest_document(
    source_path: str,
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    structured_extractor=None,
    vision_fallback=None,
    node_splitter=None,
    chunk_embedder=None,
    qdrant_vs=None,
    collection_name: str = "menu_index",
    source_type: str = "menu",
    skip_embedding: bool = False,
) -> str:
    """
    Full ingestion lifecycle for a single document.

    States: pending → parsing → parsed → extracting → extracted → embedding → indexed → complete
    On exception: → failed

    Returns the doc_id.
    """
    path = Path(source_path)
    doc_id = sha256_file(path)

    existing = _log_get(log_con, source_path)
    if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
        log.info(f"Skip (already complete): {source_path}")
        return doc_id

    _log_upsert(log_con, doc_id, source_path, "pending")

    try:
        # ── 1. Parsing ───────────────────────────────────────────────────
        _log_set_status(log_con, doc_id, "parsing")

        parsed_text: str = ""
        parsing_confidence: str = "high"

        if structured_extractor is not None:
            from src.ingestion.structured_extraction import extract_entities
            extraction_result = extract_entities(
                source_path=source_path,
                source_type=source_type,
                doc_id=doc_id,
            )
            parsing_confidence = extraction_result.get("parsing_confidence", "high")

            if parsing_confidence == "low" and vision_fallback is not None:
                log.warning(f"Low confidence for {source_path}, switching to vision fallback")
                from src.ingestion.vision_fallback import parse_with_vision
                parsed_text = parse_with_vision(source_path)
                extraction_result = extract_entities(
                    source_path=source_path,
                    source_type=source_type,
                    doc_id=doc_id,
                    raw_text=parsed_text,
                )

            # Save parsed output
            parsed_out = PARSED_DIR / f"{doc_id}.json"
            import json
            with open(parsed_out, "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, ensure_ascii=False, indent=2)

        _log_set_status(log_con, doc_id, "parsed")

        # ── 2. Structured store ──────────────────────────────────────────
        _log_set_status(log_con, doc_id, "extracting")

        if structured_extractor is not None:
            from src.ingestion.structured_extraction import write_to_duckdb
            write_to_duckdb(extraction_result, doc_id, source_path, source_type, facts_con)

        _log_set_status(log_con, doc_id, "extracted")

        # ── 3. Vector index ──────────────────────────────────────────────
        if not skip_embedding and chunk_embedder is not None and qdrant_vs is not None:
            _log_set_status(log_con, doc_id, "embedding")
            # Chunking + embedding is handled by the caller via IngestionPipeline
            _log_set_status(log_con, doc_id, "indexed")

        _log_set_status(log_con, doc_id, "complete")
        log.info(f"Ingestion complete: {source_path} ({doc_id})")
        return doc_id

    except Exception as exc:
        _log_set_status(log_con, doc_id, "failed", error_message=str(exc))
        log.error(f"Ingestion failed for {source_path}: {exc}", exc_info=True)
        raise


def update_document(
    source_path: str,
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    **kwargs,
) -> str:
    """
    Update a document: invalidate old artefacts, then re-ingest.
    Follows the 'invalidate first, insert after' protocol.
    """
    path = Path(source_path)
    new_doc_id = sha256_file(path)

    existing = _log_get(log_con, source_path)
    if not existing:
        return ingest_document(source_path, facts_con, log_con, **kwargs)
    if existing["doc_id"] == new_doc_id:
        log.info(f"No change detected for {source_path}, skipping update.")
        return new_doc_id

    old_doc_id = existing["doc_id"]
    log.info(f"Updating {source_path}: {old_doc_id[:8]}… → {new_doc_id[:8]}…")

    # 1. Purge from vector store
    try:
        vs = _get_qdrant_client()
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        client = vs.get_client()
        from src.app.config import ALL_COLLECTIONS
        for col in ALL_COLLECTIONS:
            client.delete(
                collection_name=col,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=old_doc_id))]
                ),
            )
    except Exception as exc:
        log.warning(f"Qdrant purge failed for {old_doc_id}: {exc}")

    # 2. Cascade delete from DuckDB
    _cascade_delete_duckdb(facts_con, old_doc_id)

    # 3. Remove parsed artefact
    parsed_out = PARSED_DIR / f"{old_doc_id}.json"
    if parsed_out.exists():
        parsed_out.unlink()

    # 4. Remove ingestion log entry for old doc_id
    log_con.execute("DELETE FROM ingestion_log WHERE doc_id = ?", [old_doc_id])

    # 5. Re-ingest
    return ingest_document(source_path, facts_con, log_con, **kwargs)


def delete_document(
    source_path: str,
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Fully remove a document from all stores.
    """
    existing = _log_get(log_con, source_path)
    if not existing:
        log.info(f"Nothing to delete for {source_path}")
        return

    doc_id = existing["doc_id"]
    log.info(f"Deleting {source_path} ({doc_id[:8]}…)")

    # 1. Qdrant
    try:
        vs = _get_qdrant_client()
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        client = vs.get_client()
        from src.app.config import ALL_COLLECTIONS
        for col in ALL_COLLECTIONS:
            client.delete(
                collection_name=col,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
    except Exception as exc:
        log.warning(f"Qdrant delete failed for {doc_id}: {exc}")

    # 2. DuckDB cascade
    _cascade_delete_duckdb(facts_con, doc_id)

    # 3. Parsed dir
    parsed_out = PARSED_DIR / f"{doc_id}.json"
    if parsed_out.exists():
        parsed_out.unlink()

    # 4. Ingestion log
    log_con.execute("DELETE FROM ingestion_log WHERE source_path = ?", [source_path])
    log.info(f"Deleted {source_path}")


def retry_failed(
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    **kwargs,
) -> list[str]:
    """Re-attempt ingestion for all documents with status='failed'."""
    rows = log_con.execute(
        "SELECT source_path FROM ingestion_log WHERE status = 'failed'"
    ).fetchall()
    retried = []
    for (source_path,) in rows:
        try:
            ingest_document(source_path, facts_con, log_con, **kwargs)
            retried.append(source_path)
        except Exception as exc:
            log.error(f"Retry failed for {source_path}: {exc}")
    return retried


def health_check(
    facts_con: duckdb.DuckDBPyConnection,
    log_con: duckdb.DuckDBPyConnection,
    **kwargs,
) -> list[str]:
    """
    Check consistency between DuckDB and ingestion_log.
    Documents marked complete in log but missing from documents table → re-ingest.
    """
    log_doc_ids = {
        row[0]
        for row in log_con.execute(
            "SELECT doc_id FROM ingestion_log WHERE status = 'complete'"
        ).fetchall()
    }
    db_doc_ids = {
        row[0]
        for row in facts_con.execute("SELECT doc_id FROM documents").fetchall()
    }
    orphans_in_log = log_doc_ids - db_doc_ids
    if orphans_in_log:
        log.warning(f"Health check: {len(orphans_in_log)} orphan doc_ids in log not in DuckDB")

    # Find source paths for orphans and re-ingest
    reprocessed = []
    for doc_id in orphans_in_log:
        row = log_con.execute(
            "SELECT source_path FROM ingestion_log WHERE doc_id = ?", [doc_id]
        ).fetchone()
        if row:
            source_path = row[0]
            log.info(f"Health check: re-ingesting orphan {source_path}")
            try:
                update_document(source_path, facts_con, log_con, **kwargs)
                reprocessed.append(source_path)
            except Exception as exc:
                log.error(f"Health check re-ingest failed for {source_path}: {exc}")
    return reprocessed


# ---------------------------------------------------------------------------
# Main initialisation entry point
# ---------------------------------------------------------------------------

def init_all() -> tuple[duckdb.DuckDBPyConnection, duckdb.DuckDBPyConnection]:
    """
    Initialise all storage backends. Call this once at application start.

    Returns:
        (facts_con, log_con) — open DuckDB connections.
    """
    facts_con = init_facts_db()
    log_con = init_ingestion_log_db()
    init_qdrant_collections()
    return facts_con, log_con


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    facts_con, log_con = init_all()
    print("✅ All databases and Qdrant collections initialised successfully.")
    facts_con.close()
    log_con.close()
