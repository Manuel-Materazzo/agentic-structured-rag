"""
runner.py — Entry point for the ingestion pipeline.

Responsibilities:
- Create data/database/facts.db with the full schema (§7.3)
- Create data/database/ingestion_log.db with the ingestion_log table
- Create the four Qdrant collections (menu_index, manual_index, code_index, blog_index)
- Expose ingest_document(), update_document(), delete_document() lifecycle functions
"""

from __future__ import annotations

import logging
import os

import duckdb

from src.app.config import (
    ALL_COLLECTIONS,
    EMBEDDING_DIM,
    FACTS_DB_PATH,
    INGESTION_LOG_DB_PATH,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_LOCATION,
    QDRANT_PORT,
)

log = logging.getLogger(__name__)


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
