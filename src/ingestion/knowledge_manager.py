"""
knowledge_manager.py — Handles DB initialization and operations.

Responsibilities:
- Create data/database/facts.db with the full schema (§7.3)
- Create data/database/ingestion_log.db with the ingestion_log table
- Create the four Qdrant collections (menu_index, manual_index, code_index, blog_index)
- Provide low-level CRUD operations for DuckDB, Qdrant, and the ingestion log.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Callable, Any

import duckdb
from qdrant_client import QdrantClient

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
from src.utils.ingestion_utils import delete_qdrant_points_by_doc_id

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_FACTS_SCHEMA_SQL = """
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
    macro_category         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS technique_licenses (
    technique     TEXT REFERENCES technique_taxonomy(technique),
    license_type  TEXT NOT NULL,
    license_grade INTEGER NOT NULL,
    PRIMARY KEY (technique, license_type, license_grade)
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

_INGESTION_LOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    doc_id        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    last_updated  TIMESTAMP NOT NULL,
    error_message TEXT
);
"""


class KnowledgeManager:
    """
    Manages storage backends (DuckDB, Qdrant) and ingestion log operations.
    """

    def __init__(
            self,
            facts_con: duckdb.DuckDBPyConnection,
            log_con: duckdb.DuckDBPyConnection,
            qdrant_client: QdrantClient,
    ) -> None:
        self.facts_con = facts_con
        self.log_con = log_con
        self.qdrant_client = qdrant_client
        self._vectorstore = None

    # ------------------------------------------------------------------
    # Factory / context manager
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "KnowledgeManager":
        """Initialise all storage backends and return a ready manager."""
        km = cls.__new__(cls)
        km.facts_con = cls._init_facts_db()
        km.log_con = cls._init_ingestion_log_db()
        km._vectorstore = None  # Inizializza la cache

        vs = km.get_qdrant_vectorstore()
        client = vs.get_client()

        # Creazione collections Qdrant
        from qdrant_client.models import Distance, VectorParams
        existing_names = {c.name for c in client.get_collections().collections}
        for name in ALL_COLLECTIONS:
            if name not in existing_names:
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                log.info("Created Qdrant collection: %s", name)
            else:
                log.info("Qdrant collection already exists: %s", name)

        km.qdrant_client = client
        return km

    def close(self) -> None:
        self.facts_con.close()
        self.log_con.close()
        if hasattr(self.qdrant_client, "close"):
            self.qdrant_client.close()

    def __enter__(self) -> "KnowledgeManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Storage initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_facts_db() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(FACTS_DB_PATH))
        con.execute(_FACTS_SCHEMA_SQL)
        log.info("facts.db ready at %s", FACTS_DB_PATH)
        return con

    @staticmethod
    def _init_ingestion_log_db() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(INGESTION_LOG_DB_PATH))
        con.execute(_INGESTION_LOG_SCHEMA_SQL)
        log.info("ingestion_log.db ready at %s", INGESTION_LOG_DB_PATH)
        return con

    def get_qdrant_vectorstore(self):
        """Return a cached QdrantVectorstore using the configured connection."""
        if self._vectorstore is not None:
            return self._vectorstore

        from datapizza.vectorstores.qdrant import QdrantVectorstore

        if QDRANT_HOST:
            vs = QdrantVectorstore(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
        else:
            is_local_path = (
                    QDRANT_LOCATION
                    and QDRANT_LOCATION != ":memory:"
                    and os.path.splitdrive(QDRANT_LOCATION)[0]
            )
            if is_local_path:
                vs = QdrantVectorstore.__new__(QdrantVectorstore)
                vs.host = None
                vs.port = 6333
                vs.api_key = None
                vs.kwargs = {"path": QDRANT_LOCATION}
            else:
                vs = QdrantVectorstore(location=QDRANT_LOCATION)

        self._vectorstore = vs
        return vs

    def get_facts_db(self):
        """Return a cached Duckdb using the configured connection."""
        if self.facts_con is None:
            self.facts_con = self._init_facts_db()

        return self.facts_con

    # ------------------------------------------------------------------
    # Ingestion-log operations
    # ------------------------------------------------------------------

    def log_upsert(self, doc_id: str, source_path: str, status: str, error_message: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc)
        self.log_con.execute(
            """
            INSERT INTO ingestion_log
                (doc_id, source_path, content_hash, status, last_updated, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (doc_id) DO UPDATE SET
                status        = excluded.status,
                last_updated  = excluded.last_updated,
                error_message = excluded.error_message
            """,
            [doc_id, source_path, doc_id, status, now, error_message],
        )

    def log_set_status(self, doc_id: str, status: str, error_message: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc)
        self.log_con.execute(
            "UPDATE ingestion_log SET status=?, last_updated=?, error_message=? WHERE doc_id=?",
            [status, now, error_message, doc_id],
        )

    def log_get(self, source_path: str) -> Optional[dict]:
        row = self.log_con.execute(
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

    def log_get_by_doc_id(self, doc_id: str) -> Optional[dict]:
        """Retrieve an ingestion log entry by its doc_id."""
        row = self.log_con.execute(
            "SELECT doc_id, source_path, content_hash, status, last_updated, error_message "
            "FROM ingestion_log WHERE doc_id = ?",
            [doc_id],
        ).fetchone()
        if row is None:
            return None
        return dict(zip(
            ["doc_id", "source_path", "content_hash", "status", "last_updated", "error_message"],
            row,
        ))

    def log_delete(self, doc_id: str = None, source_path: str = None) -> None:
        if doc_id:
            self.log_con.execute("DELETE FROM ingestion_log WHERE doc_id = ?", [doc_id])
        elif source_path:
            self.log_con.execute("DELETE FROM ingestion_log WHERE source_path = ?", [source_path])

    # ------------------------------------------------------------------
    # DuckDB operations
    # ------------------------------------------------------------------

    def write_document(
            self,
            extraction_result: dict[str, Any],
            doc_id: str,
            source_path: str,
            source_type: str,
            post_write_callback: Callable[[dict[str, Any], str, duckdb.DuckDBPyConnection], None] | None = None,
    ) -> None:
        """Persist extraction results to facts.db (idempotent)."""
        now = datetime.now(timezone.utc)
        self.facts_con.execute(
            """
            INSERT INTO documents (doc_id, source_path, source_type, ingested_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            [doc_id, source_path, source_type, now],
        )

        if post_write_callback:
            post_write_callback(extraction_result, doc_id, self.facts_con)
        else:
            log.warning("No specific DuckDB writer provided for source_type: %s", source_type)

    def cascade_delete_duckdb(self, doc_id: str) -> None:
        """Remove all DuckDB rows associated with a doc_id (in cascade order)."""
        self.facts_con.execute(
            "DELETE FROM dish_techniques WHERE dish_id IN "
            "(SELECT id FROM dishes WHERE doc_id = ?)", [doc_id]
        )
        self.facts_con.execute(
            "DELETE FROM dish_ingredients WHERE dish_id IN "
            "(SELECT id FROM dishes WHERE doc_id = ?)", [doc_id]
        )
        self.facts_con.execute("DELETE FROM dishes           WHERE doc_id = ?", [doc_id])
        self.facts_con.execute("DELETE FROM restaurants      WHERE doc_id = ?", [doc_id])
        self.facts_con.execute("DELETE FROM compliance_rules WHERE doc_id = ?", [doc_id])
        self.facts_con.execute("DELETE FROM documents        WHERE doc_id = ?", [doc_id])

    # ------------------------------------------------------------------
    # Qdrant operations
    # ------------------------------------------------------------------

    def delete_qdrant_points(self, doc_id: str) -> None:
        try:
            delete_qdrant_points_by_doc_id(self.qdrant_client, ALL_COLLECTIONS, doc_id)
        except Exception as exc:
            log.warning("Qdrant purge failed for %s: %s", doc_id, exc)

    # ------------------------------------------------------------------
    # Query helpers for maintenance
    # ------------------------------------------------------------------

    def get_failed_source_paths(self) -> list[str]:
        return [
            row[0] for row in self.log_con.execute(
                "SELECT source_path FROM ingestion_log WHERE status = 'failed'"
            ).fetchall()
        ]

    def get_complete_doc_ids(self) -> set[str]:
        return {
            row[0] for row in self.log_con.execute(
                "SELECT doc_id FROM ingestion_log WHERE status = 'complete'"
            ).fetchall()
        }

    def get_db_doc_ids(self) -> set[str]:
        return {
            row[0] for row in self.facts_con.execute("SELECT doc_id FROM documents").fetchall()
        }
