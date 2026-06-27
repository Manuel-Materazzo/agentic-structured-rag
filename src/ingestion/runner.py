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
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Any

import duckdb
from qdrant_client import QdrantClient

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
from src.ingestion.shared import delete_qdrant_points_by_doc_id

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

VALID_STATUSES = frozenset(
    ["pending", "parsing", "parsed", "extracting", "extracted",
     "embedding", "indexed", "complete", "failed"]
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's binary content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# IngestionPipeline
# ---------------------------------------------------------------------------

class IngestionManager:
    """
    Manages the full document ingestion lifecycle across DuckDB and Qdrant.

    Typical usage::

        pipeline = IngestionPipeline.create()          # open / init all stores
        pipeline.ingest_document("path/to/menu.pdf")
        pipeline.update_document("path/to/menu.pdf")
        pipeline.delete_document("path/to/menu.pdf")
        pipeline.close()

    Or as a context manager::

        with IngestionPipeline.create() as pipeline:
            pipeline.ingest_document("path/to/menu.pdf")
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

    # ------------------------------------------------------------------
    # Factory / context manager
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "IngestionManager":
        """Initialise all storage backends and return a ready pipeline."""
        facts_con = cls._init_facts_db()
        log_con = cls._init_ingestion_log_db()
        qdrant_client = cls._init_qdrant_collections()
        return cls(facts_con, log_con, qdrant_client)

    def close(self) -> None:
        self.facts_con.close()
        self.log_con.close()
        # QdrantClient has no explicit close in all versions; guard defensively.
        if hasattr(self.qdrant_client, "close"):
            self.qdrant_client.close()

    def __enter__(self) -> "IngestionManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Storage initialisation (class-level helpers)
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

    @staticmethod
    def _init_qdrant_collections() -> QdrantClient:
        from qdrant_client.models import Distance, VectorParams

        vs = IngestionManager.get_qdrant_vectorstore()
        client = vs.get_client()

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

        return client

    @staticmethod
    def get_qdrant_vectorstore():
        """Return a QdrantVectorstore using the configured connection."""
        from datapizza.vectorstores.qdrant import QdrantVectorstore

        if QDRANT_HOST:
            return QdrantVectorstore(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

        # Windows local paths (e.g. C:\...) must be passed as 'path', not 'location',
        # otherwise QdrantClient mistakes 'c' as a URL scheme.
        # TODO: open an issue on datapizza-ai to fix the wrapper validation.
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
            return vs

        return QdrantVectorstore(location=QDRANT_LOCATION)

    # ------------------------------------------------------------------
    # Ingestion-log helpers
    # ------------------------------------------------------------------

    def _log_upsert(
            self,
            doc_id: str,
            source_path: str,
            status: str,
            error_message: Optional[str] = None,
    ) -> None:
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

    def _log_set_status(
            self,
            doc_id: str,
            status: str,
            error_message: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.log_con.execute(
            "UPDATE ingestion_log SET status=?, last_updated=?, error_message=? WHERE doc_id=?",
            [status, now, error_message, doc_id],
        )

    def _log_get(self, source_path: str) -> Optional[dict]:
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

    # ------------------------------------------------------------------
    # DuckDB cascade delete
    # ------------------------------------------------------------------

    def _cascade_delete_duckdb(self, doc_id: str) -> None:
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
    # Document lifecycle
    # ------------------------------------------------------------------

    def ingest_document(
            self,
            source_path: str,
            system_prompt: str,
            post_write_callback: Callable[[dict[str, Any], str, duckdb.DuckDBPyConnection], None] | None = None,
            source_type: str = "menu",
            use_vision_fallback: bool = True,
            skip_embedding: bool = False,
            vector_indexer=None,
    ) -> str:
        """
        Full ingestion lifecycle for a single document.

        States: pending → parsing → parsed → extracting → extracted
                       → embedding → indexed → complete
        On exception: → failed

        Args:
            post_write_callback: Hook to allow writing to a specific table based on source_type.
            system_prompt: System prompt to ba used during document extraction.
            source_path: Path to the source file.
            source_type: One of "menu", "manual", "code", "blog".
            use_vision_fallback: Re-extract via OCR when parsing confidence is low.
            skip_embedding: Skip the vector-indexing step entirely.
            vector_indexer: Optional callable ``(doc_id, source_path, extraction_result) -> None``
                that handles chunking, embedding, and upserting into Qdrant. When
                omitted the embedding step is skipped regardless of skip_embedding.

        Returns:
            The doc_id (SHA-256 of the source file).
        """
        path = Path(source_path)
        doc_id = _sha256_file(path)

        existing = self._log_get(source_path)
        if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
            log.info("Skip (already complete): %s", source_path)
            return doc_id

        self._log_upsert(doc_id, source_path, "pending")

        try:
            # 1. Parsing + LLM entity extraction
            self._log_set_status(doc_id, "parsing")
            from src.ingestion.structured_extraction import extract_entities
            extraction_result = extract_entities(
                source_path=source_path,
                system_prompt=system_prompt,
                source_type=source_type,
                doc_id=doc_id,
            )

            if use_vision_fallback and extraction_result.get("parsing_confidence") == "low":
                log.warning("Low confidence for %s, switching to vision fallback", source_path)
                from src.ingestion.vision_fallback import parse_with_vision
                raw_text = parse_with_vision(source_path)
                extraction_result = extract_entities(
                    source_path=source_path,
                    system_prompt=system_prompt,
                    source_type=source_type,
                    doc_id=doc_id,
                    raw_text=raw_text,
                )

            parsed_out = PARSED_DIR / f"{doc_id}.json"
            parsed_out.parent.mkdir(parents=True, exist_ok=True)
            with open(parsed_out, "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, ensure_ascii=False, indent=2)

            self._log_set_status(doc_id, "parsed")

            # 2. Structured store (DuckDB)
            self._log_set_status(doc_id, "extracting")
            from src.ingestion.structured_extraction import write_to_duckdb
            write_to_duckdb(extraction_result, doc_id, source_path, source_type, self.facts_con, post_write_callback)
            self._log_set_status(doc_id, "extracted")

            # 3. Vector index
            if not skip_embedding and vector_indexer is not None:
                self._log_set_status(doc_id, "embedding")
                vector_indexer(doc_id, source_path, extraction_result)
                self._log_set_status(doc_id, "indexed")

            self._log_set_status(doc_id, "complete")
            log.info("Ingestion complete: %s (%s)", source_path, doc_id)
            return doc_id

        except Exception as exc:
            self._log_set_status(doc_id, "failed", error_message=str(exc))
            log.error("Ingestion failed for %s: %s", source_path, exc, exc_info=True)
            raise

    def update_document(self, source_path: str, **kwargs) -> str:
        """
        Update a document: invalidate old artefacts, then re-ingest.
        Follows the 'invalidate first, insert after' protocol.
        """
        path = Path(source_path)
        new_doc_id = _sha256_file(path)

        existing = self._log_get(source_path)
        if not existing:
            return self.ingest_document(source_path, **kwargs)
        if existing["doc_id"] == new_doc_id:
            log.info("No change detected for %s, skipping update.", source_path)
            return new_doc_id

        old_doc_id = existing["doc_id"]
        log.info("Updating %s: %s… → %s…", source_path, old_doc_id[:8], new_doc_id[:8])

        # 1. Purge from vector store
        try:
            vs = self.get_qdrant_vectorstore()
            client = vs.get_client()
            delete_qdrant_points_by_doc_id(client, ALL_COLLECTIONS, old_doc_id)
        except Exception as exc:
            log.warning("Qdrant purge failed for %s: %s", old_doc_id, exc)

        # 2. Cascade delete from DuckDB
        self._cascade_delete_duckdb(old_doc_id)

        # 3. Remove parsed artefact
        parsed_out = PARSED_DIR / f"{old_doc_id}.json"
        if parsed_out.exists():
            parsed_out.unlink()

        # 4. Remove stale ingestion log entry
        self.log_con.execute("DELETE FROM ingestion_log WHERE doc_id = ?", [old_doc_id])

        # 5. Re-ingest
        return self.ingest_document(source_path, **kwargs)

    def delete_document(self, source_path: str) -> None:
        """Fully remove a document from all stores."""
        existing = self._log_get(source_path)
        if not existing:
            log.info("Nothing to delete for %s", source_path)
            return

        doc_id = existing["doc_id"]
        log.info("Deleting %s (%s…)", source_path, doc_id[:8])

        # 1. Qdrant
        try:
            vs = self.get_qdrant_vectorstore()
            client = vs.get_client()
            delete_qdrant_points_by_doc_id(client, ALL_COLLECTIONS, doc_id)
        except Exception as exc:
            log.warning("Qdrant delete failed for %s: %s", doc_id, exc)

        # 2. DuckDB cascade
        self._cascade_delete_duckdb(doc_id)

        # 3. Parsed artefact
        parsed_out = PARSED_DIR / f"{doc_id}.json"
        if parsed_out.exists():
            parsed_out.unlink()

        # 4. Ingestion log
        self.log_con.execute("DELETE FROM ingestion_log WHERE source_path = ?", [source_path])
        log.info("Deleted %s", source_path)

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------

    def retry_failed(self, **kwargs) -> list[str]:
        """Re-attempt ingestion for all documents with status='failed'."""
        rows = self.log_con.execute(
            "SELECT source_path FROM ingestion_log WHERE status = 'failed'"
        ).fetchall()
        retried = []
        for (source_path,) in rows:
            try:
                self.ingest_document(source_path, **kwargs)
                retried.append(source_path)
            except Exception as exc:
                log.error("Retry failed for %s: %s", source_path, exc)
        return retried

    def health_check(self, **kwargs) -> list[str]:
        """
        Check consistency between DuckDB and the ingestion log.
        Documents marked complete in the log but absent from the documents
        table are re-ingested automatically.
        """
        log_doc_ids = {
            row[0]
            for row in self.log_con.execute(
                "SELECT doc_id FROM ingestion_log WHERE status = 'complete'"
            ).fetchall()
        }
        db_doc_ids = {
            row[0]
            for row in self.facts_con.execute("SELECT doc_id FROM documents").fetchall()
        }
        orphans = log_doc_ids - db_doc_ids
        if orphans:
            log.warning("Health check: %d orphan doc_id(s) in log not in DuckDB", len(orphans))

        reprocessed = []
        for doc_id in orphans:
            row = self.log_con.execute(
                "SELECT source_path FROM ingestion_log WHERE doc_id = ?", [doc_id]
            ).fetchone()
            if not row:
                continue
            source_path = row[0]
            log.info("Health check: re-ingesting orphan %s", source_path)
            try:
                self.update_document(source_path, **kwargs)
                reprocessed.append(source_path)
            except Exception as exc:
                log.error("Health check re-ingest failed for %s: %s", source_path, exc)

        return reprocessed


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with IngestionManager.create() as pipeline:
        print("✅ All databases and Qdrant collections initialised successfully.")
