"""
ingestion_manager.py — Entry point for the ingestion pipeline.

Responsibilities:
- Handle data ingestion, transformation, and consistency.
- Orchestrate the document lifecycle (ingest, update, delete).
- Coordinate parsing, extraction, and vector indexing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Any

import duckdb

from src.app.config import PARSED_DIR
from src.ingestion.knowledge_manager import KnowledgeManager

log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's binary content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class IngestionManager:
    """
    Manages the full document ingestion lifecycle.
    Delegates storage operations to KnowledgeManager.
    """

    def __init__(self, knowledge_manager: KnowledgeManager) -> None:
        self.km = knowledge_manager

    @classmethod
    def create(cls) -> "IngestionManager":
        """Initialise storage backends and return a ready pipeline."""
        return cls(KnowledgeManager.create())

    def close(self) -> None:
        self.km.close()

    def __enter__(self) -> "IngestionManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------

    def ingest_document(
            self,
            source_path: str,
            system_prompt: str,
            post_write_callback: Callable[[dict[str, Any], str, duckdb.DuckDBPyConnection], None] | None = None,
            source_type: str = "menu",
            raw_text: Optional[str] = None,
            use_vision_fallback: bool = True,
            skip_embedding: bool = False,
            vector_indexer=None,
    ) -> str:
        """
        Full ingestion lifecycle for a single document.
        """
        path = Path(source_path)
        doc_id = _sha256_file(path)

        existing = self.km.log_get(source_path)
        if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
            log.info("Skip (already complete): %s", source_path)
            return doc_id

        self.km.log_upsert(doc_id, source_path, "pending")

        try:
            # 1. Parsing + LLM entity extraction
            self.km.log_set_status(doc_id, "parsing")
            from src.ingestion.structured_extraction import extract_entities
            extraction_result = extract_entities(
                source_path=source_path,
                system_prompt=system_prompt,
                source_type=source_type,
                doc_id=doc_id,
                raw_text=raw_text,
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

            parsed_out = PARSED_DIR / f"{doc_id}.txt"
            parsed_out.parent.mkdir(parents=True, exist_ok=True)
            with open(parsed_out, "w", encoding="utf-8") as f:
                json.dump(extraction_result, f, ensure_ascii=False, indent=2)

            self.km.log_set_status(doc_id, "parsed")

            # 2. Structured store (DuckDB)
            self.km.log_set_status(doc_id, "extracting")
            self.km.write_document(extraction_result, doc_id, source_path, source_type, post_write_callback)
            self.km.log_set_status(doc_id, "extracted")

            # 3. Vector index
            if not skip_embedding and vector_indexer is not None:
                self.km.log_set_status(doc_id, "embedding")
                vector_indexer(doc_id, source_path, extraction_result)
                self.km.log_set_status(doc_id, "indexed")

            self.km.log_set_status(doc_id, "complete")
            log.info("Ingestion complete: %s (%s)", source_path, doc_id)
            return doc_id

        except Exception as exc:
            self.km.log_set_status(doc_id, "failed", error_message=str(exc))
            log.error("Ingestion failed for %s: %s", source_path, exc, exc_info=True)
            raise

    def update_document(self, source_path: str, **kwargs) -> str:
        """
        Update a document: invalidate old artefacts, then re-ingest.
        Follows the 'invalidate first, insert after' protocol.
        """
        path = Path(source_path)
        new_doc_id = _sha256_file(path)

        existing = self.km.log_get(source_path)
        if not existing:
            return self.ingest_document(source_path, **kwargs)
        if existing["doc_id"] == new_doc_id:
            log.info("No change detected for %s, skipping update.", source_path)
            return new_doc_id

        old_doc_id = existing["doc_id"]
        log.info("Updating %s: %s… → %s…", source_path, old_doc_id[:8], new_doc_id[:8])

        # 1. Purge from vector store
        self.km.delete_qdrant_points(old_doc_id)

        # 2. Cascade delete from DuckDB
        self.km.cascade_delete_duckdb(old_doc_id)

        # 3. Remove parsed artefact
        parsed_out = PARSED_DIR / f"{old_doc_id}.txt"
        if parsed_out.exists():
            parsed_out.unlink()

        # 4. Remove stale ingestion log entry
        self.km.log_delete(doc_id=old_doc_id)

        # 5. Re-ingest
        return self.ingest_document(source_path, **kwargs)

    def delete_document(self, source_path: str) -> None:
        """Fully remove a document from all stores."""
        existing = self.km.log_get(source_path)
        if not existing:
            log.info("Nothing to delete for %s", source_path)
            return

        doc_id = existing["doc_id"]
        log.info("Deleting %s (%s…)", source_path, doc_id[:8])

        # 1. Qdrant
        self.km.delete_qdrant_points(doc_id)

        # 2. DuckDB cascade
        self.km.cascade_delete_duckdb(doc_id)

        # 3. Parsed artefact
        parsed_out = PARSED_DIR / f"{doc_id}.txt"
        if parsed_out.exists():
            parsed_out.unlink()

        # 4. Ingestion log
        self.km.log_delete(source_path=source_path)
        log.info("Deleted %s", source_path)

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------

    def retry_failed(self, **kwargs) -> list[str]:
        """Re-attempt ingestion for all documents with status='failed'."""
        retried = []
        for source_path in self.km.get_failed_source_paths():
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
        log_doc_ids = self.km.get_complete_doc_ids()
        db_doc_ids = self.km.get_db_doc_ids()

        orphans = log_doc_ids - db_doc_ids
        if orphans:
            log.warning("Health check: %d orphan doc_id(s) in log not in DuckDB", len(orphans))

        reprocessed = []
        for doc_id in orphans:
            existing = self.km.log_get_by_doc_id(doc_id)
            if not existing:
                continue
            source_path = existing["source_path"]
            log.info("Health check: re-ingesting orphan %s", source_path)
            try:
                self.update_document(source_path, **kwargs)
                reprocessed.append(source_path)
            except Exception as exc:
                log.error("Health check re-ingest failed for %s: %s", source_path, exc)

        return reprocessed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with KnowledgeManager.create() as km:
        pipeline = IngestionManager(km)
        print("✅ All databases and Qdrant collections initialised successfully.")