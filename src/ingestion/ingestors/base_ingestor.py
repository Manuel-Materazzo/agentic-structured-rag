"""
base_ingestor.py — Abstract base class for document-specific ingestors.

Defines the interface for handling specific data types, ensuring consistency
across different ingestion pipelines (menu, galactic code, etc.).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable
from src.app.config import PARSED_DIR

import duckdb

from src.ingestion.ingestion_manager import IngestionManager

log = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Abstract base class for all document ingestors."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """e.g., 'menu', 'codice', 'manual'."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The LLM system prompt used for structured extraction."""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """The target Qdrant collection name."""
        pass

    @property
    @abstractmethod
    def chunk_max_char(self) -> int:
        """Maximum character count for chunking."""
        pass

    @property
    @abstractmethod
    def directory(self) -> Path:
        """Directory where the source files are located."""
        pass

    @property
    def file_glob(self) -> str:
        """Glob pattern to find files in the directory (defaults to PDF)."""
        return "*"

    @property
    def use_vision_fallback(self) -> bool:
        """Whether to use OCR vision fallback if LLM confidence is low."""
        return True

    def parse_document(self, source_path: Path) -> str:
        """
        Parse a document to plain text using DoclingParser.
        Subclasses can override this to add custom normalization.
        """
        try:
            from datapizza.modules.parsers.docling import DoclingParser
            parser = DoclingParser()
            result = parser(str(source_path))

            # Rendi l'iterazione sicura
            nodes = result if isinstance(result, (list, tuple)) else [result]

            texts = []
            for n in nodes:
                text = getattr(n, "text", None) or getattr(n, "content", "")
                if text:
                    texts.append(text)

            text = "\n\n".join(texts)
            if text.strip():
                return text
        except Exception as exc:
            log.warning("DoclingParser failed for %s: %s", source_path, exc)

        return Path(source_path).read_text(encoding="utf-8", errors="ignore")

    @abstractmethod
    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str,
                         facts_con: duckdb.DuckDBPyConnection) -> None:
        pass

    @abstractmethod
    def read_db_entries_for_embedding(self, doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        """
        Retrieve metadata from DB for a doc_id.
        Needed to make Ingestion Phases independent.
        """
        pass

    @abstractmethod
    def make_vector_indexer(
            self,
            ingestion_manager: IngestionManager,
            source_path: Path,
            raw_text: str
    ) -> Callable[[str, str, dict], None]:
        """
        Return a callable that chunks, embeds, and upserts data into Qdrant.
        Signature of returned callable: (doc_id, source_path, extraction_result) -> None
        """
        pass

    def ingest_single(self, ingestion_manager: IngestionManager, source_path: Path) -> str:
        """Ingest a single document through the full pipeline."""

        # 1. Parse the document ONCE
        raw_text = self.parse_document(source_path)

        # 2. Prepare the vector indexer passing the already parsed text
        vector_indexer = self.make_vector_indexer(ingestion_manager, source_path, raw_text)

        return ingestion_manager.ingest_document(
            source_path=str(source_path),
            system_prompt=self.system_prompt,
            source_type=self.source_type,
            raw_text=raw_text,  # Pass raw_text to avoid re-parsing in extract_entities
            use_vision_fallback=self.use_vision_fallback,
            vector_indexer=vector_indexer,
            post_write_callback=self.write_db_entries,
        )

    def ingest_all(self, ingestion_manager: IngestionManager) -> list[str]:
        """Orchestrate ingestion in three sequential phases."""
        files = self._get_files()
        if not files:
            return []

        parsed_docs = self._phase_parse(files, ingestion_manager)
        extracted_docs = self._phase_extract(parsed_docs, ingestion_manager)
        return self._phase_embed(extracted_docs, ingestion_manager, total=len(files))

    def _phase_parse(
            self,
            files: list[Path],
            ingestion_manager: IngestionManager,
    ) -> list[tuple[str, str, Path]]:
        """Phase 1 – parse every file; return (doc_id, source_path, cache_path) triples."""
        log.info("Phase 1: Parse all documents")
        PARSED_DIR.mkdir(parents=True, exist_ok=True)
        parsed_docs: list[tuple[str, str, Path]] = []

        for file_path in files:
            doc_id = self._sha256_file(file_path)
            source_path = str(file_path)
            existing = ingestion_manager.km.log_get(source_path)

            # Check and skip documents that have already completed fist ingestion phase
            if existing and existing["doc_id"] == doc_id and existing["status"] in [
                "complete", "indexed", "embedding", "extracted"
            ]:
                cache_path = PARSED_DIR / f"{doc_id}.txt"
                if cache_path.exists():
                    log.info("Loading cached parsed text for %s", file_path.name)
                    parsed_docs.append((doc_id, source_path, cache_path))
                    continue
                else:
                    # Safety fallback: if cache was deleted but DB says "extracted",
                    # we must reparse, but without touching DB status!
                    log.warning("Cache missing for %s, reparsing without changing DB status.", file_path.name)
                    raw_text = self.parse_document(file_path)
                    cache_path.write_text(raw_text, encoding="utf-8")
                    parsed_docs.append((doc_id, source_path, cache_path))
                    continue

            # File Change Handling (Update)
            if existing and existing["doc_id"] != doc_id:
                self._handle_file_change(existing["doc_id"], doc_id, source_path, ingestion_manager)
                # Now existing has been deleted, proceed as if it's a new file

            # New file or failed during parsing: update status to "parsing"
            ingestion_manager.km.log_upsert(doc_id, source_path, "parsing")

            cache_path = PARSED_DIR / f"{doc_id}.txt"

            if cache_path.exists():
                log.info("Loading cached parsed text for %s", file_path.name)
                ingestion_manager.km.log_set_status(doc_id, "parsed")
                parsed_docs.append((doc_id, source_path, cache_path))
                continue

            try:
                raw_text = self.parse_document(file_path)
                cache_path.write_text(raw_text, encoding="utf-8")
                log.info("Saved parsed document to %s", cache_path)
                ingestion_manager.km.log_set_status(doc_id, "parsed")
                parsed_docs.append((doc_id, source_path, cache_path))
            except Exception as exc:
                log.error("Failed to parse %s: %s", file_path.name, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        return parsed_docs

    def _phase_extract(
            self,
            parsed_docs: list[tuple[str, str, Path]],
            ingestion_manager: IngestionManager,
    ) -> list[tuple[str, str, Path, dict]]:
        """Phase 2 – LLM entity extraction; return 4-tuples with the extraction result appended."""
        from src.ingestion.structured_extraction import extract_entities

        log.info("Phase 2: LLM entity extraction and relations")
        extracted_docs: list[tuple[str, str, Path, dict]] = []

        for doc_id, source_path, cache_path in parsed_docs:
            # Check current db state
            existing = ingestion_manager.km.log_get(source_path)
            log.info("checking extraction on: %s, i got %s", source_path, existing)
            if (existing and existing["doc_id"] == doc_id and
                    existing["status"] in ["extracted", "embedding", "indexed", "complete"]):
                log.info("Skipping LLM extraction (already extracted): %s", source_path)
                db_data = self.read_db_entries_for_embedding(doc_id, ingestion_manager.km.facts_con)
                extracted_docs.append((doc_id, source_path, cache_path, db_data))
                continue

            log.info("starting extraction")
            # if it's not extracted, do it
            ingestion_manager.km.log_set_status(doc_id, "extracting")
            try:
                raw_text = cache_path.read_text(encoding="utf-8")
                extraction_result = extract_entities(
                    source_path=source_path,
                    system_prompt=self.system_prompt,
                    source_type=self.source_type,
                    doc_id=doc_id,
                    raw_text=raw_text,
                )

                if self.use_vision_fallback and extraction_result.get("parsing_confidence") == "low":
                    log.warning("Low confidence for %s, switching to vision fallback", source_path)
                    extraction_result = self._run_vision_fallback(source_path, doc_id, extraction_result)

                ingestion_manager.km.write_document(
                    extraction_result, doc_id, source_path, self.source_type, self.write_db_entries
                )
                ingestion_manager.km.log_set_status(doc_id, "extracted")

                # Recover data from DB
                db_data = self.read_db_entries_for_embedding(doc_id, ingestion_manager.km.facts_con)
                extracted_docs.append((doc_id, source_path, cache_path, db_data))

            except RuntimeError as exc:
                log.error("LLM API error for %s. Skipping. %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))
            except Exception as exc:
                log.exception("Failed to extract/write %s: %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        return extracted_docs

    def _phase_embed(
            self,
            extracted_docs: list[tuple[str, str, Path, dict]],
            ingestion_manager: IngestionManager,
            *,
            total: int,
    ) -> list[str]:
        """Phase 3 – embedding generation and vector store population."""
        log.info("Phase 3: Embedding generation and vector store population")
        successful_doc_ids: list[str] = []

        for doc_id, source_path, cache_path, extraction_result in extracted_docs:
            existing = ingestion_manager.km.log_get(source_path)
            if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
                log.info("Skip (already complete): %s", source_path)
                successful_doc_ids.append(doc_id)
                continue

            ingestion_manager.km.log_set_status(doc_id, "embedding")
            try:
                raw_text = cache_path.read_text(encoding="utf-8")
                vector_indexer = self.make_vector_indexer(ingestion_manager, Path(source_path), raw_text)
                vector_indexer(doc_id, source_path, extraction_result)
                ingestion_manager.km.log_set_status(doc_id, "indexed")
                ingestion_manager.km.log_set_status(doc_id, "complete")
                successful_doc_ids.append(doc_id)
                log.info("Ingestion complete: %s (%s)", source_path, doc_id)
            except Exception as exc:
                log.error("Failed to embed/index %s: %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        log.info(
            "%s ingestion complete: %d/%d succeeded",
            self.source_type,
            len(successful_doc_ids),
            total,
        )
        return successful_doc_ids

    def _get_files(self) -> list[Path]:
        files = sorted(self.directory.glob(self.file_glob))
        if not files:
            log.warning("No files found in %s", self.directory)
        else:
            log.info("Found %d files to ingest for %s", len(files), self.source_type)
        return files

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _handle_file_change(
            self,
            old_doc_id: str,
            new_doc_id: str,
            source_path: str,
            ingestion_manager: IngestionManager,
    ) -> None:
        """Purge stale data and re-register under the new hash."""
        log.info("File change detected for %s. Triggering update protocol...", source_path)
        ingestion_manager.km.delete_qdrant_points(old_doc_id)
        ingestion_manager.km.cascade_delete_duckdb(old_doc_id)
        stale_cache = PARSED_DIR / f"{old_doc_id}.txt"
        if stale_cache.exists():
            stale_cache.unlink()
        ingestion_manager.km.log_upsert(new_doc_id, source_path, "parsing")

    def _run_vision_fallback(
            self,
            source_path: str,
            doc_id: str,
            extraction_result: dict,
    ) -> dict:
        """Re-run extraction via OCR when the LLM returned low confidence."""
        from src.ingestion.vision_fallback import parse_with_vision

        try:
            raw_text_vision = parse_with_vision(source_path)
        except Exception as exc:
            log.error("Vision fallback failed for %s: %s", source_path, exc)
            return extraction_result

        if not raw_text_vision or not raw_text_vision.strip():
            log.warning(
                "Vision fallback returned empty text for %s. Keeping original extraction.",
                source_path,
            )
            return extraction_result

        from src.ingestion.structured_extraction import extract_entities

        return extract_entities(
            source_path=source_path,
            system_prompt=self.system_prompt,
            source_type=self.source_type,
            doc_id=doc_id,
            raw_text=raw_text_vision,
        )
