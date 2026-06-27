"""
base_ingestor.py — Abstract base class for document-specific ingestors.

Defines the interface for handling specific data types, ensuring consistency
across different ingestion pipelines (menu, galactic code, etc.).
"""

from __future__ import annotations

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
        return "*.pdf"

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
        """Ingest all files in three separate batched phases using file cache for parsed text."""
        import hashlib

        def _sha256_file(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        files = sorted(self.directory.glob(self.file_glob))
        if not files:
            log.warning("No files found in %s", self.directory)
            return []

        log.info("Found %d files to ingest for %s", len(files), self.source_type)

        # Ensure the cache directory exists
        PARSED_DIR.mkdir(parents=True, exist_ok=True)

        # List of documents ready for extraction (with already parsed text and in cache)
        parsed_docs = []

        # Phase 1: Parse all documents
        log.info("Phase 1: Parse all documents")
        for file_path in files:
            doc_id = _sha256_file(file_path)
            source_path = str(file_path)

            existing = ingestion_manager.km.log_get(source_path)
            if existing and existing["doc_id"] == doc_id and existing["status"] == "complete":
                log.info("Skip (already complete): %s", file_path.name)
                continue

            parsed_file_path = PARSED_DIR / f"{doc_id}.txt"

            # If the cache file already exists, read it directly skipping Docling
            if parsed_file_path.exists():
                log.info("Loading cached parsed text for %s", file_path.name)
                ingestion_manager.km.log_upsert(doc_id, source_path, "parsed")
                parsed_docs.append((doc_id, source_path, parsed_file_path))
            else:
                ingestion_manager.km.log_upsert(doc_id, source_path, "parsing")
                try:
                    raw_text = self.parse_document(file_path)
                    # Save parsed text to file for future runs
                    # TODO: remove if changed
                    with open(parsed_file_path, "w", encoding="utf-8") as f:
                        f.write(raw_text)
                        log.info(f"Saved parsed document to {parsed_file_path}")

                    ingestion_manager.km.log_set_status(doc_id, "parsed")
                    parsed_docs.append((doc_id, source_path, parsed_file_path))
                except Exception as exc:
                    log.error("Failed to parse %s: %s", file_path.name, exc)
                    ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        log.info("Phase 2: LLM Entity extraction and relations")
        from src.ingestion.structured_extraction import extract_entities

        extracted_docs = []

        for doc_id, source_path, parsed_file_path in parsed_docs:
            ingestion_manager.km.log_set_status(doc_id, "extracting")
            try:
                # Read text from cache file
                raw_text = parsed_file_path.read_text(encoding="utf-8")

                extraction_result = extract_entities(
                    source_path=source_path,
                    system_prompt=self.system_prompt,
                    source_type=self.source_type,
                    doc_id=doc_id,
                    raw_text=raw_text,
                )

                # Vision Fallback only if LLM responded but gave low confidence
                if self.use_vision_fallback and extraction_result.get("parsing_confidence") == "low":
                    log.warning("Low confidence for %s, switching to vision fallback", source_path)
                    try:
                        from src.ingestion.vision_fallback import parse_with_vision
                        raw_text_vision = parse_with_vision(source_path)

                        if raw_text_vision and raw_text_vision.strip():
                            # Ritenta l'estrazione con il testo OCR
                            extraction_result = extract_entities(
                                source_path=source_path,
                                system_prompt=self.system_prompt,
                                source_type=self.source_type,
                                doc_id=doc_id,
                                raw_text=raw_text_vision,
                            )
                        else:
                            log.warning("Vision fallback returned empty text for %s. Keeping original extraction.",
                                        source_path)
                    except Exception as vision_exc:
                        log.error("Vision fallback failed for %s: %s", source_path, vision_exc)

                ingestion_manager.km.write_document(
                    extraction_result,
                    doc_id,
                    source_path,
                    self.source_type,
                    self.write_db_entries
                )
                ingestion_manager.km.log_set_status(doc_id, "extracted")
                extracted_docs.append((doc_id, source_path, parsed_file_path, extraction_result))

            except RuntimeError as exc:
                # LLM API error (e.g., model not loaded). Skipping document.
                log.error("LLM API Error for %s. Skipping document. %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))
            except Exception as exc:
                log.error("Failed to extract/write %s: %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        # Phase 3: Embedding generation and vector store population
        log.info("Phase 3: Embedding generation and vector store population")
        successful_doc_ids = []

        for doc_id, source_path, parsed_file_path, extraction_result in extracted_docs:
            ingestion_manager.km.log_set_status(doc_id, "embedding")
            try:
                # Read the file again from the cache
                raw_text = parsed_file_path.read_text(encoding="utf-8")

                vector_indexer = self.make_vector_indexer(ingestion_manager, Path(source_path), raw_text)
                vector_indexer(doc_id, source_path, extraction_result)

                ingestion_manager.km.log_set_status(doc_id, "indexed")
                ingestion_manager.km.log_set_status(doc_id, "complete")
                successful_doc_ids.append(doc_id)
                log.info("Ingestion complete: %s (%s)", source_path, doc_id)
            except Exception as exc:
                log.error("Failed to embed/index %s: %s", source_path, exc)
                ingestion_manager.km.log_set_status(doc_id, "failed", str(exc))

        log.info("%s ingestion complete: %d/%d succeeded", self.source_type, len(successful_doc_ids), len(files))
        return successful_doc_ids
