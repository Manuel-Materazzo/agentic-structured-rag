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
        """Ingest all files matching the glob in the specific directory."""
        files = sorted(self.directory.glob(self.file_glob))
        if not files:
            log.warning("No files found in %s", self.directory)
            return []

        log.info("Found %d files to ingest for %s", len(files), self.source_type)
        doc_ids: list[str] = []

        for file_path in files:
            try:
                doc_ids.append(self.ingest_single(ingestion_manager, file_path))
            except Exception as exc:
                log.error("Failed to ingest %s: %s", file_path.name, exc)

        log.info("%s ingestion complete: %d/%d succeeded", self.source_type, len(doc_ids), len(files))
        return doc_ids
