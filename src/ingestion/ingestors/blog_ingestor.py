"""
blog_ingestion.py — Ingestion pipeline for blog HTML posts.
Implements BaseIngestor to handle metadata extraction and semantic chunking.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import duckdb

from ingestion.ingestors.base_ingestor import BaseIngestor
from app.config import CHUNK_MAX_CHAR_BLOG, COLLECTION_BLOG, KB_DIR, OPENAI_EMBEDDER_API_KEY, \
    OPENAI_EMBEDDER_BASE_URL
from ingestion.ingestion_manager import IngestionManager
from utils.ingestion_utils import build_payload, normalize_whitespace, read_text_fallback, sectionize_html

log = logging.getLogger(__name__)

# TODO: configurable
BLOG_DIR: Path = KB_DIR / "blogpost"

# TODO: extract to config
# TODO: Replace manual JSON schema declaration with a Pydantic model.
SYSTEM_PROMPT = """You are a structured entity extractor for a series of fictional culinary blog posts.

Read the blog post text and extract the relevant narrative information. Return a JSON object with this EXACT schema:

{
  "restaurant": {
    "name": "<string or null if not mentioned>",
    "planet": "<string or null>",
    "chef": "<string or null>",
    "narrative_details": "<string: brief summary of anomalies, special techniques, or narrative context discussed in the blog>"
  },
  "parsing_confidence": "high" or "low",
  "parsing_issues": "<description or null>"
}

CRITICAL RULES:
- Focus on extracting qualitative and narrative details that might not be in a standard menu (e.g. "The chef uses an unlicensed quantum oven", "The restaurant is built inside a meteorite").
- If the blog does not mention a specific restaurant, set the fields to null.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
"""


def _build_datapizza_pipeline(vs, collection_name: str, chunk_max_char: int):
    """Build a datapizza-ai IngestionPipeline."""
    from datapizza.embedders import ChunkEmbedder
    from datapizza.embedders.openai import OpenAIEmbedder
    from datapizza.modules.parsers.docling import DoclingParser
    from datapizza.modules.splitters import NodeSplitter
    from datapizza.pipeline import IngestionPipeline as DatapizzaPipeline
    from app.config import EMBEDDING_MODEL

    embedder_client = OpenAIEmbedder(
        api_key=OPENAI_EMBEDDER_API_KEY,
        base_url=OPENAI_EMBEDDER_BASE_URL,
        model_name=EMBEDDING_MODEL
    )
    return DatapizzaPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=chunk_max_char),
            ChunkEmbedder(client=embedder_client),
        ],
        vector_store=vs,
        collection_name=collection_name,
    )


class BlogIngestor(BaseIngestor):
    @property
    def source_type(self) -> str:
        return "blog"

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def collection_name(self) -> str:
        return COLLECTION_BLOG

    @property
    def chunk_max_char(self) -> int:
        return CHUNK_MAX_CHAR_BLOG

    @property
    def directory(self) -> Path:
        return BLOG_DIR

    @property
    def file_glob(self) -> str:
        return "*.html"

    @property
    def use_vision_fallback(self) -> bool:
        return False

    def parse_document(self, source_path: Path) -> str:
        """
        HTML blogs require specific parsing. We use sectionize_html to extract structured text, falling back to raw reading.
        """
        try:
            html_text = read_text_fallback(source_path)
            sections = sectionize_html(html_text)
            if sections:
                # Combine sections into a single text for LLM and chunking
                text = "\n\n".join([f"## {title}\n{text}" for title, text in sections])
                if text.strip():
                    return normalize_whitespace(text)
        except Exception as exc:
            log.warning("HTML sectionizing failed for %s: %s", source_path, exc)

        # Fallback to base parser (Docling) if HTML fails
        raw_text = super().parse_document(source_path)
        return normalize_whitespace(raw_text)

    def write_db_entries(self, extraction_result: dict[str, Any], doc_id: str,
                         facts_con: duckdb.DuckDBPyConnection) -> None:
        """
        Blogs don't have a dedicated DuckDB table for facts (other than the documents table).
        We use them primarily to enrich chunk metadata in Qdrant.
        However, we can log any matches with known restaurants for debugging.
        """
        rest_data = extraction_result.get("restaurant", {})
        if rest_data and (rest_data.get("name") or "unknown"):
            log.info("Blog %s references restaurant: %s", doc_id[:8], (rest_data.get("name") or ""))
        # TODO: No relational table writes for blogs in this MVP.
        pass

    def make_vector_indexer(self, ingestion_manager: IngestionManager, source_path: Path, raw_text: str) -> Callable[
        [str, str, dict], None]:
        """Return a vector_indexer that chunks raw_text and upserts into Qdrant."""

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
            tmp.write(raw_text)
            tmp_path = tmp.name

        def _index(doc_id: str, source_path: str, extraction_result: dict) -> None:
            try:
                vs = ingestion_manager.km.get_qdrant_vectorstore()
                datapizza_pipeline = _build_datapizza_pipeline(vs, self.collection_name, self.chunk_max_char)

                # Extract the restaurant name to enrich the chunk metadata
                restaurant_name = extraction_result.get("restaurant", {}).get("name", "unknown")

                datapizza_pipeline.run(
                    file_path=tmp_path,
                    metadata=build_payload(
                        doc_id=doc_id,
                        source_path=source_path,
                        source_type=self.source_type,
                        text="",
                        restaurant=restaurant_name,  # Enrich Qdrant payload
                    ),
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return _index

    def read_db_entries_for_embedding(self, doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
        return {}
