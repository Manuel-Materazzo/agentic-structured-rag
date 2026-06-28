"""
shared.py - Common helpers for ingestion modules.

Provides lightweight, source-agnostic helpers for:
- text extraction / chunking
- Qdrant payload normalization
- collection routing
- safe parsing fallbacks
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass
class IngestionChunk:
    text: str
    metadata: dict


def read_text_fallback(path: Path) -> str:
    """Read a file as text with robust fallbacks."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split text into roughly max_chars chunks on paragraph boundaries."""
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            for start in range(0, len(paragraph), max_chars):
                pieces = paragraph[start:start + max_chars].strip()
                if pieces:
                    chunks.append(pieces)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def sectionize_html(html_text: str) -> list[tuple[str, str]]:
    """
    Extract a simple hierarchy from HTML text using headings.

    Returns pairs of (section_title, section_text).
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return [("HTML", normalize_whitespace(html_text))]

    soup = BeautifulSoup(html_text, "html.parser")
    blocks: list[tuple[str, str]] = []
    current_title = "HTML"
    current_lines: list[str] = []

    def flush():
        nonlocal current_lines
        body = normalize_whitespace("\n".join(current_lines))
        if body:
            blocks.append((current_title, body))
        current_lines = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if element.name in {"h1", "h2", "h3"}:
            flush()
            current_title = normalize_whitespace(element.get_text(" ", strip=True)) or "HTML"
        else:
            current_lines.append(element.get_text(" ", strip=True))
    flush()
    return blocks or [("HTML", normalize_whitespace(soup.get_text(" ", strip=True)))]


def build_payload(
    *,
    doc_id: str,
    source_path: str,
    source_type: str,
    section: str | None = None,
    restaurant: str | None = None,
    dish: str | None = None,
    page: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Build the standardized Qdrant payload contract."""
    payload = {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "source_path": source_path,
        "source_type": source_type,
        "page": page,
        "section": section,
        "restaurant": restaurant,
        "dish": dish,
    }
    if extra:
        payload.update(extra)
    return payload


def delete_qdrant_points_by_doc_id(client, collection_names: list[str], doc_id: str) -> None:
    """Delete all Qdrant points matching a doc_id across the given collections."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    selector = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    for collection_name in collection_names:
        client.delete(
            collection_name=collection_name,
            points_selector=selector,
        )


def chunk_records(
    records: Iterable[tuple[str, dict]],
    max_chars: int,
) -> list[IngestionChunk]:
    """Split labeled records into chunks while preserving the label metadata."""
    out: list[IngestionChunk] = []
    for label, metadata in records:
        text = metadata.pop("text", "")
        for chunk in split_text(text, max_chars=max_chars):
            out.append(IngestionChunk(text=chunk, metadata={"section": label, **metadata}))
    return out
