"""
structured_extraction.py — LLM-based entity extraction from parsed documents.

Dynamically selects the appropriate extraction prompt and transformation strategy
based on source_type ('menu', 'codice', 'manual', 'blog').
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

# TODO: Externalise to config.
_MAX_INPUT_CHARS = 12_000
_MAX_PLAUSIBLE_GRAMS = 100_000.0  # 100 kg — clearly wrong if exceeded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(
        source_path: str,
        source_type: str,
        doc_id: str,
        system_prompt: str,
        raw_text: Optional[str] = None,
        llm_client=None,
) -> dict[str, Any]:
    """
    Extract structured entities from a document using an LLM.

    Args:
        system_prompt: System prompt to ba used during document extraction.
        source_path: Path to the source document.
        source_type: One of 'menu', 'manual', 'codice', 'blog'.
        doc_id: SHA-256 hash of the document content.
        raw_text: Pre-parsed text; parsed from file when omitted.
        llm_client: datapizza OpenAI client; created from env when omitted.

    Returns:
        Extraction result dict with dishes, restaurant, parsing_confidence,
        and parsing_issues.
    """
    from src.app.config import LLM_MODEL

    if raw_text is None:
        raw_text = _parse_document(source_path, source_type)

    if llm_client is None:
        llm_client = _get_llm_client(LLM_MODEL)

    user_prompt = _build_extraction_prompt(raw_text, source_type)

    try:
        response = llm_client.invoke(
            input=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        result = _parse_json_response(response.text)
    except Exception as exc:
        log.error("LLM extraction failed for %s: %s", source_path, exc)
        result = {
            "parsing_confidence": "low",
            "parsing_issues": str(exc),
            source_type: []  # ensure a list exists for the expected entities
        }

    # Post-process quantities for menus, it's a special case
    if source_type == "menu":
        result = _postprocess_quantities(result)

    log.info(
        "Extraction complete for %s [confidence=%s]",
        source_path,
        result.get("parsing_confidence", "unknown"),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_document(source_path: str, source_type: str) -> str:
    from datapizza.modules.parsers.docling import DoclingParser
    parser = DoclingParser()
    try:
        nodes = parser([source_path])
        return "\n\n".join(getattr(n, "text", n.content) for n in nodes if getattr(n, "text", n.content))
    except Exception as exc:
        log.warning("DoclingParser failed for %s: %s — falling back.", source_path, exc)
        return Path(source_path).read_text(encoding="utf-8", errors="ignore")


def _get_llm_client(model: str):
    from datapizza.clients.openai import OpenAIClient
    from src.app.config import OPENAI_API_KEY, OPENAI_BASE_URL
    return OpenAIClient(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model=model)


def _build_extraction_prompt(raw_text: str, source_type: str) -> str:
    hint = f"This is a {source_type} document. Extract all relevant entities according to the requested schema."
    if len(raw_text) > _MAX_INPUT_CHARS:
        log.warning("Truncating document text to %d chars", _MAX_INPUT_CHARS)
    return f"{hint}\n\nDOCUMENT TEXT:\n{raw_text[:_MAX_INPUT_CHARS]}"


def _parse_json_response(text: str) -> dict[str, Any]:
    #TODO: use json_repair library
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _postprocess_quantities(result: dict[str, Any]) -> dict[str, Any]:
    #TODO: generalize to all quantities
    for dish in result.get("dishes", []):
        for ingredient in dish.get("ingredients", []):
            qty = ingredient.get("quantity_grams")
            if qty is None: continue
            try:
                qty = float(qty)
            except (TypeError, ValueError):
                ingredient["quantity_grams"] = None
                continue
            if math.isclose(qty, 0.0, abs_tol=1e-9) or qty < 0 or qty > _MAX_PLAUSIBLE_GRAMS:
                ingredient["quantity_grams"] = None
            else:
                ingredient["quantity_grams"] = qty
    return result