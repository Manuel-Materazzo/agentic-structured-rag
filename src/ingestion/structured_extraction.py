"""
structured_extraction.py — LLM-based entity extraction from parsed documents.

Extracts per dish:
  name, restaurant, ingredients[] (quantity_grams FLOAT|null, quantity_raw TEXT),
  techniques[], preparation_notes, chef, planet, chef_license, professional_orders[]

Always returns parsing_confidence ("high"/"low") and parsing_issues.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import duckdb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
# TODO: Externalize this prompts to a config file
# TODO: replace manual JSON declaration with pydantic model
EXTRACTION_SYSTEM_PROMPT = """You are a structured entity extractor for a restaurant menu database.

Extract ALL dishes from the provided menu text and return a JSON object with this EXACT schema:

{
  "restaurant": {
    "name": "<string>",
    "chef": "<string or null>",
    "planet": "<string or null>",
    "chef_license": "<string or null>",
    "professional_orders": ["<order1>", ...]
  },
  "dishes": [
    {
      "name": "<dish name>",
      "ingredients": [
        {
          "name": "<ingredient name>",
          "quantity_grams": <float or null>,
          "quantity_raw": "<original quantity text>"
        }
      ],
      "techniques": ["<technique1>", "<technique2>"],
      "preparation_notes": "<text or null>"
    }
  ],
  "parsing_confidence": "high" or "low",
  "parsing_issues": "<description or null>"
}

CRITICAL RULES:
- quantity_grams: FLOAT with decimal point (e.g. 200.0) or null. NEVER use 0.0 for "as much as needed", "to taste", "traces", or unquantifiable amounts.
- quantity_raw: ALWAYS populate with the original text (e.g. "200g", "as much as needed", "3 leaves").
- parsing_confidence: Use "low" ONLY if you could NOT extract coherent entity structure. NOT just because the text was messy.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
"""


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_entities(
    source_path: str,
    source_type: str,
    doc_id: str,
    raw_text: Optional[str] = None,
    llm_client=None,
) -> dict[str, Any]:
    """
    Extract structured entities from a document using an LLM.

    Args:
        source_path: Path to the source document
        source_type: Type of document ('menu', 'manual', 'codice', 'blog')
        doc_id: SHA-256 hash of the document content
        raw_text: Pre-parsed text (if None, will parse from file)
        llm_client: datapizza OpenAI client (if None, creates one from env)
        llm_model: LLM model to use

    Returns:
        Extraction result dict with dishes, restaurant, parsing_confidence, parsing_issues
    """
    from src.app.config import LLM_MODEL

    # Parse document if text not provided
    if raw_text is None:
        raw_text = _parse_document(source_path, source_type)

    # Get LLM client
    if llm_client is None:
        llm_client = _get_llm_client(LLM_MODEL)

    # Build prompt based on source type
    user_prompt = _build_extraction_prompt(raw_text, source_type)

    # Call LLM
    try:
        response = llm_client.invoke(
            input=user_prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
        )
        response_text = response.text
        result = _parse_json_response(response_text)
    except Exception as exc:
        log.error(f"LLM extraction failed for {source_path}: {exc}")
        result = {
            "restaurant": {"name": "unknown", "chef": None, "planet": None,
                          "chef_license": None, "professional_orders": []},
            "dishes": [],
            "parsing_confidence": "low",
            "parsing_issues": str(exc),
        }

    # Post-processing: range check on quantity_grams
    # TODO: better validation with a pydantic model
    result = _postprocess_quantities(result)

    log.info(
        f"Extracted {len(result.get('dishes', []))} dishes from {source_path} "
        f"[confidence={result.get('parsing_confidence', 'unknown')}]"
    )
    return result


def _parse_document(source_path: str, source_type: str) -> str:
    """Parse a document to text using DoclingParser."""
    from datapizza.modules.parsers.docling import DoclingParser
    from datapizza.type import Node

    parser = DoclingParser()
    path = Path(source_path)

    try:
        nodes: list[Node] = parser([str(path)])
        return "\n\n".join(n.content for n in nodes if n.content)
    except Exception as exc:
        log.warning(f"DoclingParser failed for {source_path}: {exc}. Falling back to raw read.")
        return path.read_text(encoding="utf-8", errors="ignore")


def _get_llm_client(model: str):
    """Create an OpenAI client using environment variables."""
    import os
    from datapizza.clients.openai import OpenAIClient
    # TODO: get from config
    return OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY", ""), model=model)


def _build_extraction_prompt(raw_text: str, source_type: str) -> str:
    """Build the user prompt for entity extraction."""
    # TODO: Externalize these prompts to a config file
    type_hints = {
        "menu": "This is a restaurant menu PDF. Extract all dishes with their ingredients and techniques.",
        "manual": "This is a culinary manual. Extract technique taxonomy, license levels, and professional orders.",
        "codice": "This is a regulatory document (Codice Galattico). Extract compliance rules: ingredient quantity limits and technique license requirements.",
        "blog": "This is a blog post about a restaurant. Extract relevant dish and restaurant details.",
    }
    hint = type_hints.get(source_type, "Extract all relevant entities.")
    # TODO: Add document split handling
    if len(raw_text) > 12000:
        log.warning(f"Truncating document text to 12000 chars for {source_type}")
    return f"{hint}\n\nDOCUMENT TEXT:\n{raw_text[:12000]}"  # truncate to avoid token limits


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse the LLM JSON response, stripping markdown fences if present."""
    # TODO: use json_repair lib for advanced broken JSON parsing
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(text)


def _postprocess_quantities(result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and fix quantity_grams values:
    - Must be FLOAT or null
    - 0.0 for unquantifiable amounts → null
    - Implausibly large values → null with warning
    """
    MAX_PLAUSIBLE_GRAMS = 100_000.0  # 100kg, clearly wrong if exceeded

    # TODO: business logic probably simplifiable
    for dish in result.get("dishes", []):
        for ingredient in dish.get("ingredients", []):
            qty = ingredient.get("quantity_grams")
            if qty is not None:
                try:
                    qty = float(qty)
                except (TypeError, ValueError):
                    log.warning(
                        f"Non-numeric quantity_grams '{qty}' for ingredient "
                        f"'{ingredient.get('name')}' → forcing to null"
                    )
                    ingredient["quantity_grams"] = None
                    continue

                # Floating point equality is bad
                if math.isclose(qty, 0.0, rel_tol=1e-09, abs_tol=1e-09):
                    # 0.0 is ambiguous, treat as unquantifiable
                    ingredient["quantity_grams"] = None
                elif qty < 0 or qty > MAX_PLAUSIBLE_GRAMS:
                    log.warning(
                        f"Out-of-range quantity_grams {qty} for ingredient "
                        f"'{ingredient.get('name')}' → forcing to null"
                    )
                    ingredient["quantity_grams"] = None
                else:
                    ingredient["quantity_grams"] = qty

    return result


# ---------------------------------------------------------------------------
# DuckDB write helper
# ---------------------------------------------------------------------------

def write_to_duckdb(
    extraction_result: dict[str, Any],
    doc_id: str,
    source_path: str,
    source_type: str,
    facts_con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Persist extraction results to facts.db.
    Inserts into: documents, restaurants, dishes, dish_ingredients, dish_techniques.
    """
    # TODO: extract this and use a proper pattern
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    # documents
    facts_con.execute(
        "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?)",
        [doc_id, source_path, source_type, now],
    )

    rest_data = extraction_result.get("restaurant", {})
    rest_name = rest_data.get("name") or "Unknown"

    # restaurants, get next ID
    next_rest_id = facts_con.execute(
        "SELECT COALESCE(MAX(id), 0) + 1 FROM restaurants"
    ).fetchone()[0]

    # Check if restaurant already exists for this doc_id
    existing_rest = facts_con.execute(
        "SELECT id FROM restaurants WHERE doc_id = ? AND name = ?",
        [doc_id, rest_name],
    ).fetchone()

    if existing_rest:
        rest_id = existing_rest[0]
    else:
        facts_con.execute(
            "INSERT INTO restaurants (id, name, chef, planet, chef_license, professional_orders, doc_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                next_rest_id,
                rest_name,
                rest_data.get("chef"),
                rest_data.get("planet"),
                rest_data.get("chef_license"),
                rest_data.get("professional_orders", []),
                doc_id,
            ],
        )
        rest_id = next_rest_id

    # dishes
    for dish in extraction_result.get("dishes", []):
        next_dish_id = facts_con.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM dishes"
        ).fetchone()[0]

        dish_name = dish.get("name", "").strip()
        if not dish_name:
            continue

        # Skip if already inserted (idempotent)
        existing_dish = facts_con.execute(
            "SELECT id FROM dishes WHERE name = ? AND restaurant_id = ?",
            [dish_name, rest_id],
        ).fetchone()

        if existing_dish:
            dish_id = existing_dish[0]
        else:
            facts_con.execute(
                "INSERT INTO dishes (id, name, restaurant_id, preparation_notes, doc_id) "
                "VALUES (?, ?, ?, ?, ?)",
                [next_dish_id, dish_name, rest_id, dish.get("preparation_notes"), doc_id],
            )
            dish_id = next_dish_id

        # dish_ingredients
        for ing in dish.get("ingredients", []):
            ing_name = ing.get("name", "").strip()
            if not ing_name:
                continue
            try:
                facts_con.execute(
                    "INSERT OR IGNORE INTO dish_ingredients "
                    "(dish_id, ingredient, quantity_grams, quantity_raw, is_regulated) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [dish_id, ing_name, ing.get("quantity_grams"), ing.get("quantity_raw", ""), False],
                )
            except Exception as exc:
                log.warning(f"Failed to insert ingredient '{ing_name}': {exc}")

        # dish_techniques
        for tech in dish.get("techniques", []):
            tech_name = tech.strip() if isinstance(tech, str) else str(tech)
            if not tech_name:
                continue
            try:
                facts_con.execute(
                    "INSERT OR IGNORE INTO dish_techniques (dish_id, technique) VALUES (?, ?)",
                    [dish_id, tech_name],
                )
            except Exception as exc:
                log.warning(f"Failed to insert technique '{tech_name}': {exc}")

    log.info(
        f"Wrote {len(extraction_result.get('dishes', []))} dishes to DuckDB "
        f"for {source_path} ({doc_id[:8]}…)"
    )
