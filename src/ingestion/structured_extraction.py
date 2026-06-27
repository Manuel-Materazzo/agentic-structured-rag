"""
structured_extraction.py — LLM-based entity extraction from parsed documents.

Dynamically selects the appropriate extraction prompt and DuckDB write strategy
based on source_type ('menu', 'codice', 'manual', 'blog').
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------
# TODO: Externalise these prompts to a config file.
# TODO: Replace manual JSON schema declaration with a Pydantic model.

EXTRACTION_SYSTEM_PROMPTS = {
    "menu": """You are a structured entity extractor for a restaurant menu database.

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
""",
    "codice": """You are a structured entity extractor for a regulatory compliance document (Codice Galattico).

Extract ALL compliance rules regarding ingredient limits or technique license requirements. Return a JSON object with this EXACT schema:

{
  "compliance_rules": [
    {
      "rule_type": "ingredient_limit" or "technique_license",
      "subject": "<name of ingredient or technique>",
      "constraint_value": "<max quantity (e.g. '500g') or required license level (e.g. 'Level B')>",
      "scope": "<context of the rule, e.g. 'per dish' or null>"
    }
  ],
  "parsing_confidence": "high" or "low",
  "parsing_issues": "<description or null>"
}

CRITICAL RULES:
- rule_type: must be exactly 'ingredient_limit' or 'technique_license'.
- constraint_value: string representing the limit or the required license.
- Output ONLY the JSON object. No additional text, markdown, or explanation.
""",
    "manual": "Extract technique taxonomy, macro categories, and required license levels.",
    "blog": "Extract relevant dish and restaurant details mentioned in the blog post."
}

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
        raw_text: Optional[str] = None,
        llm_client=None,
) -> dict[str, Any]:
    """
    Extract structured entities from a document using an LLM.

    Args:
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

    system_prompt = EXTRACTION_SYSTEM_PROMPTS.get(
        source_type, "Extract all relevant structured entities as JSON."
    )
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

    if source_type == "menu":
        result = _postprocess_quantities(result)

    log.info(
        "Extraction complete for %s [confidence=%s]",
        source_path,
        result.get("parsing_confidence", "unknown"),
    )
    return result


def write_to_duckdb(
        extraction_result: dict[str, Any],
        doc_id: str,
        source_path: str,
        source_type: str,
        facts_con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Persist extraction results to facts.db (idempotent).
    Routes to the correct writer based on source_type.
    """
    now = datetime.now(timezone.utc)

    facts_con.execute(
        """
        INSERT INTO documents (doc_id, source_path, source_type, ingested_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (doc_id) DO NOTHING
        """,
        [doc_id, source_path, source_type, now],
    )

    if source_type == "menu":
        _write_menu_entities(extraction_result, doc_id, facts_con)
    elif source_type == "codice":
        _write_compliance_rules(extraction_result, doc_id, facts_con)
    else:
        log.warning("No specific DuckDB writer implemented for source_type: %s", source_type)


# ---------------------------------------------------------------------------
# Private writers
# ---------------------------------------------------------------------------

def _write_menu_entities(extraction_result: dict[str, Any], doc_id: str, facts_con: duckdb.DuckDBPyConnection) -> None:
    rest_data = extraction_result.get("restaurant", {})
    rest_name = rest_data.get("name") or "Unknown"

    existing_rest = facts_con.execute(
        "SELECT id FROM restaurants WHERE doc_id = ? AND name = ?",
        [doc_id, rest_name],
    ).fetchone()

    if existing_rest:
        rest_id = existing_rest[0]
    else:
        rest_id = facts_con.execute("SELECT nextval('restaurants_id_seq')").fetchone()[0]
        facts_con.execute(
            """
            INSERT INTO restaurants
                (id, name, chef, planet, chef_license, professional_orders, doc_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                rest_id, rest_name, rest_data.get("chef"), rest_data.get("planet"),
                rest_data.get("chef_license"), rest_data.get("professional_orders", []), doc_id,
            ],
        )

    for dish in extraction_result.get("dishes", []):
        dish_name = (dish.get("name") or "").strip()
        if not dish_name:
            continue

        existing_dish = facts_con.execute(
            "SELECT id FROM dishes WHERE name = ? AND restaurant_id = ?",
            [dish_name, rest_id],
        ).fetchone()

        if existing_dish:
            dish_id = existing_dish[0]
        else:
            dish_id = facts_con.execute("SELECT nextval('dishes_id_seq')").fetchone()[0]
            facts_con.execute(
                """
                INSERT INTO dishes (id, name, restaurant_id, preparation_notes, doc_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                [dish_id, dish_name, rest_id, dish.get("preparation_notes"), doc_id],
            )

        for ing in dish.get("ingredients", []):
            ing_name = (ing.get("name") or "").strip()
            if not ing_name: continue
            try:
                facts_con.execute(
                    """
                    INSERT INTO dish_ingredients
                        (dish_id, ingredient, quantity_grams, quantity_raw, is_regulated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (dish_id, ingredient) DO NOTHING
                    """,
                    [dish_id, ing_name, ing.get("quantity_grams"), ing.get("quantity_raw", ""), False],
                )
            except Exception as exc:
                log.warning("Failed to insert ingredient '%s': %s", ing_name, exc)

        for tech in dish.get("techniques", []):
            tech_name = tech.strip() if isinstance(tech, str) else str(tech)
            if not tech_name: continue
            try:
                facts_con.execute(
                    """
                    INSERT INTO dish_techniques (dish_id, technique)
                    VALUES (?, ?)
                    ON CONFLICT (dish_id, technique) DO NOTHING
                    """,
                    [dish_id, tech_name],
                )
            except Exception as exc:
                log.warning("Failed to insert technique '%s': %s", tech_name, exc)


def _write_compliance_rules(extraction_result: dict[str, Any], doc_id: str,
                            facts_con: duckdb.DuckDBPyConnection) -> None:
    rules = extraction_result.get("compliance_rules", [])
    for rule in rules:
        rule_type = rule.get("rule_type")
        subject = rule.get("subject")
        constraint_value = rule.get("constraint_value")
        scope = rule.get("scope")

        if not rule_type or not subject or not constraint_value:
            log.warning("Skipping incomplete compliance rule: %s", rule)
            continue

        rule_id = facts_con.execute("SELECT nextval('compliance_rules_id_seq')").fetchone()[0]
        facts_con.execute(
            """
            INSERT INTO compliance_rules
                (id, rule_type, subject, constraint_value, scope, doc_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [rule_id, rule_type, subject, constraint_value, scope, doc_id],
        )
    log.info("Wrote %d compliance rules for doc %s…", len(rules), doc_id[:8])


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
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _postprocess_quantities(result: dict[str, Any]) -> dict[str, Any]:
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
