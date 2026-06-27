"""
sql_agent.py — LLM-driven SQL execution layer with bounded retry support.

The agent asks the LLM to translate natural-language requests into SQL, then
executes the query against DuckDB and returns raw rows with column names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from ingestion.knowledge_manager import KnowledgeManager
from ingestion.structured_extraction import parse_json_response
from src.app.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_BASE_URL
from src.utils.sql_utils import get_schema_overview, run_sql

log = logging.getLogger(__name__)
# TODO: specify pydantic model
# TODO: extract config
# TODO: better prompting
# TODO: separate question entity extraction from SQL generation to simplify inference
SYSTEM_PROMPT = """You are a precise SQL translator for a DuckDB database.
Your responsibility is to receive a request in natural language, translate it into a correct DuckDB SQL query, and return it.

CRITICAL RULES:
- Use ONLY tables and columns from the provided SCHEMA.
- Do not interpret or filter the results manually: just return the raw SQL.
- Take a deep breath and think step by step before responding, but keep the rationale short and commit to a single interpretation. 
- Do not second-guess yourself mid-rationale. long rationales with multiple hypotheses produce inconsistent queries.
- Return a valid JSON object with this exact schema: {"sql": "SELECT ...", "rationale": "brief explanation"}
- If the request is ambiguous, still produce the best single SQL query you can infer.
- If the request asks for a list of dishes, return only the dish names, not other details.
- Make sure to escape single quotes by doubling them: 'L''aquila', never with backslash.

FUZZY MATCHING:
- Names in the DB may differ slightly from how they appear in the user's request (typos, gender, accents, truncations).
- NEVER filter with exact equality (=) on free-text fields like technique, ingredient, or dish name.
- Always use jaro_winkler_similarity(field, 'value') > 0.93 for matching these fields.
- Always use LOWERCASE('value') for TEXT comparations.
- Use EXISTS / NOT EXISTS subqueries with jaro_winkler_similarity, never IN / NOT IN with exact strings.
- Example of correct technique filter:
    EXISTS (
      SELECT 1 FROM dish_techniques dt
      WHERE dt.dish_id = d.id
      AND jaro_winkler_similarity(dt.technique, LOWER('Fermentazione Quantico Biometrica')) > 0.93
    )

QUERY STRUCTURE RULES:
- When filtering a parent entity by properties of its child rows, NEVER JOIN the child table
  in the outer query. Use EXISTS / NOT EXISTS subqueries only.
- This rule applies unconditionally: whether the check is positive (has technique X),
  negative (does not have technique X), or both. It applies to every parent-child
  relationship in the schema (dishes→techniques, dishes→ingredients, restaurants→dishes, etc.)
- Why: a JOIN filters rows, not entities. If a dish has 10 technique rows, joining produces
  10 output rows — or 0 if the WHERE excludes them all. Multiple JOINs on the same child
  table create a cartesian product and break the logic entirely.
- Correct pattern — "entities that have A and B but not C":
    SELECT e.*
    FROM entity e
    WHERE EXISTS     (SELECT 1 FROM child c WHERE c.entity_id = e.id AND c.property = 'A')
    AND   EXISTS     (SELECT 1 FROM child c WHERE c.entity_id = e.id AND c.property = 'B')
    AND   NOT EXISTS (SELECT 1 FROM child c WHERE c.entity_id = e.id AND c.property = 'C')
- WRONG — even if there is no NOT EXISTS:
    SELECT e.*
    FROM entity e
    JOIN child c1 ON c1.entity_id = e.id  -- cartesian product
    JOIN child c2 ON c2.entity_id = e.id  -- cartesian product
    WHERE c1.property = 'A'
    AND c2.property = 'B'
"""


@dataclass
class SQLAgentResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    sql: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "error": self.error,
        }


class SQLAgent:
    """LLM-driven SQL agent with bounded retry support and auto-correction."""

    def __init__(
            self,
            knowledge_manager: KnowledgeManager | None = None,
            llm_client=None,
    ):
        self._connection = knowledge_manager.get_facts_db()
        self._llm_client = llm_client or self._get_llm_client()
        # Preload DB schema
        self._schema = get_schema_overview(self._connection)

    def execute(self, request: str, max_retries: int = 2) -> SQLAgentResult:
        # Build the initial prompt with schema and request
        user_prompt = f"SCHEMA:\n{self._schema}\n\nREQUEST:\n{request}\n"

        attempts = 0
        last_error: str | None = None
        current_prompt = user_prompt

        while attempts <= max_retries:
            attempts += 1
            try:
                response = self._llm_client.invoke(
                    input=current_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0,
                )
                sql = self._parse_sql_from_response(response.text)

                # Execute DB query
                # TODO: abstract complexities from LLM by post-editing query
                # example: add jaro_winkler_similarity programmatically
                result = run_sql(self._connection, sql, [])
                log.info(f"SQL Executed: {result}")
                return SQLAgentResult(
                    columns=result.columns,
                    rows=result.rows,
                    sql=sql
                )

            except Exception as exc:
                last_error = str(exc)
                log.warning("SQL attempt %s failed: %s", attempts, exc)
                if attempts > max_retries:
                    break

                # AUTO-CORRECTION: tell the LLM where it went wrong
                current_prompt = (
                    f"{user_prompt}\n"
                    f"Your previous SQL attempt failed with this error:\n"
                    f"---\n{last_error}\n---\n"
                    f"Please correct the SQL query and return the JSON again."
                )

        return SQLAgentResult(
            columns=[],
            rows=[],
            error=last_error,
            sql="max retries exceeded",
        )

    def _get_llm_client(self):
        from datapizza.clients.openai_like import OpenAILikeClient

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the SQL agent. "
                "Pass llm_client explicitly in tests or set the environment variable."
            )
        return OpenAILikeClient(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    def _parse_sql_from_response(self, response_text: str) -> str:
        text = response_text.strip()
        try:
            # repair json
            parsed = parse_json_response(text)
            if not parsed:
                raise json.JSONDecodeError("Json repair returned an invalid object.", text, 0)
            # extract data
            sql = parsed.get("sql", "")
            rationale = parsed.get('rationale', '')

            # log sql and rationale
            log.info(f"SQL Agent generated SQL: \"{sql}\" with the rationale: \"{rationale}\"")

            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("LLM response did not include a valid 'sql' field")

            return sql.strip().rstrip(";")
        except Exception:
            log.warning(f"SQL Agent ran into an exception while parsing the LLM response. Raw response text: \"{text}\"")
            # Emergency fallback: try to find the SQL in the string
            upper = text.upper()
            start = upper.find("SELECT")
            if start == -1:
                raise ValueError("Could not parse SQL from LLM response")
            return text[start:].strip().rstrip(";")
