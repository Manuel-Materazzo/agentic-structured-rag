"""
lookup_tools.py — Deterministic DuckDB helpers exposed to the orchestrator.

These helpers keep the SQL layer small, explicit, and easy to test. They do not
interpret the answer; they only return raw rows and schema metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb


@dataclass(frozen=True)
class QueryResult:
    """Container for a raw SQL result."""

    columns: list[str]
    rows: list[tuple[Any, ...]]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


def get_schema_overview(con: duckdb.DuckDBPyConnection) -> str:
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    lines = []
    for table in sorted(tables):
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_defs = ", ".join(f"{row[0]} {row[1]}" for row in cols)
        lines.append(f"{table}({col_defs})")
    return "\n".join(lines)


def run_sql(
        con: duckdb.DuckDBPyConnection,
        sql: str,
        params: list[Any] | None = None,
) -> QueryResult:
    """Execute SQL and return the raw rows and column names."""
    cur = con.execute(sql, params or [])
    columns = [desc[0] for desc in cur.description or []]
    rows = cur.fetchall()
    return QueryResult(columns=columns, rows=rows)
