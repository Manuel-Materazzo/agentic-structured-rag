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

    # Load all schema_meta rows once
    meta_rows = con.execute("SELECT table_name, column_name, visible, description FROM schema_meta").fetchall()

    table_meta = {}  # table_name -> (visible, description)
    column_meta = {}  # (table_name, column_name) -> (visible, description)
    for table_name, column_name, visible, description in meta_rows:
        if column_name == '':
            table_meta[table_name] = (visible, description)
        else:
            column_meta[(table_name, column_name)] = (visible, description)

    lines = []
    for table in sorted(tables):
        if table == "schema_meta":
            continue

        t_visible, t_desc = table_meta.get(table, (True, None))
        if not t_visible:
            continue

        cols = con.execute(f"DESCRIBE {table}").fetchall()
        col_parts = []
        for row in cols:
            col_name, col_type = row[0], row[1]
            c_visible, c_desc = column_meta.get((table, col_name), (True, None))
            if not c_visible:
                continue
            entry = f"\n{col_name} {col_type}"
            if c_desc:
                entry += f" -- {c_desc}"
            col_parts.append(entry)

        if not col_parts:
            continue

        line = f"{table}({', '.join(col_parts)}\n)"
        if t_desc:
            line += f"  -- {t_desc}"
        lines.append(line)

    return "\n\n".join(lines)


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
