"""Tools the agent can call. Each tool returns a JSON-serialisable dict or raises ToolError."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

from .guardrails import UnsafeSQLError, validate_sql


class ToolError(Exception):
    """A tool failed in a way the agent should see and recover from."""


class Database:
    """Loads every CSV in a folder into an in-memory SQLite database, then locks it read-only."""

    def __init__(self, data_dir: str | Path, max_rows: int = 50, timeout_s: float = 5.0):
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.tables: dict[str, pd.DataFrame] = {}
        for csv in sorted(Path(data_dir).glob("*.csv")):
            name = re.sub(r"\W+", "_", csv.stem).lower()
            df = pd.read_csv(csv)
            df.to_sql(name, self.conn, index=False)
            self.tables[name] = df
        if not self.tables:
            raise ValueError(f"No CSV files found in {data_dir}")
        self.conn.execute("PRAGMA query_only = ON")  # layer 2: the DB itself refuses writes

    # -------------------------------------------------------------- tools
    def list_tables(self) -> dict:
        return {"tables": {n: {"rows": len(df), "columns": list(df.columns)} for n, df in self.tables.items()}}

    def describe_table(self, table: str) -> dict:
        if table not in self.tables:
            raise ToolError(f"Unknown table '{table}'. Available tables: {sorted(self.tables)}")
        df = self.tables[table]
        columns = []
        for c in df.columns:
            col = {"name": c, "dtype": str(df[c].dtype), "nulls": int(df[c].isna().sum()),
                   "distinct": int(df[c].nunique())}
            if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 12:
                col["values"] = sorted(map(str, df[c].dropna().unique()))
            columns.append(col)
        sample = json.loads(df.head(3).to_json(orient="records"))
        return {"table": table, "rows": len(df), "columns": columns, "sample_rows": sample}

    def run_sql(self, query: str) -> dict:
        try:
            q = validate_sql(query)
        except UnsafeSQLError as e:
            raise ToolError(f"Rejected query: {e}") from e

        deadline = time.monotonic() + self.timeout_s
        # Abort runaway queries (e.g. accidental cross joins). Returning 1 interrupts SQLite.
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = self.conn.execute(q)
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise ToolError(f"Query exceeded {self.timeout_s}s time limit. Simplify it.") from e
            raise ToolError(f"SQL error: {e}") from e
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        finally:
            self.conn.set_progress_handler(None, 0)

        cols = [d[0] for d in cur.description]
        truncated = len(rows) > self.max_rows
        return {"columns": cols, "rows": [list(r) for r in rows[: self.max_rows]],
                "row_count": min(len(rows), self.max_rows), "truncated": truncated}


TOOL_SPECS = {
    "list_tables": {"args": {}, "description": "List all tables with row counts and column names."},
    "describe_table": {"args": {"table": "string"},
                       "description": "Column types, null counts, distinct values and 3 sample rows for one table."},
    "run_sql": {"args": {"query": "string"},
                "description": "Run ONE read-only SQLite SELECT query. Returns at most 50 rows."},
    "final_answer": {"args": {"answer": "string", "value": "number | string | null"},
                     "description": "Finish. 'answer' is a one-sentence reply; 'value' is the single key result."},
}


def call_tool(db: Database, name: str, args: dict) -> dict:
    if name == "list_tables":
        return db.list_tables()
    if name == "describe_table":
        return db.describe_table(str(args.get("table", "")))
    if name == "run_sql":
        return db.run_sql(str(args.get("query", "")))
    raise ToolError(f"Unknown tool '{name}'")
