import pytest

from tabsage.tools import ToolError


def test_list_tables(db):
    tables = db.list_tables()["tables"]
    assert set(tables) == {"customers", "orders", "products"}
    assert tables["orders"]["rows"] == 600


def test_describe_table_includes_categorical_values(db):
    d = db.describe_table("orders")
    status = next(c for c in d["columns"] if c["name"] == "status")
    assert status["values"] == ["cancelled", "completed", "returned"]


def test_describe_unknown_table_lists_alternatives(db):
    with pytest.raises(ToolError, match="Available tables"):
        db.describe_table("sales")


def test_run_sql_select(db):
    out = db.run_sql("SELECT COUNT(*) AS n FROM customers")
    assert out["rows"] == [[60]]


@pytest.mark.parametrize("q", [
    "DELETE FROM orders",
    "DROP TABLE orders",
    "SELECT 1; DROP TABLE orders",
    "UPDATE orders SET quantity = 0",
    "PRAGMA query_only = OFF",
    "",
])
def test_run_sql_rejects_unsafe(db, q):
    with pytest.raises(ToolError):
        db.run_sql(q)


def test_keyword_inside_string_literal_is_allowed(db):
    out = db.run_sql("SELECT COUNT(*) FROM orders WHERE status = 'delete'")
    assert out["rows"] == [[0]]


def test_database_is_read_only_even_if_validator_bypassed(db):
    import sqlite3
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("DELETE FROM orders")


def test_bad_sql_returns_helpful_error(db):
    with pytest.raises(ToolError, match="no such column"):
        db.run_sql("SELECT revenue FROM orders")


def test_results_truncated(db):
    out = db.run_sql("SELECT * FROM orders")
    assert out["truncated"] is True and out["row_count"] == 50


def test_runaway_query_times_out(db):
    db.timeout_s = 0.2
    with pytest.raises(ToolError, match="time limit"):
        db.run_sql("SELECT COUNT(*) FROM orders a, orders b, orders c")
