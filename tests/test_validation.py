import pytest

from sqlrewriter.validation import categorize_error, clean_sql, lexical_safety_check, validate_candidate


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE orders SET status = 'X'",
    "DROP TABLE orders",
    "SELECT * FROM orders; DELETE FROM orders",
    "INSERT INTO regions VALUES (9, 'X')",
    "SELECT * FROM orders FOR UPDATE",
    "CREATE TABLE x AS SELECT * FROM orders",
    "PRAGMA writable_schema = 1",
])
def test_unsafe_statements_rejected(sql, backend):
    v = validate_candidate(sql, backend)
    assert not v.passed and v.stage == "safety" and v.error_category == "unsafe_statement"


def test_valid_select_and_with_accepted(backend):
    assert validate_candidate("SELECT order_id FROM orders WHERE status = 'SHIPPED'", backend).passed
    assert validate_candidate("WITH x AS (SELECT customer_id FROM orders) SELECT COUNT(*) FROM x", backend).passed


def test_keywords_inside_literals_do_not_trigger_safety(backend):
    v = validate_candidate("SELECT 'DELETE me' AS note, order_id FROM orders", backend)
    assert v.passed


def test_unknown_column_is_unexecutable(backend):
    v = validate_candidate("SELECT no_such_col FROM orders", backend)
    assert not v.passed and v.stage == "executable" and v.error_category == "invalid_identifier"


def test_unknown_table(backend):
    v = validate_candidate("SELECT * FROM nope", backend)
    assert not v.passed and v.error_category == "missing_table"


def test_syntax_error(backend):
    v = validate_candidate("SELECT order_id FROM orders WHERE", backend)
    assert not v.passed and v.stage in ("syntax", "executable")


def test_clean_sql_strips_fences_and_semicolons():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert clean_sql("SELECT 1;;") == "SELECT 1"


def test_lexical_check():
    assert lexical_safety_check("SELECT 1") is None
    assert "forbidden" in lexical_safety_check("SELECT 1 FROM t WHERE x = 1 -- ok\n UNION SELECT 1 FROM t; DROP TABLE t") \
        or "multiple" in lexical_safety_check("SELECT 1; DROP TABLE t")


def test_error_categories():
    assert categorize_error("ORA-00904: \"FOO\": invalid identifier") == "invalid_identifier"
    assert categorize_error("ORA-00942: table or view does not exist") == "missing_table"
    assert categorize_error("ORA-00979: not a GROUP BY expression") == "group_by_error"
    assert categorize_error('near "FROM": syntax error') == "syntax_error"
    assert categorize_error("no such function: FOO") == "unsupported_function"
