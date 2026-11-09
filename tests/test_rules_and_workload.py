import sqlglot

from sqlrewriter.complexity import classify, complexity_metrics
from sqlrewriter.datagen import generate
from sqlrewriter.llm.rule_based import (rule_exists_to_in, rule_having_to_where, rule_in_to_exists,
                                        rule_or_to_in, rule_remove_redundant_distinct, rule_sargable_predicates,
                                        rule_union_to_or)
from sqlrewriter.workload import load_workload


def apply(rule, sql, dialect="oracle"):
    out = rule(sqlglot.parse_one(sql, read=dialect), dialect)
    return None if out is None else out.sql(dialect=dialect)


# ------------------------------------------------------------------ data generator
def test_datagen_is_deterministic_and_has_nulls():
    a, b = generate(0.05, seed=7), generate(0.05, seed=7)
    assert a.tables["orders"][:50] == b.tables["orders"][:50]
    assert generate(0.05, seed=8).tables["orders"][:50] != a.tables["orders"][:50]
    assert any(r[6] is None for r in a.tables["customers"])  # credit_limit
    assert any(r[3] is None for r in a.tables["products"])  # supplier_id
    counts = a.row_counts()
    assert counts["regions"] == 5 and counts["nations"] == 25 and counts["order_items"] > counts["orders"]


def test_referential_integrity():
    d = generate(0.05, seed=1)
    order_ids = {r[0] for r in d.tables["orders"]}
    product_ids = {r[0] for r in d.tables["products"]}
    assert all(r[0] in order_ids and r[2] in product_ids for r in d.tables["order_items"])


# ------------------------------------------------------------------ workload
def test_workload_has_all_levels():
    qs = load_workload()
    assert len(qs) == 20
    assert {q.complexity for q in qs} == {"basic", "intermediate", "advanced", "complex"}


def test_every_workload_query_runs_on_sqlite_and_returns_rows(backend):
    for q in load_workload():
        rs = backend.execute(backend.prepare_sql(q.sql), 10**6, 60)
        assert rs.rows, f"{q.id} returned no rows"


def test_complexity_score_increases_with_level():
    qs = load_workload()
    avg = {}
    for lvl in ("basic", "complex"):
        scores = [complexity_metrics(q.sql)["score"] for q in qs if q.complexity == lvl]
        avg[lvl] = sum(scores) / len(scores)
    assert avg["complex"] > avg["basic"]
    assert classify({"score": 1}) == "basic" and classify({"score": 40}) == "complex"


# ------------------------------------------------------------------ rules
def test_sargable_year_extract():
    out = apply(rule_sargable_predicates, "SELECT 1 FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2024")
    assert "EXTRACT" not in out.upper() and "2024-01-01" in out and "2025-01-01" in out


def test_sargable_division():
    out = apply(rule_sargable_predicates, "SELECT 1 FROM orders WHERE total_amount / 2 > 400")
    assert out.endswith("total_amount > 800")


def test_or_to_in():
    out = apply(rule_or_to_in, "SELECT 1 FROM p WHERE (a = 1 OR a = 2 OR a = 3) AND b > 0")
    assert "IN (1, 2, 3)" in out


def test_or_to_in_not_applied_to_different_columns():
    assert apply(rule_or_to_in, "SELECT 1 FROM p WHERE a = 1 OR b = 2") is None


def test_having_to_where():
    out = apply(rule_having_to_where,
                "SELECT status, COUNT(*) FROM orders GROUP BY status HAVING status IN ('A','B') AND COUNT(*) > 1")
    assert "WHERE status IN ('A', 'B')" in out and "HAVING COUNT(*) > 1" in out


def test_redundant_distinct_removed_only_with_pk():
    assert "DISTINCT" not in apply(rule_remove_redundant_distinct,
                                   "SELECT DISTINCT c.customer_id, c.name FROM customers c")
    assert apply(rule_remove_redundant_distinct, "SELECT DISTINCT c.name FROM customers c") is None


def test_in_to_exists():
    out = apply(rule_in_to_exists,
                "SELECT c.name FROM customers c WHERE c.customer_id IN (SELECT o.customer_id FROM orders o WHERE o.total_amount > 5)")
    assert "EXISTS" in out and "o.customer_id = c.customer_id" in out


def test_not_exists_to_not_in_guarded_by_nullability():
    safe = ("SELECT c.name FROM customers c WHERE NOT EXISTS "
            "(SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)")
    assert "NOT c.customer_id IN" in apply(rule_exists_to_in, safe)
    nullable = ("SELECT s.name FROM suppliers s WHERE NOT EXISTS "
                "(SELECT 1 FROM products p WHERE p.supplier_id = s.supplier_id)")
    assert apply(rule_exists_to_in, nullable) is None  # products.supplier_id is nullable


def test_union_to_or():
    out = apply(rule_union_to_or,
                "SELECT order_id, total_amount FROM orders WHERE a = 1 UNION SELECT order_id, total_amount FROM orders WHERE b = 2 ORDER BY order_id")
    assert "UNION" not in out and " OR " in out and "ORDER BY order_id" in out


def test_naive_union_all_rule_is_caught_by_verifier(settings, backend):
    from sqlrewriter.models import ExperimentCondition
    from sqlrewriter.pipeline import RewritePipeline
    from sqlrewriter.llm.rule_based import RuleBasedRewriter

    q = next(q for q in load_workload() if q.id == "Q10")
    res = RewritePipeline(settings, backend, [RuleBasedRewriter(rules=["union_to_union_all", "union_to_or"])]) \
        .evaluate(q, ExperimentCondition(0.1, "fk_indexes", "sqlite"))
    verdicts = {o.candidate.strategy: o.verdict for o in res.outcomes}
    assert verdicts["union_to_union_all"] == "rejected_nonequivalent"  # overlapping branches -> duplicates
    assert verdicts["union_to_or"] in ("improved", "neutral", "regressed")
