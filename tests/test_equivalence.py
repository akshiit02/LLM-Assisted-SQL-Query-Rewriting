import datetime as dt
from decimal import Decimal

from sqlrewriter.config import EquivalenceSettings
from sqlrewriter.db.base import ResultSet
from sqlrewriter.equivalence import compare_results, normalize_value, order_by_key_indices

CFG = EquivalenceSettings()


def rs(cols, rows, types=None):
    return ResultSet(cols, types or ["number"] * len(cols), rows)


def test_identical_unordered_results_in_any_order():
    a = rs(["id", "v"], [(1, 10), (2, 20), (3, 30)])
    b = rs(["id", "v"], [(3, 30), (1, 10), (2, 20)])
    r = compare_results(a, b, "SELECT id, v FROM t", "sqlite", CFG)
    assert r.equivalent and not r.ordered


def test_duplicates_matter_bag_semantics():
    a = rs(["id"], [(1,), (1,), (2,)])
    b = rs(["id"], [(1,), (2,), (2,)])
    r = compare_results(a, b, "SELECT id FROM t", "sqlite", CFG)
    assert not r.equivalent and r.category == "values"
    assert r.only_in_original == [[1]] and r.only_in_rewrite == [[2]]


def test_row_count_difference_reported():
    a = rs(["id"], [(1,), (2,)])
    b = rs(["id"], [(1,)])
    r = compare_results(a, b, "SELECT id FROM t", "sqlite", CFG)
    assert not r.equivalent and r.category == "row_count"


def test_float_noise_is_tolerated_but_real_differences_are_not():
    a = rs(["s"], [(0.1 + 0.2,), (1234.56,)])
    b = rs(["s"], [(0.3,), (1234.5600000001,)])
    assert compare_results(a, b, "SELECT s FROM t", "sqlite", CFG).equivalent
    c = rs(["s"], [(0.3,), (1234.57,)])
    assert not compare_results(a, c, "SELECT s FROM t", "sqlite", CFG).equivalent


def test_order_matters_when_original_is_ordered():
    a = rs(["id", "v"], [(1, 5), (2, 4), (3, 3)])
    b = rs(["id", "v"], [(3, 3), (2, 4), (1, 5)])
    r = compare_results(a, b, "SELECT id, v FROM t ORDER BY id", "sqlite", CFG)
    assert not r.equivalent and r.category == "order"


def test_ties_in_order_by_may_be_permuted():
    a = rs(["grp", "id"], [(1, 10), (1, 11), (2, 12)])
    b = rs(["grp", "id"], [(1, 11), (1, 10), (2, 12)])
    r = compare_results(a, b, "SELECT grp, id FROM t ORDER BY grp", "sqlite", CFG)
    assert r.equivalent and r.ordered


def test_unmappable_order_key_falls_back_to_exact_positions():
    a = rs(["id"], [(1,), (2,)])
    b = rs(["id"], [(2,), (1,)])
    r = compare_results(a, b, "SELECT id FROM t ORDER BY other_col", "sqlite", CFG)
    assert not r.equivalent


def test_column_name_and_count_checks():
    a = rs(["id", "v"], [(1, 2)])
    assert compare_results(a, rs(["id"], [(1,)]), "SELECT 1", "sqlite", CFG).category == "columns"
    r = compare_results(a, rs(["ID", "value"], [(1, 2)]), "SELECT 1", "sqlite", CFG)
    assert not r.equivalent and r.category == "columns"
    lax = EquivalenceSettings(strict_column_names=False)
    r = compare_results(a, rs(["id", "value"], [(1, 2)]), "SELECT 1", "sqlite", lax)
    assert r.equivalent and r.column_warnings


def test_case_insensitive_column_names():
    a = rs(["ORDER_ID"], [(1,)])
    b = rs(["order_id"], [(1,)])
    assert compare_results(a, b, "SELECT 1", "oracle", CFG).equivalent


def test_type_mismatch_detected():
    a = rs(["v"], [("1",)], ["text"])
    b = rs(["v"], [(1,)], ["number"])
    assert compare_results(a, b, "SELECT 1", "sqlite", CFG).category == "types"


def test_vacuous_equivalence_flagged():
    r = compare_results(rs(["id"], []), rs(["id"], []), "SELECT id FROM t", "sqlite", CFG)
    assert r.equivalent and r.vacuous


def test_null_handling():
    a = rs(["v"], [(None,), (1,)])
    b = rs(["v"], [(1,), (None,)])
    assert compare_results(a, b, "SELECT v FROM t", "sqlite", CFG).equivalent
    c = rs(["v"], [(0,), (1,)])
    assert not compare_results(a, c, "SELECT v FROM t", "sqlite", CFG).equivalent


def test_normalize_values():
    assert normalize_value(Decimal("2.50")) == 2.5
    assert normalize_value(3.0) == 3
    assert normalize_value(dt.datetime(2024, 1, 2)) == "2024-01-02"
    assert normalize_value(dt.datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02 03:04:05"
    assert normalize_value(True) == 1


def test_order_by_key_mapping():
    cols = ["nation", "customer_id", "revenue"]
    sql = "SELECT n.name AS nation, t.customer_id, t.revenue FROM t JOIN n ON 1=1 ORDER BY n.name, t.customer_id"
    assert order_by_key_indices(sql, "oracle", cols) == [0, 1]
    assert order_by_key_indices("SELECT a, b FROM t ORDER BY 2, 1", "oracle", ["a", "b"]) == [1, 0]
    assert order_by_key_indices("SELECT a FROM t", "oracle", ["a"]) == []
    assert order_by_key_indices("SELECT a FROM t ORDER BY b", "oracle", ["a"]) is None
