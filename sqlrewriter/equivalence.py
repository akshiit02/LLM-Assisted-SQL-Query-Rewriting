"""Stage 5 – Semantic equivalence verification.

The original query and the candidate are executed on the same database state
and their result sets are compared *strictly*:

* same number of columns, (case-insensitive) column names and type categories;
* identical multisets of rows (bag semantics – duplicates matter);
  numeric values are compared with a small relative/absolute tolerance so that
  floating-point summation order does not cause false rejections;
* if the original query has a top-level ``ORDER BY`` the row *order* must match
  too.  Ordering is checked tie-aware: rows whose ORDER BY keys are equal may
  appear in any order (the SQL standard does not define their order).  If the
  keys cannot be mapped to output columns the check falls back to an exact
  positional comparison (conservative – it can only reject, never wrongly
  accept).

Result-based verification cannot *prove* equivalence for every database
state; it is run for every experimental condition (every scale factor and
index configuration), and empty-vs-empty results are flagged as *vacuous*.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from decimal import Decimal
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from .config import EquivalenceSettings
from .db.base import ResultSet
from .models import EquivalenceResult


# --------------------------------------------------------------------------- values
def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if v.is_integer() and abs(v) < 2**53:
            return int(v)
        return v
    if isinstance(v, dt.datetime):
        if v.time() == dt.time(0, 0):
            return v.date().isoformat()
        return v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    if hasattr(v, "read"):  # LOB
        return normalize_value(v.read())
    return v


def _hash_key(v: Any) -> Any:
    """Coarse key used for fast multiset comparison (floats rounded to 9 s.f.)."""
    if isinstance(v, float):
        return ("f", float(f"{v:.9g}"))
    if isinstance(v, int):
        return ("f", float(f"{float(v):.9g}")) if abs(v) >= 2**53 else ("f", float(v))
    if v is None:
        return ("n",)
    return ("s", str(v))


def _values_close(a: Any, b: Any, rel: float, abs_: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    return a == b


def _rows_close(r1: tuple, r2: tuple, rel: float, abs_: float) -> bool:
    return len(r1) == len(r2) and all(_values_close(a, b, rel, abs_) for a, b in zip(r1, r2))


def _sort_key(row: tuple) -> tuple:
    out = []
    for v in row:
        if v is None:
            out.append((0, 0, ""))
        elif isinstance(v, (int, float)):
            out.append((1, float(v), ""))
        else:
            out.append((2, 0, str(v)))
    return tuple(out)


def _multiset_equal(a: list[tuple], b: list[tuple], rel: float, abs_: float) -> bool:
    if len(a) != len(b):
        return False
    ca = Counter(tuple(_hash_key(v) for v in r) for r in a)
    cb = Counter(tuple(_hash_key(v) for v in r) for r in b)
    if ca == cb:
        return True
    # tolerance-aware fallback: sort both and compare pairwise
    sa = sorted(a, key=_sort_key)
    sb = sorted(b, key=_sort_key)
    return all(_rows_close(x, y, rel, abs_) for x, y in zip(sa, sb))


def _multiset_diff(a: list[tuple], b: list[tuple], limit: int) -> tuple[list, list]:
    ca = Counter(tuple(_hash_key(v) for v in r) for r in a)
    cb = Counter(tuple(_hash_key(v) for v in r) for r in b)
    lookup_a = {tuple(_hash_key(v) for v in r): r for r in a}
    lookup_b = {tuple(_hash_key(v) for v in r): r for r in b}
    only_a = [list(lookup_a[k]) for k in (ca - cb)][:limit]
    only_b = [list(lookup_b[k]) for k in (cb - ca)][:limit]
    return only_a, only_b


# --------------------------------------------------------------------------- ordering
def order_by_key_indices(sql: str, dialect: str, columns: list[str]) -> Optional[list[int]]:
    """Map the top-level ORDER BY keys to output column positions.

    Returns ``[]`` when the query has no top-level ORDER BY, ``None`` when at
    least one key cannot be mapped to an output column.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None
    order = tree.args.get("order")
    if order is None:
        return []
    names = [c.lower() for c in columns]
    first_select = tree if isinstance(tree, exp.Select) else next(tree.find_all(exp.Select), None)
    select_exprs = list(first_select.expressions) if first_select is not None else []
    indices: list[int] = []
    for ordered in order.expressions:
        e = ordered.this
        idx = None
        if isinstance(e, exp.Literal) and e.is_int:
            idx = int(e.this) - 1
        elif isinstance(e, exp.Column) and e.name.lower() in names:
            # a bare name matches an alias/column of the output
            idx = names.index(e.name.lower())
        else:
            target = e.sql(dialect=dialect).lower()
            for i, se in enumerate(select_exprs):
                inner = se.this if isinstance(se, exp.Alias) else se
                if inner.sql(dialect=dialect).lower() == target:
                    idx = i
                    break
        if idx is None or not (0 <= idx < len(columns)):
            return None
        indices.append(idx)
    return indices


def has_top_level_order_by(sql: str, dialect: str) -> bool:
    try:
        return sqlglot.parse_one(sql, read=dialect).args.get("order") is not None
    except Exception:
        return "order by" in sql.lower()


def _ordered_equal(a: list[tuple], b: list[tuple], key_idx: Optional[list[int]],
                   rel: float, abs_: float) -> tuple[bool, str]:
    if key_idx is None:
        ok = all(_rows_close(x, y, rel, abs_) for x, y in zip(a, b))
        return ok, "rows are not in the same order (exact positional comparison)"
    keys_a = [tuple(r[i] for i in key_idx) for r in a]
    keys_b = [tuple(r[i] for i in key_idx) for r in b]
    if not all(_rows_close(x, y, rel, abs_) for x, y in zip(keys_a, keys_b)):
        return False, "ORDER BY key sequence differs"
    # compare multisets inside each run of equal keys
    start = 0
    for i in range(1, len(a) + 1):
        if i == len(a) or not _rows_close(keys_a[i], keys_a[start], rel, abs_):
            if not _multiset_equal(a[start:i], b[start:i], rel, abs_):
                return False, f"rows within an ORDER BY tie group differ (rows {start + 1}-{i})"
            start = i
    return True, ""


# --------------------------------------------------------------------------- main API
def compare_results(original: ResultSet, rewrite: ResultSet, original_sql: str, dialect: str,
                    settings: EquivalenceSettings) -> EquivalenceResult:
    rel, abs_ = settings.float_rel_tolerance, settings.float_abs_tolerance
    n_o, n_r = len(original.rows), len(rewrite.rows)

    def result(ok: bool, reason: str, category: str, **kw) -> EquivalenceResult:
        return EquivalenceResult(equivalent=ok, reason=reason, category=category,
                                 original_rows=n_o, rewrite_rows=n_r, **kw)

    if original.truncated or rewrite.truncated:
        return result(False, f"result exceeds max_rows={settings.max_rows}; equivalence cannot be verified",
                      "error")

    if len(original.columns) != len(rewrite.columns):
        return result(False, f"column count differs ({len(original.columns)} vs {len(rewrite.columns)})",
                      "columns")

    warnings = []
    orig_names = [c.lower() for c in original.columns]
    rew_names = [c.lower() for c in rewrite.columns]
    if orig_names != rew_names:
        msg = f"column names differ: {original.columns} vs {rewrite.columns}"
        if settings.strict_column_names:
            return result(False, msg, "columns")
        warnings.append(msg)

    for i, (t1, t2) in enumerate(zip(original.column_types, rewrite.column_types)):
        if "unknown" in (t1, t2) or t1 == t2:
            continue
        return result(False, f"type of column {original.columns[i]!r} differs ({t1} vs {t2})", "types",
                      column_warnings=warnings)

    rows_o = [tuple(normalize_value(v) for v in r) for r in original.rows]
    rows_r = [tuple(normalize_value(v) for v in r) for r in rewrite.rows]
    ordered = has_top_level_order_by(original_sql, dialect)

    if n_o != n_r:
        only_o, only_r = _multiset_diff(rows_o, rows_r, settings.diff_sample_size)
        return result(False, f"row count differs ({n_o} vs {n_r})", "row_count", ordered=ordered,
                      column_warnings=warnings, only_in_original=only_o, only_in_rewrite=only_r)

    if not _multiset_equal(rows_o, rows_r, rel, abs_):
        only_o, only_r = _multiset_diff(rows_o, rows_r, settings.diff_sample_size)
        return result(False, "row values differ", "values", ordered=ordered, column_warnings=warnings,
                      only_in_original=only_o, only_in_rewrite=only_r)

    if ordered:
        key_idx = order_by_key_indices(original_sql, dialect, original.columns)
        ok, why = _ordered_equal(rows_o, rows_r, key_idx, rel, abs_)
        if not ok:
            return result(False, why, "order", ordered=True, column_warnings=warnings)

    vacuous = n_o == 0
    reason = "identical result sets" + (" (ordered)" if ordered else " (as multisets)")
    if vacuous:
        reason += " – WARNING: both results are empty, equivalence evidence is vacuous"
    return result(True, reason, "equivalent", ordered=ordered, vacuous=vacuous, column_warnings=warnings)
