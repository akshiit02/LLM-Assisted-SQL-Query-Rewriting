"""Structural SQL complexity metrics (supports RQ4).

Besides the hand-assigned complexity *level* of each workload query, the
analysis uses these measured features and a weighted *complexity score* so
the relationship between complexity and rewrite success can be studied as a
continuous variable as well.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

WEIGHTS = {
    "tables": 1.0,
    "joins": 1.5,
    "subqueries": 2.5,
    "correlated_subqueries": 3.5,
    "ctes": 2.0,
    "window_functions": 2.5,
    "aggregates": 1.0,
    "set_operations": 2.0,
    "predicates": 0.5,
    "case_expressions": 1.0,
    "max_nesting_depth": 2.0,
}


def _select_depth(node: exp.Expression) -> int:
    best = 0

    def walk(n: exp.Expression, depth: int) -> None:
        nonlocal best
        for child in n.iter_expressions():
            d = depth + 1 if isinstance(child, exp.Select) else depth
            best = max(best, d)
            walk(child, d)

    start = 1 if isinstance(node, exp.Select) else 0
    best = start
    walk(node, start)
    return best


def _sources_defined(select: exp.Select) -> set[str]:
    names = set()
    for t in select.find_all(exp.Table):
        names.add((t.alias_or_name or "").lower())
    for sq in select.find_all(exp.Subquery):
        if sq.alias:
            names.add(sq.alias.lower())
    return names


def _is_correlated(sub: exp.Select, cte_names: set[str]) -> bool:
    local = _sources_defined(sub) | cte_names
    for col in sub.find_all(exp.Column):
        q = (col.table or "").lower()
        if q and q not in local:
            return True
    return False


def complexity_metrics(sql: str, dialect: str = "oracle") -> dict[str, Any]:
    tree = sqlglot.parse_one(sql, read=dialect)
    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}

    selects = list(tree.find_all(exp.Select))
    subqueries = 0
    correlated = 0
    for s in selects:
        parent = s.parent
        # a SELECT is a subquery if it is nested inside an expression other than a CTE / set-op root
        if parent is None or isinstance(parent, (exp.CTE, exp.Union, exp.Intersect, exp.Except)):
            continue
        if s is tree:
            continue
        subqueries += 1
        if _is_correlated(s, cte_names):
            correlated += 1

    predicate_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like,
                       exp.Between, exp.Exists, exp.Is)
    metrics = {
        "tables": sum(1 for t in tree.find_all(exp.Table) if t.name.lower() not in cte_names),
        "joins": sum(1 for _ in tree.find_all(exp.Join)),
        "subqueries": subqueries,
        "correlated_subqueries": correlated,
        "ctes": len(cte_names),
        "window_functions": sum(1 for _ in tree.find_all(exp.Window)),
        "aggregates": sum(1 for _ in tree.find_all(exp.AggFunc)),
        "set_operations": sum(1 for _ in tree.find_all(exp.Union, exp.Intersect, exp.Except)),
        "predicates": sum(1 for _ in tree.find_all(*predicate_types)),
        "case_expressions": sum(1 for _ in tree.find_all(exp.Case)),
        "max_nesting_depth": _select_depth(tree),
        "distinct": sum(1 for _ in tree.find_all(exp.Distinct)),
        "length_chars": len(sql),
    }
    metrics["score"] = round(sum(metrics[k] * w for k, w in WEIGHTS.items()), 2)
    return metrics


def classify(metrics: dict[str, Any]) -> str:
    score = metrics["score"]
    if score < 8:
        return "basic"
    if score < 16:
        return "intermediate"
    if score < 28:
        return "advanced"
    return "complex"
