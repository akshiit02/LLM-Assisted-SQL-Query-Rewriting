"""Rule-based query rewriter (offline baseline, no API key required).

This plays two roles in the study:

1. **Baseline** – "traditional query rewriting is rule-based" (Research Gap).
   Running the same verification + benchmarking pipeline on classic
   transformation rules lets the analysis compare LLM rewrites against a
   rule engine.
2. **Offline fallback** – the full pipeline can be demonstrated end-to-end on
   any machine, with no network access or credentials.

Rules are implemented as sqlglot AST transformations.  Some are always
equivalent (e.g. OR-chain -> IN), some are only equivalent under conditions
the rule deliberately does *not* check (``not_in_to_not_exists`` ignores
NULLs, exactly like a naive rewrite would) – the verifier is expected to
catch those, which exercises the rejection paths of the pipeline.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal
from typing import Callable, Optional

import sqlglot
from sqlglot import exp

from .. import schema as S
from ..models import Candidate
from .base import RewriteRequest, Rewriter

# --------------------------------------------------------------------------- helpers


def _from(sel: exp.Select) -> Optional[exp.From]:
    return sel.args.get("from_") or sel.args.get("from")


def _from_key(sel: exp.Select) -> str:
    return "from_" if "from_" in sel.args or "from" not in sel.args else "from"


def _unparen(e: exp.Expression) -> exp.Expression:
    while isinstance(e, exp.Paren):
        e = e.this
    return e


def _conjuncts(e: Optional[exp.Expression]) -> list[exp.Expression]:
    if e is None:
        return []
    e = _unparen(e)
    if isinstance(e, exp.And):
        return _conjuncts(e.this) + _conjuncts(e.expression)
    return [e]


def _disjuncts(e: exp.Expression) -> list[exp.Expression]:
    e = _unparen(e)
    if isinstance(e, exp.Or):
        return _disjuncts(e.this) + _disjuncts(e.expression)
    return [e]


def _and_all(parts: list[exp.Expression]) -> Optional[exp.Expression]:
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = exp.and_(out, p)
    return out


def _date_literal(value: str, dialect: str) -> exp.Expression:
    if dialect == "sqlite":
        return exp.Literal.string(value)  # dates are ISO-8601 text in SQLite
    return exp.DateStrToDate(this=exp.Literal.string(value))


def _num_literal(e: exp.Expression) -> Optional[Decimal]:
    e = _unparen(e)
    if isinstance(e, exp.Literal) and not e.is_string:
        try:
            return Decimal(e.this)
        except Exception:
            return None
    if isinstance(e, exp.Neg):
        v = _num_literal(e.this)
        return -v if v is not None else None
    return None


def _year_column(e: exp.Expression) -> Optional[exp.Column]:
    """Return the column if ``e`` extracts the YEAR of a plain column."""
    e = _unparen(e)
    if isinstance(e, exp.Extract) and e.this is not None and e.this.name.upper() == "YEAR":
        return e.expression if isinstance(e.expression, exp.Column) else None
    if isinstance(e, exp.Cast):  # SQLite: CAST(STRFTIME('%Y', col) AS INTEGER)
        inner = e.this
        if isinstance(inner, exp.TimeToStr) and inner.args.get("format") is not None \
                and inner.args["format"].name == "%Y":
            col = inner.this
            if isinstance(col, exp.TsOrDsToTimestamp):
                col = col.this
            return col if isinstance(col, exp.Column) else None
    return None


def _alias_map(sel: exp.Select) -> dict[str, str]:
    """alias -> table name for tables directly in this SELECT's FROM/JOINs."""
    out = {}
    f = _from(sel)
    sources = [f.this] if f is not None else []
    sources += [j.this for j in sel.args.get("joins") or []]
    for src in sources:
        if isinstance(src, exp.Table):
            out[(src.alias_or_name or "").lower()] = src.name.lower()
    return out


def _local_aliases(sel: exp.Select) -> set[str]:
    names = set()
    for t in sel.find_all(exp.Table):
        names.add((t.alias_or_name or "").lower())
    for sq in sel.find_all(exp.Subquery):
        if sq.alias:
            names.add(sq.alias.lower())
    return names


def _is_simple_select(sel: exp.Expression) -> bool:
    return (isinstance(sel, exp.Select) and not sel.args.get("group") and not sel.args.get("having")
            and not sel.args.get("limit") and not sel.args.get("fetch") and not sel.args.get("offset")
            and not any(isinstance(n, exp.AggFunc) for n in sel.expressions for n in n.walk())
            and not sel.args.get("with") and not sel.find(exp.Window))


def _column_nullable(table: str, column: str) -> bool:
    t = S.TABLES_BY_NAME.get(table.lower())
    if t is None:
        return True
    try:
        return t.column(column.lower()).nullable
    except KeyError:
        return True


# --------------------------------------------------------------------------- rules
Rule = Callable[[exp.Expression, str], Optional[exp.Expression]]


def rule_sargable_predicates(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if not isinstance(node, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            return node
        left, right = node.this, node.expression
        # --- EXTRACT(YEAR FROM col) <op> N  ->  date range on col
        col, lit, op = _year_column(left), _num_literal(right), type(node)
        if col is None:
            col, lit = _year_column(right), _num_literal(left)
            op = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}.get(op, op)
        if col is not None and lit is not None and lit == int(lit):
            y = int(lit)
            start = _date_literal(dt.date(y, 1, 1).isoformat(), dialect)
            nxt = _date_literal(dt.date(y + 1, 1, 1).isoformat(), dialect)
            changed = True
            if op is exp.EQ:
                return exp.Paren(this=exp.and_(exp.GTE(this=col.copy(), expression=start),
                                               exp.LT(this=col.copy(), expression=nxt)))
            if op is exp.GTE:
                return exp.GTE(this=col.copy(), expression=start)
            if op is exp.GT:
                return exp.GTE(this=col.copy(), expression=nxt)
            if op is exp.LT:
                return exp.LT(this=col.copy(), expression=start)
            if op is exp.LTE:
                return exp.LT(this=col.copy(), expression=nxt)
        # --- col / k <op> c  (k > 0)  ->  col <op> c * k
        l = _unparen(left)
        if isinstance(l, exp.Div):
            num = _unparen(l.this)
            if isinstance(num, exp.Cast) and isinstance(num.this, exp.Column) and num.to.is_type(
                    "real", "double", "float", "decimal"):
                num = num.this  # SQLite emits CAST(col AS REAL) / k for Oracle division
            k, c = _num_literal(l.expression), _num_literal(right)
            if isinstance(num, exp.Column) and k is not None and c is not None and k > 0:
                changed = True
                return type(node)(this=num.copy(),
                                  expression=exp.Literal.number(format((c * k).normalize(), "f")))
        return node

    new = tree.transform(transform)
    return new if changed else None


def rule_or_to_in(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if not isinstance(node, exp.Or) or isinstance(node.parent, exp.Or):
            return node
        parts = _disjuncts(node)
        if len(parts) < 2 or not all(isinstance(p, exp.EQ) for p in parts):
            return node
        cols, lits = set(), []
        for p in parts:
            a, b = p.this, p.expression
            if isinstance(a, exp.Literal) and isinstance(b, exp.Column):
                a, b = b, a
            if not (isinstance(a, exp.Column) and isinstance(b, exp.Literal)):
                return node
            cols.add(a.sql())
            lits.append(b.copy())
        if len(cols) != 1:
            return node
        changed = True
        return exp.In(this=_unparen(parts[0]).this.copy() if isinstance(parts[0].this, exp.Column)
                      else parts[0].expression.copy(), expressions=lits)

    new = tree.transform(transform)
    return new if changed else None


def _in_subquery_to_exists(node: exp.In) -> Optional[exp.Exists]:
    query = node.args.get("query")
    outer = node.this
    if query is None or not isinstance(outer, exp.Column) or not outer.table:
        return None
    sub = query.this if isinstance(query, exp.Subquery) else query
    if not _is_simple_select(sub) or len(sub.expressions) != 1:
        return None
    if outer.table.lower() in _local_aliases(sub):
        return None  # alias would be captured by the inner scope
    sub = sub.copy()
    proj = sub.expressions[0]
    inner = proj.this if isinstance(proj, exp.Alias) else proj
    corr = exp.EQ(this=inner.copy(), expression=outer.copy())
    where = sub.args.get("where")
    sub.set("where", exp.Where(this=exp.and_(where.this, corr) if where else corr))
    sub.set("expressions", [exp.Literal.number(1)])
    sub.set("distinct", None)
    return exp.Exists(this=sub)


def rule_in_to_exists(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if isinstance(node, exp.In) and not isinstance(node.parent, exp.Not):
            ex = _in_subquery_to_exists(node)
            if ex is not None:
                changed = True
                return ex
        return node

    new = tree.transform(transform)
    return new if changed else None


def rule_not_in_to_not_exists(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    """Naive rule: ignores NULL semantics on purpose (verifier must catch unsafe cases)."""
    changed = False

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if isinstance(node, exp.Not) and isinstance(node.this, exp.In):
            ex = _in_subquery_to_exists(node.this)
            if ex is not None:
                changed = True
                return exp.Not(this=ex)
        return node

    new = tree.transform(transform)
    return new if changed else None


def _exists_to_in(node: exp.Exists, negated: bool) -> Optional[exp.In]:
    sub = node.this
    if not _is_simple_select(sub):
        return None
    local = _local_aliases(sub)
    amap = _alias_map(sub)
    conj = _conjuncts(sub.args["where"].this) if sub.args.get("where") else []
    for i, c in enumerate(conj):
        if not isinstance(c, exp.EQ) or not all(isinstance(x, exp.Column) and x.table for x in (c.this, c.expression)):
            continue
        a, b = c.this, c.expression
        if a.table.lower() in local and b.table.lower() not in local:
            inner, outer = a, b
        elif b.table.lower() in local and a.table.lower() not in local:
            inner, outer = b, a
        else:
            continue
        # the remaining predicates must not reference the outer query
        rest = conj[:i] + conj[i + 1:]
        if any(col.table and col.table.lower() not in local for r in rest for col in r.find_all(exp.Column)):
            return None
        if negated:
            table = amap.get(inner.table.lower())
            if table is None or _column_nullable(table, inner.name):
                return None  # NOT IN would change semantics with NULLs
        new_sub = sub.copy()
        new_sub.set("expressions", [inner.copy()])
        new_sub.set("where", exp.Where(this=_and_all([r.copy() for r in rest])) if rest else None)
        return exp.In(this=outer.copy(), query=exp.Subquery(this=new_sub))
    return None


def rule_exists_to_in(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Exists):
            res = _exists_to_in(node.this, negated=True)
            if res is not None:
                changed = True
                return exp.Not(this=res)
        elif isinstance(node, exp.Exists) and not isinstance(node.parent, exp.Not):
            res = _exists_to_in(node, negated=False)
            if res is not None:
                changed = True
                return res
        return node

    new = tree.transform(transform)
    return new if changed else None


def rule_having_to_where(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False
    for sel in list(tree.find_all(exp.Select)):
        having = sel.args.get("having")
        group = sel.args.get("group")
        if not having or not group:
            continue
        group_sql = {g.sql(dialect=dialect).lower() for g in group.expressions}
        keep, move = [], []
        for c in _conjuncts(having.this):
            cols = list(c.find_all(exp.Column))
            if (not any(isinstance(n, (exp.AggFunc, exp.Subquery, exp.Select)) for n in c.walk())
                    and cols and all(col.sql(dialect=dialect).lower() in group_sql for col in cols)):
                move.append(c.copy())
            else:
                keep.append(c.copy())
        if not move:
            continue
        where = sel.args.get("where")
        parts = ([where.this.copy()] if where else []) + move
        sel.set("where", exp.Where(this=_and_all(parts)))
        sel.set("having", exp.Having(this=_and_all(keep)) if keep else None)
        changed = True
    return tree if changed else None


def _pk_projected(sel: exp.Select, alias: str, table: str) -> bool:
    t = S.TABLES_BY_NAME.get(table)
    if t is None:
        return False
    projected = set()
    for e in sel.expressions:
        inner = e.this if isinstance(e, exp.Alias) else e
        if isinstance(inner, exp.Column) and (not inner.table or inner.table.lower() == alias):
            projected.add(inner.name.lower())
    return set(t.primary_key) <= projected


def rule_remove_redundant_distinct(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    changed = False
    for sel in list(tree.find_all(exp.Select)):
        if not sel.args.get("distinct") or sel.args.get("joins") or sel.args.get("group"):
            continue
        f = _from(sel)
        if f is None or not isinstance(f.this, exp.Table):
            continue
        alias, table = (f.this.alias_or_name or "").lower(), f.this.name.lower()
        if _pk_projected(sel, alias, table):
            sel.set("distinct", None)
            changed = True
    return tree if changed else None


def rule_distinct_join_to_exists(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    """SELECT DISTINCT a.* FROM a JOIN b ... WHERE ...  ->  SELECT a.* FROM a WHERE EXISTS (...)."""
    sel = tree if isinstance(tree, exp.Select) else None
    if sel is None or not sel.args.get("distinct") or not sel.args.get("joins") or sel.args.get("group"):
        return None
    f = _from(sel)
    if f is None or not isinstance(f.this, exp.Table):
        return None
    base_alias = (f.this.alias_or_name or "").lower()
    base_table = f.this.name.lower()
    for e in sel.expressions:
        cols = list(e.find_all(exp.Column))
        if not cols or any((c.table or "").lower() != base_alias for c in cols):
            return None
    joins = sel.args["joins"]
    if any(j.side or (j.kind and j.kind.upper() not in ("INNER",)) for j in joins):
        return None
    first = joins[0]
    if not isinstance(first.this, exp.Table) or first.args.get("on") is None:
        return None
    inner_where = [c.copy() for c in _conjuncts(first.args["on"])]
    outer_where = []
    for c in _conjuncts(sel.args["where"].this) if sel.args.get("where") else []:
        aliases = {(col.table or "").lower() for col in c.find_all(exp.Column)}
        (outer_where if aliases <= {base_alias} else inner_where).append(c.copy())
    sub = exp.select("1").from_(first.this.copy())
    for j in joins[1:]:
        sub.append("joins", j.copy())
    sub.set("where", exp.Where(this=_and_all(inner_where)))
    new = sel.copy()
    new.set("joins", None)
    new.set("where", exp.Where(this=_and_all(outer_where + [exp.Exists(this=sub)])))
    if _pk_projected(new, base_alias, base_table):
        new.set("distinct", None)
    return new


def rule_union_to_or(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    if not isinstance(tree, exp.Union) or not tree.args.get("distinct", True):
        return None
    left, right = tree.this, tree.expression
    if not (isinstance(left, exp.Select) and isinstance(right, exp.Select)):
        return None
    for s in (left, right):
        if s.args.get("joins") or s.args.get("group") or s.args.get("having") or s.args.get("distinct") \
                or s.args.get("limit") or s.args.get("fetch") or not s.args.get("where"):
            return None
    fl, fr = _from(left), _from(right)
    if not (fl and fr and isinstance(fl.this, exp.Table) and isinstance(fr.this, exp.Table)):
        return None
    if fl.this.sql() != fr.this.sql():
        return None
    if [e.sql() for e in left.expressions] != [e.sql() for e in right.expressions]:
        return None
    new = left.copy()
    new.set("where", exp.Where(this=exp.or_(exp.Paren(this=left.args["where"].this.copy()),
                                            exp.Paren(this=right.args["where"].this.copy()))))
    alias, table = (fl.this.alias_or_name or "").lower(), fl.this.name.lower()
    if not _pk_projected(new, alias, table):
        new.set("distinct", exp.Distinct())
    for arg in ("order", "limit", "fetch", "offset"):
        if tree.args.get(arg) is not None:
            new.set(arg, tree.args[arg].copy())
    return new


def rule_union_to_union_all(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
    """Naive rule: UNION -> UNION ALL avoids the sort/hash de-duplication but is only
    equivalent when the branches are disjoint and duplicate-free (not checked on purpose)."""
    changed = False
    for node in list(tree.find_all(exp.Union)):
        if node.args.get("distinct", True):
            node.set("distinct", False)
            changed = True
    return tree if changed else None


# ---- sqlglot optimizer based rules ---------------------------------------------------

def _qualified(tree: exp.Expression, dialect: str) -> exp.Expression:
    from sqlglot.optimizer.qualify import qualify

    return qualify(tree, dialect=dialect, schema=S.sqlglot_schema(), quote_identifiers=False,
                   identify=False, validate_qualify_columns=False)


def _optimizer_rule(rule_name: str) -> Rule:
    def apply(tree: exp.Expression, dialect: str) -> Optional[exp.Expression]:
        if rule_name == "eliminate_subqueries":
            from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries

            if not list(tree.find_all(exp.Subquery)):
                return None
            return eliminate_subqueries(tree)
        q = _qualified(tree, dialect)
        before = q.sql(dialect=dialect)
        if rule_name == "unnest_subqueries":
            from sqlglot.optimizer.unnest_subqueries import unnest_subqueries

            out = unnest_subqueries(q)
        elif rule_name == "merge_subqueries":
            from sqlglot.optimizer.merge_subqueries import merge_subqueries

            out = merge_subqueries(q)
        elif rule_name == "pushdown_predicates":
            from sqlglot.optimizer.pushdown_predicates import pushdown_predicates

            out = pushdown_predicates(q)
        elif rule_name == "optimize":
            from sqlglot.optimizer import RULES, optimize

            rules = [r for r in RULES if r.__name__ not in ("quote_identifiers", "qualify")]
            out = optimize(q, schema=S.sqlglot_schema(), dialect=dialect, rules=rules)
        else:  # pragma: no cover
            raise ValueError(rule_name)
        return out if out.sql(dialect=dialect) != before else None

    apply.__name__ = f"rule_{rule_name}"
    return apply


RULES: list[tuple[str, Rule, str]] = [
    ("sargable_predicates", rule_sargable_predicates,
     "Rewrites functions/arithmetic applied to columns (EXTRACT(YEAR ...), col / k) into range predicates "
     "on the bare column so indexes and partition pruning can be used."),
    ("or_to_in", rule_or_to_in, "Collapses an OR chain of equalities on one column into an IN list."),
    ("having_to_where", rule_having_to_where,
     "Moves HAVING predicates that only reference grouping columns into WHERE (filter before aggregation)."),
    ("remove_redundant_distinct", rule_remove_redundant_distinct,
     "Drops DISTINCT when the projection contains the primary key of the only table."),
    ("in_to_exists", rule_in_to_exists, "Rewrites an IN (subquery) semi-join as a correlated EXISTS."),
    ("exists_to_in", rule_exists_to_in,
     "Rewrites a correlated [NOT] EXISTS with an equality correlation as an uncorrelated [NOT] IN."),
    ("not_in_to_not_exists", rule_not_in_to_not_exists,
     "Rewrites NOT IN (subquery) as NOT EXISTS (naive rule – does not check NULL semantics)."),
    ("distinct_join_to_exists", rule_distinct_join_to_exists,
     "Replaces JOIN + DISTINCT used only for filtering with an EXISTS semi-join."),
    ("union_to_or", rule_union_to_or,
     "Merges a UNION of two filters over the same table into a single scan with OR."),
    ("union_to_union_all", rule_union_to_union_all,
     "Replaces UNION with UNION ALL to skip duplicate elimination (naive rule – assumes disjoint branches)."),
    ("subqueries_to_ctes", _optimizer_rule("eliminate_subqueries"),
     "Hoists derived tables / subqueries into WITH clauses (sqlglot eliminate_subqueries)."),
    ("decorrelate_subqueries", _optimizer_rule("unnest_subqueries"),
     "Unnests correlated subqueries into joins against grouped derived tables (sqlglot unnest_subqueries)."),
    ("merge_derived_tables", _optimizer_rule("merge_subqueries"),
     "Merges derived tables and CTEs into the outer query (sqlglot merge_subqueries)."),
    ("predicate_pushdown", _optimizer_rule("pushdown_predicates"),
     "Pushes predicates down into derived tables and join conditions (sqlglot pushdown_predicates)."),
    ("sqlglot_optimizer", _optimizer_rule("optimize"),
     "Full sqlglot logical optimizer pipeline (normalization, unnesting, merging, simplification)."),
]

_UNDERSCORE_IDENT = re.compile(r"^_+")


def _fix_identifiers(tree: exp.Expression) -> exp.Expression:
    """Oracle does not allow unquoted identifiers starting with '_' (sqlglot emits _u_0, _q_0 ...)."""
    for ident in tree.find_all(exp.Identifier):
        if not ident.quoted and ident.name.startswith("_"):
            ident.set("this", "q" + ident.name)
    return tree


def _normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


class RuleBasedRewriter(Rewriter):
    name = "rule_based"
    source = "rule"

    def __init__(self, rules: Optional[list[str]] = None, max_candidates: Optional[int] = None):
        super().__init__()
        self.rules = [r for r in RULES if rules is None or r[0] in rules]
        self.max_candidates = max_candidates

    def generate(self, request: RewriteRequest) -> list[Candidate]:
        dialect = request.dialect
        tree = sqlglot.parse_one(request.sql, read=dialect)
        seen = {_normalize(tree.sql(dialect=dialect))}
        out: list[Candidate] = []
        for strategy, fn, description in self.rules:
            try:
                new = fn(tree.copy(), dialect)
            except Exception as exc:  # a rule that does not apply cleanly is simply skipped
                self.stats.notes.append(f"{request.query.id}:{strategy}: {type(exc).__name__}: {exc}")
                continue
            if new is None:
                continue
            new = _fix_identifiers(new)
            sql = new.sql(dialect=dialect, pretty=True)
            norm = _normalize(sql)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(Candidate(id=f"R{len(out) + 1}", query_id=request.query.id, sql=sql, source="rule",
                                 strategy=strategy, rationale=description, generator="rule_based"))
            if self.max_candidates and len(out) >= self.max_candidates:
                break
        return out
