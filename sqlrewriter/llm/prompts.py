"""Prompt templates for the LLM query rewriter (Stage 3)."""

from __future__ import annotations

from .base import RepairItem, RewriteRequest

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string"},
                    "rationale": {"type": "string"},
                    "sql": {"type": "string"},
                },
                "required": ["strategy", "rationale", "sql"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a database performance engineer who rewrites SQL queries into faster, \
semantically equivalent formulations. Your rewrites are executed against a real \
database and then checked automatically: a rewrite is rejected if it fails to \
compile or if its result set differs from the original in any way.

A rewrite is correct only if it returns exactly the same result as the original \
on every possible database state that satisfies the schema constraints, not just \
on the current data. In particular:
- Preserve NULL semantics (e.g. NOT IN vs NOT EXISTS when a column is nullable; \
COUNT(*) vs COUNT(col) across outer joins; an aggregate over zero rows).
- Preserve bag semantics: duplicate rows must appear the same number of times.
- Keep the output columns identical: same number, order, names/aliases and data types.
- Keep the top-level ORDER BY semantics and any row limit (FETCH FIRST / LIMIT).
- Preserve numeric rounding behaviour: ROUND/TRUNC must be applied at the same logical point.
- Use only tables and columns that exist in the provided schema.
- Output exactly one read-only SELECT statement per candidate (WITH clauses allowed), \
no trailing semicolon, no optimizer hints, no DDL/DML.

Aim for rewrites that give the optimizer a genuinely better plan: sargable predicates, \
decorrelated subqueries, semi/anti-joins instead of join+DISTINCT, single-pass \
aggregation instead of repeated scans, window functions instead of self-referencing \
subqueries, removal of redundant operations. Use the schema's indexes, keys and row \
counts to decide what will help. Make the candidates meaningfully different from each \
other and from the original."""


def build_rewrite_prompt(req: RewriteRequest) -> str:
    parts = [
        f"Target SQL dialect: {req.dialect_label}. Every candidate must run unchanged on this engine.",
        "",
        "## Database schema and statistics",
        req.context.text,
        "",
        f"## Original query ({req.query.id}, complexity: {req.query.complexity})",
    ]
    if req.query.title and req.query.title != "Ad-hoc query":
        parts.append(f"Purpose: {req.query.title}")
    parts += ["```sql", req.sql, "```"]
    if req.original_plan:
        parts += ["", "## Current execution plan of the original query", "```", req.original_plan, "```"]
    parts += [
        "",
        f"Produce {req.num_candidates} candidate rewrites. For each give a short strategy name, "
        "a one-to-three sentence rationale explaining why it is equivalent and why it should be faster, "
        "and the complete SQL.",
        'Return JSON of the form {"candidates": [{"strategy": "...", "rationale": "...", "sql": "..."}]}.',
    ]
    return "\n".join(parts)


def build_repair_prompt(req: RewriteRequest, items: list[RepairItem]) -> str:
    parts = [
        f"Target SQL dialect: {req.dialect_label}.",
        "",
        "## Database schema and statistics",
        req.context.text,
        "",
        f"## Original query ({req.query.id})",
        "```sql",
        req.sql,
        "```",
        "",
        "The following candidate rewrites were rejected by the automatic verifier. "
        "Fix each one so that it compiles and is exactly equivalent to the original, "
        "keeping the optimization idea when possible. If an idea cannot be made equivalent, "
        "propose a different correct rewrite instead.",
    ]
    for i, item in enumerate(items, start=1):
        parts += [
            "",
            f"### Rejected candidate {i} (strategy: {item.candidate.strategy or 'n/a'})",
            "```sql",
            item.candidate.sql,
            "```",
            f"Verifier feedback: {item.problem}",
        ]
    parts += [
        "",
        f"Return exactly {len(items)} corrected candidates, in the same order, as JSON: "
        '{"candidates": [{"strategy": "...", "rationale": "...", "sql": "..."}]}.',
    ]
    return "\n".join(parts)
