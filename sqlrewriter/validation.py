"""Stage 4 – Query validation: safety, syntax and executability gates.

1. **Safety**   – the candidate must be exactly one read-only query
                   (SELECT / WITH ... SELECT / set operation).  DML, DDL,
                   PL/SQL, SELECT ... INTO and FOR UPDATE are rejected before
                   anything touches the database.
2. **Syntax**   – parsed with sqlglot in the backend's dialect.  Because a
                   parser can lag behind a real engine, a sqlglot failure is
                   *not* fatal on its own: the database parser is the final
                   judge (step 3) and the fallback safety check is lexical.
3. **Executability** – the database compiles the statement without running
                   it (Oracle: ``cursor.parse``; SQLite: ``EXPLAIN``).
"""

from __future__ import annotations

import re
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from .db.base import DatabaseBackend
from .models import ValidationResult

_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Into, exp.Lock, exp.TruncateTable, exp.Grant,
)
_FORBIDDEN_WORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|EXECUTE|EXEC|BEGIN|DECLARE|"
    r"CALL|COMMIT|ROLLBACK|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|RENAME|LOCK)\b",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def clean_sql(sql: str) -> str:
    """Strip markdown fences, surrounding whitespace and trailing semicolons."""
    s = sql.strip()
    s = _FENCE.sub("", s).strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    if s.endswith("/") and "\n" in s:  # SQL*Plus terminator
        s = s[:-1].rstrip()
    return s


def _strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql


def lexical_safety_check(sql: str) -> Optional[str]:
    body = _strip_literals_and_comments(sql).strip()
    if ";" in body:
        return "multiple statements are not allowed"
    first = body.split(None, 1)[0].upper() if body else ""
    if first not in ("SELECT", "WITH", "("):
        return f"statement must start with SELECT or WITH (found {first or 'nothing'})"
    m = _FORBIDDEN_WORDS.search(body)
    if m:
        return f"forbidden keyword {m.group(1).upper()} in candidate"
    if re.search(r"\bFOR\s+UPDATE\b", body, re.IGNORECASE):
        return "SELECT ... FOR UPDATE is not allowed"
    return None


def categorize_error(message: str) -> str:
    m = message.lower()
    rules = [
        ("timeout", ("dpi-1067", "exceeded", "ora-01013", "interrupted")),
        ("invalid_identifier", ("ora-00904", "no such column")),
        ("missing_table", ("ora-00942", "no such table")),
        ("ambiguous_column", ("ora-00918", "ambiguous column")),
        ("group_by_error", ("ora-00979", "ora-00937", "ora-00934", "misuse of aggregate",
                            "not a group by", "must appear in the group by")),
        ("type_error", ("ora-00932", "ora-01722", "ora-01843", "ora-01861", "datatype mismatch")),
        ("unsupported_function", ("no such function", "ora-00406", "ora-00904: \"")),
        ("syntax_error", ("syntax error", "ora-009", "ora-00933", "ora-00936", "ora-00907",
                          "ora-00923", "ora-00921", "incomplete input", "parse")),
        ("unsafe_statement", ("forbidden", "not allowed", "must start with")),
    ]
    for category, needles in rules:
        if any(n in m for n in needles):
            return category
    return "other"


def validate_candidate(sql: str, backend: DatabaseBackend) -> ValidationResult:
    sql = clean_sql(sql)
    if not sql:
        return ValidationResult(False, "safety", "empty candidate", "unsafe_statement")

    # ---- 1+2: safety & syntax via AST -----------------------------------
    parse_error = None
    try:
        statements = [s for s in sqlglot.parse(sql, read=backend.dialect) if s is not None]
    except ParseError as exc:
        statements = None
        parse_error = str(exc).splitlines()[0]
    except Exception as exc:  # tokenizer errors etc.
        statements = None
        parse_error = f"{type(exc).__name__}: {exc}"

    if statements is not None:
        if len(statements) != 1:
            return ValidationResult(False, "safety", "exactly one statement is required", "unsafe_statement")
        root = statements[0]
        if not isinstance(root, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
            return ValidationResult(False, "safety",
                                    f"only read-only queries are allowed (got {type(root).__name__})",
                                    "unsafe_statement")
        for node in root.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                return ValidationResult(False, "safety",
                                        f"forbidden construct {type(node).__name__} in candidate",
                                        "unsafe_statement")
    # Lexical check always runs (defence in depth, and the only check if sqlglot failed)
    lex = lexical_safety_check(sql)
    if lex:
        return ValidationResult(False, "safety", lex, "unsafe_statement")

    # ---- 3: executability (database is authoritative for syntax) ---------
    db_error = backend.check_executable(sql)
    if db_error:
        stage = "syntax" if categorize_error(db_error) == "syntax_error" else "executable"
        return ValidationResult(False, stage, db_error, categorize_error(db_error))
    if parse_error:
        # DB accepted what sqlglot could not parse – still valid, keep a note.
        return ValidationResult(True, "ok", f"note: sqlglot could not parse ({parse_error})", None)
    return ValidationResult(True, "ok")
