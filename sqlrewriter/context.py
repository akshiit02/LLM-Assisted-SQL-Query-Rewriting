"""Stage 2 – Database & schema context packaging.

Reads the live catalog of the connected database (tables, columns, types,
nullability, primary / foreign keys, indexes, row counts) and renders it as a
compact text block for the LLM prompt.  Grounding the model in the real
catalog prevents hallucinated columns and indexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .db.base import DatabaseBackend
from .schema import INDEX_CONFIGURATIONS


@dataclass
class SchemaContext:
    backend: str
    dialect: str
    scale: float
    index_config: str
    text: str
    metadata: dict


def build_schema_context(backend: DatabaseBackend, scale: float, index_config: str,
                         include_row_counts: bool = True) -> SchemaContext:
    meta = backend.schema_metadata()
    lines = [
        f"Database engine: {backend.display_name} ({meta.get('dialect')}, version {meta.get('version')})",
        f"Dataset scale factor: {scale:g}",
    ]
    cfg = INDEX_CONFIGURATIONS.get(index_config)
    if cfg:
        lines.append(f"Index configuration: {cfg.name} – {cfg.description}")
    for note in meta.get("notes", []):
        lines.append(f"Note: {note}")
    lines.append("")
    for t in meta["tables"]:
        header = f"TABLE {t['name']}"
        if include_row_counts and t.get("row_count") is not None:
            header += f"  (~{int(t['row_count']):,} rows)"
        lines.append(header)
        if t.get("description"):
            lines.append(f"  -- {t['description']}")
        for col in t["columns"]:
            null = "NULL" if col["nullable"] else "NOT NULL"
            lines.append(f"  {col['name']:<20} {col['type']:<16} {null}")
        if t.get("primary_key"):
            lines.append(f"  PRIMARY KEY ({', '.join(t['primary_key'])})")
        for fk in t.get("foreign_keys", []):
            lines.append(f"  FOREIGN KEY ({', '.join(fk['columns'])}) REFERENCES "
                         f"{fk['ref_table']} ({', '.join(fk['ref_columns'])})")
        secondary = [ix for ix in t.get("indexes", []) if "primary key" not in ix["name"].lower()
                     and not ix["name"].lower().startswith("pk_") and not ix["name"].lower().startswith("sys_")]
        for ix in secondary:
            uniq = "UNIQUE " if ix.get("unique") else ""
            lines.append(f"  {uniq}INDEX {ix['name']} ({', '.join(ix['columns'])})")
        lines.append("")
    return SchemaContext(backend=backend.name, dialect=backend.dialect, scale=scale,
                         index_config=index_config, text="\n".join(lines).rstrip(), metadata=meta)


def summarize_plan(plan_text: str, max_lines: int = 40) -> Optional[str]:
    if not plan_text:
        return None
    lines = [ln for ln in plan_text.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (truncated)"]
    return "\n".join(lines)
