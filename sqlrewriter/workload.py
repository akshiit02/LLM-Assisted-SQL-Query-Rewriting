"""Stage 1 – SQL workload loading and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from .config import PROJECT_ROOT
from .models import COMPLEXITY_LEVELS, WorkloadQuery


def load_workload(path: str | Path = "workload/queries.yaml",
                  ids: Optional[Iterable[str]] = None,
                  complexities: Optional[Iterable[str]] = None) -> list[WorkloadQuery]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    raw = data.get("queries", data if isinstance(data, list) else [])

    queries = []
    seen = set()
    for item in raw:
        q = WorkloadQuery(
            id=str(item["id"]),
            complexity=str(item["complexity"]).lower(),
            title=item.get("title", item["id"]),
            sql=item["sql"].strip().rstrip(";"),
            description=item.get("description", ""),
            tags=list(item.get("tags", [])),
        )
        if q.complexity not in COMPLEXITY_LEVELS:
            raise ValueError(f"{q.id}: unknown complexity {q.complexity!r}; expected one of {COMPLEXITY_LEVELS}")
        if q.id in seen:
            raise ValueError(f"Duplicate query id {q.id}")
        seen.add(q.id)
        queries.append(q)

    id_filter = {i.upper() for i in ids} if ids else None
    cx_filter = {c.lower() for c in complexities} if complexities else None
    return [q for q in queries
            if (id_filter is None or q.id.upper() in id_filter)
            and (cx_filter is None or q.complexity in cx_filter)]


def adhoc_query(sql: str, query_id: str = "ADHOC", complexity: Optional[str] = None) -> WorkloadQuery:
    from .complexity import classify, complexity_metrics

    sql = sql.strip().rstrip(";")
    if complexity is None:
        try:
            complexity = classify(complexity_metrics(sql))
        except Exception:
            complexity = "intermediate"
    return WorkloadQuery(id=query_id, complexity=complexity, title="Ad-hoc query", sql=sql)
