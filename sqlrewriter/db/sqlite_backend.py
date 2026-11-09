"""SQLite backend – zero-install fallback so the whole pipeline runs anywhere.

SQLite has no cost-based optimizer cost numbers, so ``optimizer_cost`` is
reported as ``None``; instead the backend records *VM steps* (a deterministic
work counter obtained through the progress handler) and process CPU time.
The research study itself targets Oracle (see ``oracle_backend.py``).
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import sqlglot
from sqlglot import exp

from .. import schema as S
from ..config import PROJECT_ROOT, SQLiteSettings
from ..datagen import Dataset
from ..models import PlanOperation
from .base import DatabaseBackend, PlanInfo, QueryError, QueryTimeout, ResultSet, RunSample, render_plan_tree

_PROGRESS_STEP = 1000  # VM instructions between progress-handler callbacks

_EXTRACT_FORMATS = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d", "HOUR": "%H", "MINUTE": "%M"}


def _oracle_to_sqlite(node: exp.Expression) -> exp.Expression:
    """AST fix-ups for Oracle constructs sqlglot cannot emit for SQLite."""
    if isinstance(node, exp.Extract):
        unit = node.this.name.upper() if node.this is not None else ""
        fmt = _EXTRACT_FORMATS.get(unit)
        if fmt:
            return exp.Cast(
                this=exp.func("STRFTIME", exp.Literal.string(fmt), node.expression),
                to=exp.DataType.build("INTEGER"),
            )
    return node


class SQLiteBackend(DatabaseBackend):
    name = "sqlite"
    dialect = "sqlite"
    display_name = "SQLite"

    def __init__(self, settings: SQLiteSettings):
        self.settings = settings
        self.conn: Optional[sqlite3.Connection] = None
        self.scale: Optional[float] = None
        self.path: Optional[Path] = None

    # ------------------------------------------------------------------ lifecycle
    def db_path(self, scale: float) -> Path:
        p = Path(self.settings.path_template.format(scale=f"{scale:g}"))
        return p if p.is_absolute() else PROJECT_ROOT / p

    def activate(self, scale: float) -> None:
        if self.conn is not None and self.scale == scale:
            return
        self.close()
        self.path = self.db_path(scale)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA cache_size = -65536")  # 64 MB page cache
        self.scale = scale

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = None
        self.scale = None

    def _c(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("SQLite backend not activated – call activate(scale) first")
        return self.conn

    # ------------------------------------------------------------------ setup
    def dataset_info(self) -> Optional[dict]:
        try:
            rows = self._c().execute("SELECT key, value FROM sqlrw_meta").fetchall()
        except sqlite3.OperationalError:
            return None
        return dict(rows) or None

    def create_schema(self) -> None:
        c = self._c()
        c.execute("PRAGMA query_only = 0")
        for name in reversed(S.LOAD_ORDER):
            c.execute(f"DROP TABLE IF EXISTS {name}")
        c.execute("DROP TABLE IF EXISTS sqlrw_meta")
        for ddl in S.schema_ddl("sqlite"):
            c.execute(ddl)
        c.execute("CREATE TABLE sqlrw_meta (key TEXT PRIMARY KEY, value TEXT)")

    def load_dataset(self, dataset: Dataset, batch_size: int = 5000) -> None:
        c = self._c()
        c.execute("BEGIN")
        try:
            for name in S.LOAD_ORDER:
                rows = dataset.tables[name]
                ncols = len(S.TABLES_BY_NAME[name].columns)
                sql = f"INSERT INTO {name} VALUES ({', '.join('?' * ncols)})"
                for i in range(0, len(rows), batch_size):
                    batch = [tuple(v.isoformat() if isinstance(v, dt.date) else v for v in r)
                             for r in rows[i:i + batch_size]]
                    c.executemany(sql, batch)
            c.execute("DELETE FROM sqlrw_meta")
            c.executemany("INSERT INTO sqlrw_meta VALUES (?, ?)",
                          [("scale", str(dataset.scale)), ("seed", str(dataset.seed))])
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def current_indexes(self) -> list[str]:
        rows = self._c().execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'ix\\_%' ESCAPE '\\' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def apply_index_configuration(self, config_name: str) -> None:
        cfg = S.INDEX_CONFIGURATIONS[config_name]
        c = self._c()
        c.execute("PRAGMA query_only = 0")
        wanted = {ix.name for ix in cfg.indexes}
        for existing in self.current_indexes():
            if existing not in wanted:
                c.execute(f"DROP INDEX {existing}")
        present = set(self.current_indexes())
        for ix in cfg.indexes:
            if ix.name not in present:
                c.execute(S.create_index_ddl(ix))
        self.gather_statistics()

    def gather_statistics(self) -> None:
        c = self._c()
        c.execute("PRAGMA query_only = 0")
        c.execute("ANALYZE")

    # ------------------------------------------------------------------ SQL
    def prepare_sql(self, oracle_sql: str) -> str:
        tree = sqlglot.parse_one(oracle_sql, read="oracle")
        tree = tree.transform(_oracle_to_sqlite)
        return tree.sql(dialect="sqlite", pretty=True)

    def check_executable(self, sql: str) -> Optional[str]:
        try:
            self._c().execute("EXPLAIN " + sql).fetchall()
            return None
        except sqlite3.Error as exc:
            return f"{type(exc).__name__}: {exc}"

    def _install_timeout(self, timeout_s: float) -> dict:
        state = {"steps": 0, "deadline": time.perf_counter() + timeout_s, "timed_out": False}

        def handler() -> int:
            state["steps"] += 1
            if time.perf_counter() > state["deadline"]:
                state["timed_out"] = True
                return 1  # abort statement
            return 0

        self._c().set_progress_handler(handler, _PROGRESS_STEP)
        return state

    def _clear_timeout(self) -> None:
        self._c().set_progress_handler(None, _PROGRESS_STEP)

    def execute(self, sql: str, max_rows: int, timeout_s: float) -> ResultSet:
        c = self._c()
        c.execute("PRAGMA query_only = 1")
        state = self._install_timeout(timeout_s)
        try:
            cur = c.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            columns = [d[0] for d in (cur.description or [])]
        except sqlite3.OperationalError as exc:
            if state["timed_out"]:
                raise QueryTimeout(f"statement exceeded {timeout_s:g}s") from exc
            raise QueryError(str(exc)) from exc
        except sqlite3.Error as exc:
            raise QueryError(str(exc)) from exc
        finally:
            self._clear_timeout()
            c.execute("PRAGMA query_only = 0")
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return ResultSet(columns, _infer_types(rows, len(columns)), rows, truncated)

    def _run_once(self, sql: str, timeout_s: float) -> RunSample:
        c = self._c()
        c.execute("PRAGMA query_only = 1")
        state = self._install_timeout(timeout_s)
        cpu0 = time.process_time()
        t0 = time.perf_counter()
        try:
            cur = c.execute(sql)
            n = 0
            while True:
                chunk = cur.fetchmany(5000)
                if not chunk:
                    break
                n += len(chunk)
        except sqlite3.OperationalError as exc:
            if state["timed_out"]:
                raise QueryTimeout(f"statement exceeded {timeout_s:g}s") from exc
            raise QueryError(str(exc)) from exc
        finally:
            self._clear_timeout()
            c.execute("PRAGMA query_only = 0")
        elapsed = (time.perf_counter() - t0) * 1000.0
        cpu = (time.process_time() - cpu0) * 1000.0
        return RunSample(elapsed, n, cpu_ms=cpu, work_units=float(state["steps"] * _PROGRESS_STEP))

    def explain(self, sql: str) -> PlanInfo:
        rows = self._c().execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        depth: dict[int, int] = {0: -1}
        ops = []
        for node_id, parent, _unused, detail in rows:
            d = depth.get(parent, -1) + 1
            depth[node_id] = d
            ops.append(PlanOperation(id=node_id, parent_id=parent or None, operation=detail,
                                     depth=max(d, 0)))
        text = "SQLite EXPLAIN QUERY PLAN\n" + render_plan_tree(ops)
        return PlanInfo(plan_text=text, operations=ops)

    def schema_metadata(self) -> dict:
        c = self._c()
        tables = []
        for t in S.TABLES:
            cols = c.execute(f"PRAGMA table_info({t.name})").fetchall()
            if not cols:
                continue
            fks = {}
            for row in c.execute(f"PRAGMA foreign_key_list({t.name})").fetchall():
                fid, _seq, ref_table, from_col, to_col = row[0], row[1], row[2], row[3], row[4]
                fk = fks.setdefault(fid, {"columns": [], "ref_table": ref_table, "ref_columns": []})
                fk["columns"].append(from_col)
                fk["ref_columns"].append(to_col)
            indexes = []
            for ix in c.execute(f"PRAGMA index_list({t.name})").fetchall():
                ix_name, unique, origin = ix[1], bool(ix[2]), ix[3]
                ix_cols = [r[2] for r in c.execute(f"PRAGMA index_info('{ix_name}')").fetchall()]
                indexes.append({"name": ix_name if origin == "c" else f"{ix_name} (primary key)",
                                "columns": ix_cols, "unique": unique})
            count = c.execute(f"SELECT COUNT(*) FROM {t.name}").fetchone()[0]
            tables.append({
                "name": t.name,
                "description": t.description,
                "row_count": count,
                "columns": [{"name": col[1], "type": col[2], "nullable": not col[3]} for col in cols],
                "primary_key": [col[1] for col in sorted(cols, key=lambda x: x[5]) if col[5]],
                "foreign_keys": list(fks.values()),
                "indexes": indexes,
            })
        return {"backend": self.name, "dialect": "SQLite 3", "version": sqlite3.sqlite_version,
                "tables": tables, "notes": ["DATE columns are stored as ISO-8601 TEXT ('YYYY-MM-DD').",
                                            "Use STRFTIME for date part extraction."]}


def _infer_types(rows: list[tuple], ncols: int) -> list[str]:
    types = ["unknown"] * ncols
    for i in range(ncols):
        for r in rows:
            v = r[i]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                types[i] = "number"
            elif isinstance(v, str):
                types[i] = "text"
            elif isinstance(v, bytes):
                types[i] = "binary"
            break
    return types


def remove_database_files(settings: SQLiteSettings, scale: float) -> None:
    backend = SQLiteBackend(settings)
    path = backend.db_path(scale)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            os.remove(p)
