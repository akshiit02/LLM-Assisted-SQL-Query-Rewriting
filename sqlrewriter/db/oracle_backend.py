"""Oracle Database backend (the primary target of the study).

Uses ``python-oracledb`` (thin mode by default – no Oracle Client install
needed).  Implements every Oracle-specific measurement from the proposal:

* ``EXPLAIN PLAN`` + ``PLAN_TABLE``  -> optimizer cost, cardinality, CPU/IO cost
* ``DBMS_XPLAN.DISPLAY``             -> formatted estimated plan
* ``V$SQL``                          -> per-execution elapsed/CPU time, buffer gets, disk reads
* ``DBMS_XPLAN.DISPLAY_CURSOR``      -> the plan actually used at runtime
* ``DBMS_STATS.GATHER_SCHEMA_STATS`` -> fresh statistics after loads / index changes

V$ views require ``SELECT_CATALOG_ROLE`` (or ``SELECT ANY DICTIONARY``); if the
account lacks it the backend silently falls back to client-side timings.
"""

from __future__ import annotations

import time
import uuid
import warnings
from typing import Any, Optional

from .. import schema as S
from ..config import OracleSettings
from ..datagen import Dataset
from ..models import PlanOperation
from .base import DatabaseBackend, PlanInfo, QueryError, QueryTimeout, ResultSet, RunSample

try:  # optional dependency
    import oracledb
except ImportError:  # pragma: no cover - reported at activation time
    oracledb = None  # type: ignore[assignment]


def _is_missing_object(exc: Exception) -> bool:
    return "ORA-00942" in str(exc) or "ORA-01418" in str(exc)


class OracleBackend(DatabaseBackend):
    name = "oracle"
    dialect = "oracle"
    display_name = "Oracle Database"

    def __init__(self, settings: OracleSettings):
        if oracledb is None:
            raise RuntimeError("python-oracledb is not installed: pip install oracledb")
        if not settings.user or not settings.dsn:
            raise RuntimeError(
                "Oracle connection not configured. Set ORACLE_USER, ORACLE_PASSWORD and ORACLE_DSN "
                "(see .env.example) or edit the 'database.oracle' section of your config."
            )
        self.settings = settings
        self.conn = None
        self.scale: Optional[float] = None
        self._runtime_stats_ok = settings.collect_runtime_stats
        if settings.thick_mode:
            oracledb.init_oracle_client(lib_dir=settings.lib_dir or None)

    # ------------------------------------------------------------------ lifecycle
    def _connect(self) -> None:
        self.conn = oracledb.connect(user=self.settings.user, password=self.settings.password,
                                     dsn=self.settings.dsn)
        self.conn.autocommit = False

    def activate(self, scale: float) -> None:
        # Oracle keeps one schema; the data for ``scale`` is (re)loaded by ensure_dataset.
        if self.conn is None:
            self._connect()
        self.scale = scale

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

    def _cur(self):
        if self.conn is None:
            self._connect()
        return self.conn.cursor()

    def _ensure_alive(self) -> None:
        try:
            self.conn.ping()
        except Exception:
            self.close()
            self._connect()

    # ------------------------------------------------------------------ setup
    def dataset_info(self) -> Optional[dict]:
        cur = self._cur()
        try:
            cur.execute("SELECT k, v FROM sqlrw_meta")
            return dict(cur.fetchall()) or None
        except oracledb.DatabaseError as exc:
            if _is_missing_object(exc):
                return None
            raise

    def create_schema(self) -> None:
        cur = self._cur()
        for name in list(reversed(S.LOAD_ORDER)) + ["sqlrw_meta"]:
            try:
                cur.execute(f"DROP TABLE {name} CASCADE CONSTRAINTS PURGE")
            except oracledb.DatabaseError as exc:
                if not _is_missing_object(exc):
                    raise
        for ddl in S.schema_ddl("oracle"):
            cur.execute(ddl)
        cur.execute("CREATE TABLE sqlrw_meta (k VARCHAR2(30) PRIMARY KEY, v VARCHAR2(100))")

    def _input_sizes(self, table: S.Table) -> list[Any]:
        sizes: list[Any] = []
        for col in table.columns:
            t = col.type.upper()
            if t.startswith("VARCHAR"):
                sizes.append(int(t[t.index("(") + 1:t.index(")")]))
            elif t == "DATE":
                sizes.append(oracledb.DB_TYPE_DATE)
            else:
                sizes.append(oracledb.DB_TYPE_NUMBER)
        return sizes

    def load_dataset(self, dataset: Dataset, batch_size: int = 5000) -> None:
        cur = self._cur()
        for name in S.LOAD_ORDER:
            table = S.TABLES_BY_NAME[name]
            rows = dataset.tables[name]
            binds = ", ".join(f":{i + 1}" for i in range(len(table.columns)))
            sql = f"INSERT /*+ APPEND_VALUES */ INTO {name} VALUES ({binds})"
            for i in range(0, len(rows), batch_size):
                cur.setinputsizes(*self._input_sizes(table))
                cur.executemany(sql, rows[i:i + batch_size])
                self.conn.commit()  # APPEND_VALUES requires commit before next DML on table
        cur.execute("DELETE FROM sqlrw_meta")
        cur.executemany("INSERT INTO sqlrw_meta VALUES (:1, :2)",
                        [("scale", str(dataset.scale)), ("seed", str(dataset.seed))])
        self.conn.commit()

    def current_indexes(self) -> list[str]:
        cur = self._cur()
        cur.execute("SELECT LOWER(index_name) FROM user_indexes "
                    "WHERE index_name LIKE 'IX\\_%' ESCAPE '\\' ORDER BY index_name")
        return [r[0] for r in cur.fetchall()]

    def apply_index_configuration(self, config_name: str) -> None:
        cfg = S.INDEX_CONFIGURATIONS[config_name]
        cur = self._cur()
        wanted = {ix.name for ix in cfg.indexes}
        for existing in self.current_indexes():
            if existing not in wanted:
                cur.execute(f"DROP INDEX {existing}")
        present = set(self.current_indexes())
        for ix in cfg.indexes:
            if ix.name not in present:
                cur.execute(S.create_index_ddl(ix))
        self.gather_statistics()

    def gather_statistics(self) -> None:
        cur = self._cur()
        cur.execute("BEGIN DBMS_STATS.GATHER_SCHEMA_STATS(ownname => USER, cascade => TRUE, "
                    "no_invalidate => FALSE); END;")

    # ------------------------------------------------------------------ SQL
    def prepare_sql(self, oracle_sql: str) -> str:
        return oracle_sql.strip().rstrip(";").strip()

    def check_executable(self, sql: str) -> Optional[str]:
        cur = self._cur()
        try:
            cur.parse(sql)  # hard parse only – nothing is executed
            return None
        except oracledb.DatabaseError as exc:
            return str(exc).strip()

    def _begin_read_only(self, cur) -> None:
        self.conn.rollback()
        cur.execute("SET TRANSACTION READ ONLY")

    def execute(self, sql: str, max_rows: int, timeout_s: float) -> ResultSet:
        self._ensure_alive()
        cur = self._cur()
        cur.arraysize = 5000
        cur.prefetchrows = 5001
        self.conn.call_timeout = int(timeout_s * 1000)
        try:
            self._begin_read_only(cur)
            cur.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
            desc = cur.description or []
        except oracledb.DatabaseError as exc:
            self._handle_error(exc, timeout_s)
            raise  # unreachable
        finally:
            self._finish_call()
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        columns = [d.name for d in desc]
        types = [_type_category(d.type_code) for d in desc]
        return ResultSet(columns, types, [tuple(r) for r in rows], truncated)

    def _finish_call(self) -> None:
        if self.conn is None:
            return
        try:
            self.conn.call_timeout = 0
            self.conn.rollback()  # ends the read-only transaction
        except Exception:
            self.close()

    def _handle_error(self, exc: Exception, timeout_s: float) -> None:
        msg = str(exc)
        if "DPI-1067" in msg or "ORA-01013" in msg or "DPI-1080" in msg:
            self._ensure_alive_after_timeout()
            raise QueryTimeout(f"statement exceeded {timeout_s:g}s") from exc
        raise QueryError(msg.strip()) from exc

    def _ensure_alive_after_timeout(self) -> None:
        try:
            self.conn.call_timeout = 0
            self.conn.ping()
        except Exception:
            self.close()
            self._connect()

    def _tag_sql(self, sql: str, tag: str) -> str:
        return f"/* sqlrw:{tag} */ {sql}"

    def _run_once(self, sql: str, timeout_s: float) -> RunSample:
        self._ensure_alive()
        cur = self._cur()
        cur.arraysize = 5000
        cur.prefetchrows = 5001
        if self.settings.flush_buffer_cache:
            try:
                cur.execute("ALTER SYSTEM FLUSH BUFFER_CACHE")
            except oracledb.DatabaseError as exc:
                warnings.warn(f"Cannot flush buffer cache ({exc}); disabling flush_buffer_cache")
                self.settings.flush_buffer_cache = False
        self.conn.call_timeout = int(timeout_s * 1000)
        try:
            self._begin_read_only(cur)
            t0 = time.perf_counter()
            cur.execute(sql)
            n = 0
            while True:
                chunk = cur.fetchmany()
                if not chunk:
                    break
                n += len(chunk)
            elapsed = (time.perf_counter() - t0) * 1000.0
        except oracledb.DatabaseError as exc:
            self._handle_error(exc, timeout_s)
            raise
        finally:
            self._finish_call()
        return RunSample(elapsed, n)

    def explain(self, sql: str) -> PlanInfo:
        sid = "SQLRW_" + uuid.uuid4().hex[:20].upper()
        cur = self._cur()
        cur.execute(f"EXPLAIN PLAN SET STATEMENT_ID = '{sid}' FOR {sql}")
        cur.execute(
            "SELECT id, parent_id, depth, operation, options, object_name, cost, cardinality, "
            "cpu_cost, io_cost FROM plan_table WHERE statement_id = :sid ORDER BY id",
            sid=sid,
        )
        ops: list[PlanOperation] = []
        root = None
        for (op_id, parent, depth, operation, options, obj, cost, card, cpu, io) in cur.fetchall():
            ops.append(PlanOperation(id=op_id, parent_id=parent, operation=operation or "",
                                     options=options or "", object_name=obj or "",
                                     cost=_f(cost), cardinality=_f(card), depth=depth or 0))
            if op_id == 0:
                root = (cost, card, cpu, io)
        cur.execute("SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY('PLAN_TABLE', :sid, 'TYPICAL'))",
                    sid=sid)
        text = "\n".join(r[0] or "" for r in cur.fetchall())
        cur.execute("DELETE FROM plan_table WHERE statement_id = :sid", sid=sid)
        self.conn.commit()
        cost, card, cpu, io = root or (None, None, None, None)
        return PlanInfo(plan_text="Estimated plan (EXPLAIN PLAN / DBMS_XPLAN.DISPLAY)\n" + text,
                        operations=ops, cost=_f(cost), cardinality=_f(card),
                        cpu_cost=_f(cpu), io_cost=_f(io))

    def _runtime_statistics(self, sql: str) -> dict[str, Any]:
        if not self._runtime_stats_ok:
            return {}
        prefix = sql[: sql.index("*/") + 2] if sql.startswith("/* sqlrw:") else sql[:60]
        cur = self._cur()
        try:
            cur.execute(
                "SELECT sql_id, child_number, executions, elapsed_time, cpu_time, buffer_gets, "
                "disk_reads FROM v$sql WHERE sql_text LIKE :pat ESCAPE '\\' ORDER BY last_active_time DESC",
                pat=prefix.replace("%", "\\%").replace("_", "\\_") + "%",
            )
            rows = cur.fetchall()
        except oracledb.DatabaseError as exc:
            if _is_missing_object(exc) or "ORA-01031" in str(exc):
                warnings.warn("No access to V$SQL – runtime CPU/IO statistics disabled. "
                              "Grant SELECT_CATALOG_ROLE to the benchmark user to enable them.")
                self._runtime_stats_ok = False
                return {}
            raise
        if not rows:
            return {}
        execs = sum(r[2] or 0 for r in rows) or 1
        stats = {
            "db_elapsed_ms": sum(r[3] or 0 for r in rows) / execs / 1000.0,
            "cpu_time_ms": sum(r[4] or 0 for r in rows) / execs / 1000.0,
            "buffer_gets": sum(r[5] or 0 for r in rows) / execs,
            "disk_reads": sum(r[6] or 0 for r in rows) / execs,
        }
        try:
            cur.execute("SELECT plan_table_output FROM TABLE(DBMS_XPLAN.DISPLAY_CURSOR(:sid, :child, 'TYPICAL'))",
                        sid=rows[0][0], child=rows[0][1])
            stats["runtime_plan"] = ("Runtime plan (DBMS_XPLAN.DISPLAY_CURSOR)\n"
                                     + "\n".join(r[0] or "" for r in cur.fetchall()))
        except oracledb.DatabaseError:
            pass
        return stats

    def schema_metadata(self) -> dict:
        cur = self._cur()
        names = [t.upper() for t in S.LOAD_ORDER]
        placeholders = ", ".join(f":{i + 1}" for i in range(len(names)))
        cur.execute(
            f"SELECT table_name, column_name, data_type, data_length, data_precision, data_scale, nullable "
            f"FROM user_tab_columns WHERE table_name IN ({placeholders}) ORDER BY table_name, column_id",
            names,
        )
        columns: dict[str, list] = {}
        for tname, cname, dtype, length, prec, scale, nullable in cur.fetchall():
            if dtype == "NUMBER" and prec is not None:
                dtype = f"NUMBER({prec},{scale})" if scale else f"NUMBER({prec})"
            elif dtype in ("VARCHAR2", "CHAR"):
                dtype = f"{dtype}({length})"
            columns.setdefault(tname, []).append({"name": cname.lower(), "type": dtype, "nullable": nullable == "Y"})

        cur.execute(
            f"SELECT c.table_name, c.constraint_name, c.constraint_type, cc.column_name, cc.position, "
            f"r.table_name, rc.column_name "
            f"FROM user_constraints c JOIN user_cons_columns cc ON cc.constraint_name = c.constraint_name "
            f"LEFT JOIN user_constraints r ON r.constraint_name = c.r_constraint_name "
            f"LEFT JOIN user_cons_columns rc ON rc.constraint_name = c.r_constraint_name AND rc.position = cc.position "
            f"WHERE c.constraint_type IN ('P', 'R') AND c.table_name IN ({placeholders}) "
            f"ORDER BY c.table_name, c.constraint_name, cc.position",
            names,
        )
        pks: dict[str, list] = {}
        fks: dict[str, dict] = {}
        for tname, cons, ctype, col, _pos, rtable, rcol in cur.fetchall():
            if ctype == "P":
                pks.setdefault(tname, []).append(col.lower())
            else:
                fk = fks.setdefault(cons, {"table": tname, "columns": [], "ref_table": (rtable or "").lower(),
                                           "ref_columns": []})
                fk["columns"].append(col.lower())
                fk["ref_columns"].append((rcol or "").lower())

        cur.execute(
            f"SELECT i.table_name, i.index_name, i.uniqueness, ic.column_name "
            f"FROM user_indexes i JOIN user_ind_columns ic ON ic.index_name = i.index_name "
            f"WHERE i.table_name IN ({placeholders}) ORDER BY i.table_name, i.index_name, ic.column_position",
            names,
        )
        indexes: dict[str, dict] = {}
        for tname, iname, uniq, col in cur.fetchall():
            ix = indexes.setdefault(iname, {"table": tname, "name": iname.lower(), "columns": [],
                                            "unique": uniq == "UNIQUE"})
            ix["columns"].append(col.lower())

        cur.execute(f"SELECT table_name, num_rows, blocks FROM user_tables WHERE table_name IN ({placeholders})", names)
        stats = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

        tables = []
        for t in S.TABLES:
            up = t.name.upper()
            if up not in columns:
                continue
            tables.append({
                "name": t.name,
                "description": t.description,
                "row_count": stats.get(up, (None, None))[0],
                "blocks": stats.get(up, (None, None))[1],
                "columns": columns[up],
                "primary_key": pks.get(up, []),
                "foreign_keys": [{k: v for k, v in fk.items() if k != "table"} for fk in fks.values() if fk["table"] == up],
                "indexes": [{k: v for k, v in ix.items() if k != "table"} for ix in indexes.values() if ix["table"] == up],
            })
        return {"backend": self.name, "dialect": "Oracle SQL", "version": self.conn.version,
                "tables": tables, "notes": ["Row counts come from optimizer statistics (USER_TABLES.NUM_ROWS)."]}


def _f(v) -> Optional[float]:
    return None if v is None else float(v)


def _type_category(type_code) -> str:
    if oracledb is None:
        return "unknown"
    if type_code in (oracledb.DB_TYPE_NUMBER, oracledb.DB_TYPE_BINARY_DOUBLE, oracledb.DB_TYPE_BINARY_FLOAT,
                     oracledb.DB_TYPE_BINARY_INTEGER):
        return "number"
    if type_code in (oracledb.DB_TYPE_VARCHAR, oracledb.DB_TYPE_CHAR, oracledb.DB_TYPE_NVARCHAR,
                     oracledb.DB_TYPE_NCHAR, oracledb.DB_TYPE_CLOB, oracledb.DB_TYPE_NCLOB, oracledb.DB_TYPE_LONG):
        return "text"
    if type_code in (oracledb.DB_TYPE_DATE, oracledb.DB_TYPE_TIMESTAMP, oracledb.DB_TYPE_TIMESTAMP_TZ,
                     oracledb.DB_TYPE_TIMESTAMP_LTZ):
        return "date"
    if type_code in (oracledb.DB_TYPE_RAW, oracledb.DB_TYPE_BLOB, oracledb.DB_TYPE_LONG_RAW):
        return "binary"
    return "unknown"
