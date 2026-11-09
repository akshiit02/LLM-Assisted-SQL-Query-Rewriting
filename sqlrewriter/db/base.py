"""Abstract database backend used by every pipeline stage."""

from __future__ import annotations

import hashlib
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..datagen import Dataset
from ..models import PerformanceMetrics, PlanOperation


class QueryTimeout(Exception):
    """Raised when a statement exceeds the configured time budget."""


class QueryError(Exception):
    """Raised when a statement fails to compile or execute."""


@dataclass
class ResultSet:
    columns: list[str]
    column_types: list[str]  # normalised categories: number | text | date | binary | unknown
    rows: list[tuple]
    truncated: bool = False


@dataclass
class PlanInfo:
    plan_text: str
    operations: list[PlanOperation] = field(default_factory=list)
    cost: Optional[float] = None
    cardinality: Optional[float] = None
    cpu_cost: Optional[float] = None
    io_cost: Optional[float] = None

    @property
    def signature(self) -> str:
        return plan_signature(self.operations)


@dataclass
class RunSample:
    elapsed_ms: float
    rows: int
    cpu_ms: Optional[float] = None
    work_units: Optional[float] = None


def plan_signature(ops: list[PlanOperation]) -> str:
    """Stable fingerprint of the *shape* of a plan (operators + objects)."""
    text = "|".join(f"{o.depth}:{o.label()}" for o in ops)
    return hashlib.sha1(text.encode()).hexdigest()[:12] if text else ""


class DatabaseBackend(ABC):
    name: str = "abstract"
    dialect: str = "ansi"  # sqlglot dialect name
    display_name: str = "Database"

    # ------------------------------------------------------------------ lifecycle
    @abstractmethod
    def activate(self, scale: float) -> None:
        """Connect to (or switch to) the database holding data at ``scale``."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ setup
    @abstractmethod
    def dataset_info(self) -> Optional[dict]:
        """Return {'scale':..., 'seed':...} of loaded data, or None."""

    @abstractmethod
    def create_schema(self) -> None: ...

    @abstractmethod
    def load_dataset(self, dataset: Dataset, batch_size: int = 5000) -> None: ...

    def ensure_dataset(self, scale: float, seed: int, batch_size: int = 5000, force: bool = False,
                       progress=None) -> bool:
        """Load data for ``scale`` unless it is already present. Returns True if loaded."""
        from ..datagen import generate

        self.activate(scale)
        info = self.dataset_info()
        if not force and info and float(info.get("scale", -1)) == float(scale) and int(info.get("seed", -1)) == int(seed):
            return False
        if progress:
            progress(f"Generating synthetic data (scale={scale:g}, seed={seed}) ...")
        dataset = generate(scale, seed)
        if progress:
            counts = ", ".join(f"{k}={v}" for k, v in dataset.row_counts().items())
            progress(f"Creating schema and loading rows: {counts}")
        self.create_schema()
        self.load_dataset(dataset, batch_size)
        self.gather_statistics()
        return True

    @abstractmethod
    def current_indexes(self) -> list[str]: ...

    @abstractmethod
    def apply_index_configuration(self, config_name: str) -> None: ...

    @abstractmethod
    def gather_statistics(self) -> None: ...

    # ------------------------------------------------------------------ SQL
    @abstractmethod
    def prepare_sql(self, oracle_sql: str) -> str:
        """Translate a workload query (authored in Oracle SQL) to this backend's dialect."""

    @abstractmethod
    def check_executable(self, sql: str) -> Optional[str]:
        """Compile the statement without running it. Returns an error message or None."""

    @abstractmethod
    def execute(self, sql: str, max_rows: int, timeout_s: float) -> ResultSet: ...

    @abstractmethod
    def explain(self, sql: str) -> PlanInfo: ...

    @abstractmethod
    def _run_once(self, sql: str, timeout_s: float) -> RunSample: ...

    def _runtime_statistics(self, sql: str) -> dict[str, Any]:
        """Hook for backends that can report server-side statistics after benchmarking."""
        return {}

    def _tag_sql(self, sql: str, tag: str) -> str:
        return sql

    @abstractmethod
    def schema_metadata(self) -> dict:
        """Catalog information used to build the LLM schema context."""

    # ------------------------------------------------------------------ benchmark
    def benchmark(self, sql: str, warmup_runs: int, measured_runs: int, timeout_s: float,
                  tag: Optional[str] = None) -> PerformanceMetrics:
        try:
            plan = self.explain(sql)
        except Exception as exc:  # plan failure should not abort benchmarking
            plan = PlanInfo(plan_text=f"(plan unavailable: {exc})")

        run_sql = self._tag_sql(sql, tag) if tag else sql
        samples: list[RunSample] = []
        timed_out = False
        error = None
        try:
            for _ in range(max(0, warmup_runs)):
                self._run_once(run_sql, timeout_s)
            for _ in range(max(1, measured_runs)):
                samples.append(self._run_once(run_sql, timeout_s))
        except QueryTimeout as exc:
            timed_out = True
            error = str(exc)
        except Exception as exc:
            error = str(exc)

        timings = [s.elapsed_ms for s in samples]
        runtime = self._runtime_statistics(run_sql) if (samples and tag) else {}
        cpu_samples = [s.cpu_ms for s in samples if s.cpu_ms is not None]
        work_samples = [s.work_units for s in samples if s.work_units is not None]

        if timings:
            median = statistics.median(timings)
            mean = statistics.fmean(timings)
            stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
            mn, mx = min(timings), max(timings)
        else:
            penalty = timeout_s * 1000.0
            median = mean = mn = mx = penalty
            stdev = 0.0

        return PerformanceMetrics(
            median_ms=median,
            mean_ms=mean,
            min_ms=mn,
            max_ms=mx,
            stdev_ms=stdev,
            runs=len(timings),
            rows=samples[-1].rows if samples else 0,
            timings_ms=timings,
            optimizer_cost=plan.cost,
            estimated_cardinality=plan.cardinality,
            cpu_cost=plan.cpu_cost,
            io_cost=plan.io_cost,
            cpu_time_ms=runtime.get("cpu_time_ms", statistics.median(cpu_samples) if cpu_samples else None),
            db_elapsed_ms=runtime.get("db_elapsed_ms"),
            buffer_gets=runtime.get("buffer_gets"),
            disk_reads=runtime.get("disk_reads"),
            work_units=statistics.median(work_samples) if work_samples else runtime.get("work_units"),
            plan_text=plan.plan_text + (("\n\n" + runtime["runtime_plan"]) if runtime.get("runtime_plan") else ""),
            plan_operations=plan.operations,
            plan_signature=plan.signature,
            timed_out=timed_out,
            error=error,
        )


def render_plan_tree(ops: list[PlanOperation]) -> str:
    lines = []
    for op in ops:
        extra = []
        if op.cost is not None:
            extra.append(f"cost={op.cost:g}")
        if op.cardinality is not None:
            extra.append(f"rows={op.cardinality:g}")
        suffix = f"  ({', '.join(extra)})" if extra else ""
        lines.append(f"{'  ' * op.depth}{op.id:>3} {op.label()}{suffix}")
    return "\n".join(lines)
