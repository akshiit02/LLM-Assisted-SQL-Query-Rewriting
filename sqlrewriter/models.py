"""Core data structures shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


COMPLEXITY_LEVELS = ("basic", "intermediate", "advanced", "complex")


@dataclass
class WorkloadQuery:
    """Stage 1 – one original query of the SQL workload."""

    id: str
    complexity: str
    title: str
    sql: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """Stage 3 – one candidate rewrite of an original query."""

    id: str
    query_id: str
    sql: str
    source: str  # "llm" | "rule" | "repair"
    strategy: str = ""
    rationale: str = ""
    parent_id: Optional[str] = None  # set for repaired candidates
    generator: str = ""  # provider/model name


@dataclass
class ValidationResult:
    """Stage 4 – safety, syntax and executability checks."""

    passed: bool
    stage: str  # "safety" | "syntax" | "executable" | "ok"
    error: Optional[str] = None
    error_category: Optional[str] = None


@dataclass
class EquivalenceResult:
    """Stage 5 – result-set comparison between original and rewrite."""

    equivalent: bool
    reason: str
    category: str  # "equivalent" | "row_count" | "values" | "columns" | "types" | "order" | "error" | "timeout"
    original_rows: int = 0
    rewrite_rows: int = 0
    ordered: bool = False
    vacuous: bool = False  # both results empty -> weak evidence
    column_warnings: list[str] = field(default_factory=list)
    only_in_original: list[list[Any]] = field(default_factory=list)
    only_in_rewrite: list[list[Any]] = field(default_factory=list)


@dataclass
class PlanOperation:
    id: int
    parent_id: Optional[int]
    operation: str
    options: str = ""
    object_name: str = ""
    cost: Optional[float] = None
    cardinality: Optional[float] = None
    depth: int = 0

    def label(self) -> str:
        parts = [self.operation]
        if self.options:
            parts.append(self.options)
        if self.object_name:
            parts.append(self.object_name)
        return " ".join(parts)


@dataclass
class PerformanceMetrics:
    """Stage 6 – execution plan and runtime metrics for one SQL statement."""

    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    runs: int
    rows: int
    timings_ms: list[float] = field(default_factory=list)
    optimizer_cost: Optional[float] = None
    estimated_cardinality: Optional[float] = None
    cpu_cost: Optional[float] = None
    io_cost: Optional[float] = None
    cpu_time_ms: Optional[float] = None
    db_elapsed_ms: Optional[float] = None
    buffer_gets: Optional[float] = None
    disk_reads: Optional[float] = None
    work_units: Optional[float] = None  # backend-specific work counter (SQLite VM steps)
    plan_text: str = ""
    plan_operations: list[PlanOperation] = field(default_factory=list)
    plan_signature: str = ""
    timed_out: bool = False
    error: Optional[str] = None


@dataclass
class CandidateOutcome:
    candidate: Candidate
    validation: ValidationResult
    equivalence: Optional[EquivalenceResult] = None
    performance: Optional[PerformanceMetrics] = None
    speedup: Optional[float] = None  # original_median / rewrite_median (>1 = faster)
    cost_ratio: Optional[float] = None  # rewrite_cost / original_cost (<1 = cheaper)
    plan_changed: Optional[bool] = None
    verdict: str = "pending"
    # verdicts: rejected_unsafe | rejected_syntax | rejected_unexecutable |
    #           rejected_nonequivalent | improved | neutral | regressed | error


@dataclass
class ExperimentCondition:
    scale: float
    index_config: str
    backend: str

    @property
    def key(self) -> str:
        return f"{self.backend}|sf={self.scale:g}|idx={self.index_config}"


@dataclass
class QueryRunResult:
    query: WorkloadQuery
    condition: ExperimentCondition
    backend_sql: str
    complexity_metrics: dict[str, Any]
    original_performance: Optional[PerformanceMetrics]
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    error: Optional[str] = None
    llm_calls: int = 0
    llm_seconds: float = 0.0

    def best(self) -> Optional[CandidateOutcome]:
        for o in self.outcomes:
            if o.candidate.id == self.best_candidate_id:
                return o
        return None


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to JSON-friendly dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
