"""The rewrite-verify-benchmark pipeline for one query under one condition.

Implements stages 3-6 of the proposed workflow and the decision nodes:

    LLM query rewriter  ->  Query validation  --NO-->  reject
                               | YES
                               v
                    Semantic equivalence   --NO-->  reject (optionally: repair round)
                               | YES
                               v
                    Oracle performance evaluation  ->  verdict (improved / neutral / regressed)
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from .complexity import complexity_metrics
from .config import Settings
from .context import SchemaContext, build_schema_context, summarize_plan
from .db.base import DatabaseBackend, QueryError, QueryTimeout, ResultSet
from .equivalence import compare_results
from .llm.base import LLMError, RepairItem, RewriteRequest, Rewriter
from .models import (Candidate, CandidateOutcome, EquivalenceResult, ExperimentCondition, PerformanceMetrics,
                     QueryRunResult, WorkloadQuery)
from .validation import categorize_error, clean_sql, validate_candidate

Logger = Callable[[str], None]

DIALECT_LABELS = {"oracle": "Oracle SQL (Oracle Database 19c+/23ai)", "sqlite": "SQLite 3"}

REJECTED_VERDICTS = ("rejected_unsafe", "rejected_syntax", "rejected_unexecutable",
                     "rejected_nonequivalent", "rejected_timeout")
VALID_VERDICTS = ("improved", "neutral", "regressed")


def _noop(_msg: str) -> None:
    pass


class RewritePipeline:
    def __init__(self, settings: Settings, backend: DatabaseBackend, rewriters: list[Rewriter],
                 log: Logger = _noop):
        self.settings = settings
        self.backend = backend
        self.rewriters = rewriters
        self.log = log

    # ------------------------------------------------------------------ generation
    def _request(self, query: WorkloadQuery, backend_sql: str, context: SchemaContext,
                 original_perf: Optional[PerformanceMetrics]) -> RewriteRequest:
        plan = None
        if self.settings.llm.include_plan_in_prompt and original_perf is not None:
            plan = summarize_plan(original_perf.plan_text)
        return RewriteRequest(query=query, sql=backend_sql, dialect=self.backend.dialect,
                              dialect_label=DIALECT_LABELS.get(self.backend.dialect, self.backend.dialect),
                              context=context, original_plan=plan,
                              num_candidates=self.settings.llm.num_candidates)

    def generate_candidates(self, request: RewriteRequest) -> tuple[list[Candidate], list[str]]:
        candidates: list[Candidate] = []
        errors: list[str] = []
        for rw in self.rewriters:
            try:
                produced = rw.generate(request)
                self.log(f"  {rw.name}: {len(produced)} candidate(s)")
                candidates.extend(produced)
            except LLMError as exc:
                errors.append(f"{rw.name}: {exc}")
                self.log(f"  [warn] {rw.name} failed: {exc}")
        return candidates, errors

    # ------------------------------------------------------------------ evaluation
    def _candidate_timeout(self, original: Optional[PerformanceMetrics]) -> float:
        b = self.settings.benchmark
        if original is None or not original.timings_ms:
            return b.timeout_s
        adaptive = max(b.candidate_timeout_floor_s, b.candidate_timeout_factor * original.median_ms / 1000.0)
        return min(b.timeout_s, adaptive)

    def evaluate_candidate(self, cand: Candidate, backend_sql: str, original_rs: ResultSet,
                           original_perf: PerformanceMetrics) -> CandidateOutcome:
        b, eqs = self.settings.benchmark, self.settings.equivalence
        cand.sql = clean_sql(cand.sql)
        validation = validate_candidate(cand.sql, self.backend)
        outcome = CandidateOutcome(candidate=cand, validation=validation)
        if not validation.passed:
            outcome.verdict = {"safety": "rejected_unsafe", "syntax": "rejected_syntax"}.get(
                validation.stage, "rejected_unexecutable")
            return outcome

        timeout = self._candidate_timeout(original_perf)
        try:
            rs = self.backend.execute(cand.sql, eqs.max_rows, timeout)
        except QueryTimeout as exc:
            outcome.equivalence = EquivalenceResult(False, f"timed out during verification ({exc})", "timeout",
                                                    original_rows=len(original_rs.rows))
            outcome.verdict = "rejected_timeout"
            return outcome
        except QueryError as exc:
            outcome.validation = type(validation)(False, "executable", f"runtime error: {exc}",
                                                  categorize_error(str(exc)))
            outcome.verdict = "rejected_unexecutable"
            return outcome

        outcome.equivalence = compare_results(original_rs, rs, backend_sql, self.backend.dialect, eqs)
        if not outcome.equivalence.equivalent:
            outcome.verdict = "rejected_nonequivalent"
            return outcome

        perf = self.backend.benchmark(cand.sql, b.warmup_runs, b.measured_runs, timeout,
                                      tag=f"{cand.query_id}_{cand.id}_{uuid.uuid4().hex[:8]}")
        outcome.performance = perf
        if perf.error and not perf.timed_out and not perf.timings_ms:
            outcome.verdict = "error"
            return outcome
        outcome.speedup = original_perf.median_ms / perf.median_ms if perf.median_ms > 0 else None
        if perf.optimizer_cost and original_perf.optimizer_cost:
            outcome.cost_ratio = perf.optimizer_cost / original_perf.optimizer_cost
        outcome.plan_changed = (perf.plan_signature != original_perf.plan_signature
                                if perf.plan_signature and original_perf.plan_signature else None)
        thr = b.improvement_threshold
        if perf.timed_out:
            outcome.verdict = "regressed"
        elif outcome.speedup is not None and outcome.speedup >= 1 + thr:
            outcome.verdict = "improved"
        elif outcome.speedup is not None and outcome.speedup <= 1 / (1 + thr):
            outcome.verdict = "regressed"
        else:
            outcome.verdict = "neutral"
        return outcome

    @staticmethod
    def describe_failure(o: CandidateOutcome) -> str:
        if not o.validation.passed:
            return f"{o.validation.stage} check failed: {o.validation.error}"
        eq = o.equivalence
        if eq is None:
            return o.verdict
        text = f"result mismatch – {eq.reason} (original {eq.original_rows} rows, rewrite {eq.rewrite_rows} rows)."
        if eq.only_in_original:
            text += f" Example rows only in the original result: {eq.only_in_original[:3]}."
        if eq.only_in_rewrite:
            text += f" Example rows only in the rewrite result: {eq.only_in_rewrite[:3]}."
        return text

    def evaluate(self, query: WorkloadQuery, condition: ExperimentCondition,
                 candidates: Optional[list[Candidate]] = None,
                 context: Optional[SchemaContext] = None,
                 allow_repair: bool = True) -> QueryRunResult:
        b = self.settings.benchmark
        backend_sql = self.backend.prepare_sql(query.sql)
        try:
            metrics = complexity_metrics(backend_sql, self.backend.dialect)
        except Exception:
            metrics = complexity_metrics(query.sql, "oracle")
        result = QueryRunResult(query=query, condition=condition, backend_sql=backend_sql,
                                complexity_metrics=metrics, original_performance=None)

        # ---- reference result and baseline performance of the original query
        try:
            original_rs = self.backend.execute(backend_sql, self.settings.equivalence.max_rows, b.timeout_s)
        except (QueryError, QueryTimeout) as exc:
            result.error = f"original query failed: {exc}"
            self.log(f"  [error] {result.error}")
            return result
        original_perf = self.backend.benchmark(backend_sql, b.warmup_runs, b.measured_runs, b.timeout_s,
                                               tag=f"{query.id}_orig_{uuid.uuid4().hex[:8]}")
        result.original_performance = original_perf
        self.log(f"  original: {original_perf.median_ms:.2f} ms median, {len(original_rs.rows)} rows"
                 + (f", cost {original_perf.optimizer_cost:g}" if original_perf.optimizer_cost else ""))

        if context is None:
            context = build_schema_context(self.backend, condition.scale, condition.index_config,
                                           self.settings.llm.include_row_counts)
        request = self._request(query, backend_sql, context, original_perf)

        # ---- stage 3: candidate generation
        stats_before = [(rw.stats.calls, rw.stats.seconds) for rw in self.rewriters]
        if candidates is None:
            candidates, errors = self.generate_candidates(request)
            if errors and not candidates:
                result.error = "; ".join(errors)
        else:
            candidates = [Candidate(**{**c.__dict__}) for c in candidates]  # fresh copies

        # ---- stages 4-6 for each candidate
        for cand in candidates:
            o = self.evaluate_candidate(cand, backend_sql, original_rs, original_perf)
            result.outcomes.append(o)
            self._log_outcome(o)

        # ---- optional repair rounds (feedback loop to the LLM)
        if allow_repair:
            repaired_parents: set[str] = set()
            for round_no in range(1, self.settings.llm.max_repair_rounds + 1):
                for rw in self.rewriters:
                    if not rw.supports_repair:
                        continue
                    failed = [o for o in result.outcomes
                              if o.verdict in ("rejected_syntax", "rejected_unexecutable", "rejected_nonequivalent")
                              and o.candidate.source in ("llm", "repair")
                              and o.candidate.generator.startswith(rw.name)
                              and o.candidate.id not in repaired_parents]
                    if not failed:
                        continue
                    self.log(f"  repair round {round_no}: {len(failed)} rejected candidate(s) sent back to {rw.name}")
                    items = [RepairItem(o.candidate, self.describe_failure(o)) for o in failed]
                    repaired_parents.update(o.candidate.id for o in failed)
                    try:
                        fixed = rw.repair(request, items, round_no)
                    except LLMError as exc:
                        self.log(f"  [warn] repair failed: {exc}")
                        continue
                    for cand in fixed:
                        o = self.evaluate_candidate(cand, backend_sql, original_rs, original_perf)
                        result.outcomes.append(o)
                        self._log_outcome(o)

        for rw, (calls, secs) in zip(self.rewriters, stats_before):
            result.llm_calls += rw.stats.calls - calls
            result.llm_seconds += rw.stats.seconds - secs

        improved = [o for o in result.outcomes if o.verdict == "improved" and o.speedup]
        if improved:
            result.best_candidate_id = max(improved, key=lambda o: o.speedup).candidate.id
        return result

    def _log_outcome(self, o: CandidateOutcome) -> None:
        c = o.candidate
        if o.verdict.startswith("rejected"):
            why = o.validation.error if not o.validation.passed else (o.equivalence.reason if o.equivalence else "")
            self.log(f"    {c.id:<6} {c.strategy[:34]:<34} {o.verdict:<24} {(why or '')[:90]}")
        else:
            sp = f"{o.speedup:.2f}x" if o.speedup else "n/a"
            self.log(f"    {c.id:<6} {c.strategy[:34]:<34} {o.verdict:<24} speedup {sp}")


def run_single(settings: Settings, backend: DatabaseBackend, rewriters: list[Rewriter], query: WorkloadQuery,
               scale: float, index_config: str, log: Logger = _noop) -> QueryRunResult:
    """Convenience wrapper used by the CLI ``rewrite`` command and the web UI."""
    condition = ExperimentCondition(scale=scale, index_config=index_config, backend=backend.name)
    pipeline = RewritePipeline(settings, backend, rewriters, log)
    t0 = time.perf_counter()
    result = pipeline.evaluate(query, condition)
    log(f"  done in {time.perf_counter() - t0:.1f}s")
    return result
