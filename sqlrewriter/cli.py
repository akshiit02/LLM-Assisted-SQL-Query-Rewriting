"""Command-line interface:  python -m sqlrewriter <command> ...

Commands
  setup      create the benchmark schema and load synthetic data
  workload   list the workload queries with their complexity metrics
  explain    show the execution plan of a workload query or ad-hoc SQL
  rewrite    run the full pipeline for one query and print the verdicts
  run        run the full experiment (queries x scales x index configurations) and build the report
  report     (re)build the HTML/Markdown report for a finished run
  check      verify database connectivity and LLM credentials
  oracle-scripts   write standalone Oracle DDL scripts to sql/oracle/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .config import PROJECT_ROOT, Settings, load_settings

console = Console(highlight=False)

DEFAULT_CONFIG = "config/experiment.yaml"


def _log(msg: str) -> None:
    console.print(msg, markup=False)


def _settings(args) -> Settings:
    overrides: dict = {}
    if getattr(args, "backend", None):
        overrides.setdefault("database", {})["backend"] = args.backend
    if getattr(args, "provider", None):
        overrides.setdefault("llm", {})["provider"] = args.provider
    if getattr(args, "model", None):
        overrides.setdefault("llm", {})["model"] = args.model
    if getattr(args, "candidates", None):
        overrides.setdefault("llm", {})["num_candidates"] = args.candidates
    if getattr(args, "no_rule_baseline", False):
        overrides.setdefault("llm", {})["also_run_rule_baseline"] = False
    if getattr(args, "no_cache", False):
        overrides.setdefault("llm", {})["use_cache"] = False
    if getattr(args, "runs", None):
        overrides.setdefault("benchmark", {})["measured_runs"] = args.runs
    if getattr(args, "scales", None):
        overrides.setdefault("experiment", {})["scales"] = [float(x) for x in args.scales.split(",")]
    if getattr(args, "index_configs", None):
        overrides.setdefault("experiment", {})["index_configs"] = args.index_configs.split(",")
    if getattr(args, "queries", None):
        overrides.setdefault("experiment", {})["queries"] = args.queries.split(",")
    if getattr(args, "complexities", None):
        overrides.setdefault("experiment", {})["complexities"] = args.complexities.split(",")
    if getattr(args, "name", None):
        overrides.setdefault("experiment", {})["name"] = args.name
    return load_settings(args.config, overrides)


# ---------------------------------------------------------------------------- commands

def cmd_setup(args) -> int:
    from .db import create_backend

    s = _settings(args)
    scales = [float(x) for x in args.scale.split(",")] if args.scale else s.experiment.scales
    backend = create_backend(s.database)
    try:
        for sc in scales:
            loaded = backend.ensure_dataset(sc, s.database.seed, s.database.load_batch_size, force=args.force,
                                            progress=_log)
            backend.apply_index_configuration(args.index_config)
            meta = backend.schema_metadata()
            total = sum(int(t.get("row_count") or 0) for t in meta["tables"])
            _log(f"scale {sc:g}: {'loaded' if loaded else 'already present'} - {len(meta['tables'])} tables, "
                 f"{total:,} rows, indexes: {args.index_config}")
            if backend.name == "oracle":
                _log("Note: Oracle holds one scale at a time; `run` reloads data for each scale factor.")
    finally:
        backend.close()
    return 0


def cmd_workload(args) -> int:
    from .complexity import complexity_metrics
    from .workload import load_workload

    s = _settings(args)
    t = Table(title="SQL workload")
    for col in ("id", "complexity", "score", "tables", "joins", "subq", "corr", "CTEs", "windows"):
        t.add_column(col, no_wrap=True)
    t.add_column("title", overflow="ellipsis", no_wrap=True)
    for q in load_workload(s.experiment.workload_file):
        m = complexity_metrics(q.sql)
        t.add_row(q.id, q.complexity, f"{m['score']:g}", str(m["tables"]), str(m["joins"]), str(m["subqueries"]),
                  str(m["correlated_subqueries"]), str(m["ctes"]), str(m["window_functions"]), q.title)
    console.print(t)
    return 0


def _resolve_query(args, s: Settings):
    from .workload import adhoc_query, load_workload

    if getattr(args, "query", None):
        qs = load_workload(s.experiment.workload_file, [args.query])
        if not qs:
            raise SystemExit(f"Unknown query id {args.query}")
        return qs[0]
    sql = args.sql
    if getattr(args, "file", None):
        sql = Path(args.file).read_text(encoding="utf-8")
    if not sql:
        raise SystemExit("Provide --query ID, --sql 'SELECT ...' or --file path.sql")
    return adhoc_query(sql)


def cmd_explain(args) -> int:
    from .db import create_backend

    s = _settings(args)
    q = _resolve_query(args, s)
    backend = create_backend(s.database)
    try:
        backend.ensure_dataset(args.scale, s.database.seed, s.database.load_batch_size, progress=_log)
        backend.apply_index_configuration(args.index_config)
        sql = backend.prepare_sql(q.sql)
        console.print(Syntax(sql, "sql", theme="ansi_dark", word_wrap=True))
        plan = backend.explain(sql)
        console.print(plan.plan_text, markup=False)
        if plan.cost is not None:
            _log(f"optimizer cost={plan.cost:g} cardinality={plan.cardinality} cpu_cost={plan.cpu_cost} io_cost={plan.io_cost}")
    finally:
        backend.close()
    return 0


def print_run_result(res, verbose: bool = False) -> None:
    op = res.original_performance
    console.print(Panel.fit(f"[b]{res.query.id}[/b] {res.query.title}  |  complexity {res.query.complexity} "
                            f"(score {res.complexity_metrics.get('score')})  |  {res.condition.key}"))
    if res.error:
        console.print(f"[red]error:[/red] {res.error}")
    if op:
        extra = f", cost {op.optimizer_cost:g}" if op.optimizer_cost else ""
        console.print(f"Original: median {op.median_ms:.2f} ms over {op.runs} runs, {op.rows} rows{extra}")
    t = Table(show_lines=False)
    for col in ("id", "source", "strategy", "verdict", "speedup", "median ms", "cost ratio", "note"):
        t.add_column(col, overflow="fold")
    colors = {"improved": "green", "regressed": "red", "neutral": "white"}
    for o in res.outcomes:
        note = ""
        if not o.validation.passed:
            note = (o.validation.error or "")[:120]
        elif o.equivalence and not o.equivalence.equivalent:
            note = o.equivalence.reason
        elif o.equivalence and o.equivalence.vacuous:
            note = "vacuous (empty results)"
        color = colors.get(o.verdict, "yellow")
        t.add_row(o.candidate.id, o.candidate.source, o.candidate.strategy[:40], f"[{color}]{o.verdict}[/{color}]",
                  f"{o.speedup:.2f}x" if o.speedup else "–",
                  f"{o.performance.median_ms:.2f}" if o.performance else "–",
                  f"{o.cost_ratio:.2f}" if o.cost_ratio else "–", note)
    console.print(t)
    best = res.best()
    if best:
        console.print(f"[green]Best verified rewrite: {best.candidate.id} ({best.candidate.strategy}) - "
                      f"{best.speedup:.2f}x faster[/green]")
        console.print(Syntax(best.candidate.sql, "sql", theme="ansi_dark", word_wrap=True))
    else:
        console.print("[yellow]No verified rewrite beat the original - keep the original query.[/yellow]")
    if verbose:
        for o in res.outcomes:
            console.print(f"\n[b]{o.candidate.id}[/b] {o.candidate.strategy}: {o.candidate.rationale}")
            console.print(Syntax(o.candidate.sql, "sql", theme="ansi_dark", word_wrap=True))


def cmd_rewrite(args) -> int:
    from .db import create_backend
    from .experiment import build_rewriters
    from .models import to_dict
    from .pipeline import run_single

    s = _settings(args)
    q = _resolve_query(args, s)
    backend = create_backend(s.database)
    try:
        backend.ensure_dataset(args.scale, s.database.seed, s.database.load_batch_size, progress=_log)
        backend.apply_index_configuration(args.index_config)
        rewriters = build_rewriters(s)
        res = run_single(s, backend, rewriters, q, args.scale, args.index_config, log=_log)
    finally:
        backend.close()
    print_run_result(res, args.verbose)
    if args.json:
        Path(args.json).write_text(json.dumps(to_dict(res), indent=2, default=str), encoding="utf-8")
        _log(f"JSON written to {args.json}")
    return 0


def cmd_run(args) -> int:
    from .experiment import ExperimentRunner, save_settings_snapshot
    from .report import build_report

    s = _settings(args)
    runner = ExperimentRunner(s, log=_log)
    run_dir = runner.run()
    save_settings_snapshot(s, run_dir / "config_snapshot.yaml")
    report = build_report(run_dir)
    _print_summary(run_dir)
    console.print(f"\n[b green]Report:[/b green] {report}")
    return 0


def _print_summary(run_dir: Path) -> None:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for k, v in summary.get("rq_answers", {}).items():
        console.print(f"[b]{k}[/b] {v}")


def cmd_report(args) -> int:
    from .report import build_report, latest_run_dir

    s = _settings(args)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(s.resolve_path(s.output_dir))
    if run_dir is None:
        raise SystemExit("No experiment runs found - run `python -m sqlrewriter run` first")
    report = build_report(run_dir)
    _print_summary(run_dir)
    console.print(f"\n[b green]Report:[/b green] {report}")
    return 0


def cmd_check(args) -> int:
    from .db import create_backend
    from .llm import create_rewriter

    s = _settings(args)
    ok = True
    try:
        backend = create_backend(s.database)
        backend.activate(s.experiment.scales[0])
        info = backend.dataset_info()
        _log(f"[ok] database backend '{backend.name}' reachable; loaded dataset: {info or 'none (run setup)'}")
        backend.close()
    except Exception as exc:
        ok = False
        _log(f"[fail] database: {exc}")
    try:
        rw = create_rewriter(s.llm)
        _log(f"[ok] rewriter '{rw.name}' initialised"
             + (f" (model {getattr(rw, 'model', s.llm.model)})" if rw.name != "rule_based" else ""))
        if rw.name == "anthropic" and args.ping:
            msg = rw.client.messages.create(model=s.llm.model, max_tokens=64,
                                            messages=[{"role": "user", "content": "Reply with OK."}])
            _log(f"[ok] Anthropic API responded (stop_reason={msg.stop_reason})")
        if rw.name == "groq" and args.ping:
            reply = rw.client.chat.completions.create(model=rw.model, max_completion_tokens=256,
                                                      messages=[{"role": "user", "content": "Reply with OK."}])
            _log(f"[ok] Groq API responded (model {reply.model}, finish_reason={reply.choices[0].finish_reason})")
    except Exception as exc:
        ok = False
        _log(f"[fail] rewriter: {exc}")
    return 0 if ok else 1


def cmd_oracle_scripts(args) -> int:
    from .schema import write_oracle_scripts

    for p in write_oracle_scripts(str(PROJECT_ROOT / "sql" / "oracle")):
        _log(f"wrote {p}")
    return 0


# ---------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sqlrewriter",
                                description="LLM-assisted semantics-preserving SQL query rewriting study")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, db=True):
        sp.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="YAML config file (default: %(default)s)")
        if db:
            sp.add_argument("--backend", choices=["sqlite", "oracle"], help="override database.backend")

    def llm_opts(sp):
        sp.add_argument("--provider", choices=["rule_based", "anthropic", "groq", "ollama"], help="override llm.provider")
        sp.add_argument("--model", help="override llm.model")
        sp.add_argument("--candidates", type=int, help="LLM candidates per query")
        sp.add_argument("--no-rule-baseline", action="store_true", help="do not also run the rule-based baseline")
        sp.add_argument("--no-cache", action="store_true", help="ignore the LLM response cache")
        sp.add_argument("--runs", type=int, help="measured runs per benchmark")

    sp = sub.add_parser("setup", help="create schema and load synthetic data")
    common(sp)
    sp.add_argument("--scale", help="comma-separated scale factors (default: experiment.scales)")
    sp.add_argument("--index-config", default="fk_indexes")
    sp.add_argument("--force", action="store_true", help="reload even if data is present")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("workload", help="list workload queries")
    common(sp, db=False)
    sp.set_defaults(func=cmd_workload)

    for name, func, helptext in (("explain", cmd_explain, "show an execution plan"),
                                 ("rewrite", cmd_rewrite, "rewrite + verify + benchmark one query")):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        g = sp.add_mutually_exclusive_group()
        g.add_argument("-q", "--query", help="workload query id, e.g. Q11")
        g.add_argument("--sql", help="ad-hoc SQL text (Oracle dialect)")
        g.add_argument("--file", help="file containing one SQL query")
        sp.add_argument("--scale", type=float, default=1.0)
        sp.add_argument("--index-config", default="fk_indexes", choices=["baseline", "fk_indexes", "tuned"])
        if name == "rewrite":
            llm_opts(sp)
            sp.add_argument("-v", "--verbose", action="store_true", help="print every candidate's SQL")
            sp.add_argument("--json", help="write the full result as JSON")
        sp.set_defaults(func=func)

    sp = sub.add_parser("run", help="run the full experiment and build the report")
    common(sp)
    llm_opts(sp)
    sp.add_argument("--name", help="experiment name (results sub-directory prefix)")
    sp.add_argument("--scales", help="comma-separated scale factors")
    sp.add_argument("--index-configs", help="comma-separated index configurations")
    sp.add_argument("--queries", help="comma-separated query ids")
    sp.add_argument("--complexities", help="comma-separated complexity levels")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("report", help="rebuild the report of a finished run")
    common(sp, db=False)
    sp.add_argument("run_dir", nargs="?", help="results/<run> directory (default: latest)")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("check", help="check database and LLM configuration")
    common(sp)
    llm_opts(sp)
    sp.add_argument("--ping", action="store_true", help="send a tiny test request to the LLM API")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("oracle-scripts", help="write standalone Oracle DDL scripts")
    sp.set_defaults(func=cmd_oracle_scripts, config=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows legacy code pages
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
