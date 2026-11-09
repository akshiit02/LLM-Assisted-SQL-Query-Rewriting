"""Experiment runner: the full factorial study (query x scale x index configuration)."""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from . import __version__
from .config import Settings
from .context import build_schema_context
from .db import create_backend
from .llm import RuleBasedRewriter, create_rewriter
from .llm.base import Rewriter
from .models import Candidate, ExperimentCondition, to_dict
from .pipeline import RewritePipeline
from .schema import INDEX_CONFIGURATIONS
from .workload import load_workload

Logger = Callable[[str], None]


def build_rewriters(settings: Settings) -> list[Rewriter]:
    primary = create_rewriter(settings.llm)
    rewriters = [primary]
    if settings.llm.also_run_rule_baseline and not isinstance(primary, RuleBasedRewriter):
        rewriters.append(RuleBasedRewriter())
    return rewriters


class ExperimentRunner:
    def __init__(self, settings: Settings, log: Logger = print, run_dir: Optional[Path] = None):
        self.settings = settings
        self.log = log
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = settings.resolve_path(settings.output_dir)
        self.run_dir = run_dir or out / f"{settings.experiment.name}_{stamp}"

    def run(self) -> Path:
        s = self.settings
        exp_cfg = s.experiment
        for name in exp_cfg.index_configs:
            if name not in INDEX_CONFIGURATIONS:
                raise ValueError(f"Unknown index configuration {name!r}; available: {list(INDEX_CONFIGURATIONS)}")
        queries = load_workload(exp_cfg.workload_file, exp_cfg.queries or None, exp_cfg.complexities or None)
        if not queries:
            raise ValueError("No queries selected – check experiment.queries / experiment.complexities")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.run_dir / "results.jsonl"
        jsonl_path.write_text("", encoding="utf-8")

        rewriters = build_rewriters(s)
        backend = create_backend(s.database)
        pipeline = RewritePipeline(s, backend, rewriters, self.log)
        n_conditions = len(exp_cfg.scales) * len(exp_cfg.index_configs)
        self.log(f"Experiment '{exp_cfg.name}': {len(queries)} queries x {n_conditions} conditions "
                 f"on {backend.display_name}; rewriters: {', '.join(r.name for r in rewriters)}")
        self.log(f"Results directory: {self.run_dir}")

        meta = {
            "name": exp_cfg.name,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tool_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "backend": backend.name,
            "rewriters": [r.name for r in rewriters],
            "llm_model": s.llm.model if s.llm.provider not in ("rule_based", "rules", "rule") else None,
            "settings": _settings_dict(s),
            "queries": [q.id for q in queries],
        }
        pools: dict[str, list[Candidate]] = {}
        results = []
        t_start = time.perf_counter()
        try:
            for scale in exp_cfg.scales:
                loaded = backend.ensure_dataset(scale, s.database.seed, s.database.load_batch_size,
                                                progress=self.log)
                self.log(f"== Scale factor {scale:g} ({'loaded' if loaded else 'reusing existing data'})")
                for idx_name in exp_cfg.index_configs:
                    self.log(f"-- Index configuration '{idx_name}'")
                    backend.apply_index_configuration(idx_name)
                    context = build_schema_context(backend, scale, idx_name, s.llm.include_row_counts)
                    condition = ExperimentCondition(scale=scale, index_config=idx_name, backend=backend.name)
                    for q in queries:
                        self.log(f"[{q.id}] {q.title}  ({q.complexity}, sf={scale:g}, idx={idx_name})")
                        reuse = exp_cfg.generation_mode == "per_query" and q.id in pools
                        res = pipeline.evaluate(q, condition, candidates=pools.get(q.id) if reuse else None,
                                                context=context, allow_repair=not reuse)
                        if exp_cfg.generation_mode == "per_query" and q.id not in pools and not res.error:
                            pools[q.id] = [o.candidate for o in res.outcomes]
                        record = to_dict(res)
                        results.append(record)
                        with open(jsonl_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(record, default=str) + "\n")
        finally:
            backend.close()

        meta["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        meta["duration_s"] = round(time.perf_counter() - t_start, 1)
        meta["llm_usage"] = {r.name: to_dict(r.stats) for r in rewriters}
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        (self.run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
        (self.run_dir / "candidates.json").write_text(
            json.dumps({k: [to_dict(c) for c in v] for k, v in pools.items()}, indent=2), encoding="utf-8")
        self.log(f"Experiment finished in {meta['duration_s']}s - {len(results)} query runs recorded.")
        return self.run_dir


def _settings_dict(s: Settings) -> dict:
    d = to_dict(s)
    d.get("database", {}).get("oracle", {}).pop("password", None)
    return d


def load_results(run_dir: str | Path) -> tuple[list[dict], dict]:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:  # interrupted run – fall back to the incremental log
        results = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return results, meta


def save_settings_snapshot(settings: Settings, path: Path) -> None:
    path.write_text(yaml.safe_dump(_settings_dict(settings), sort_keys=False), encoding="utf-8")
