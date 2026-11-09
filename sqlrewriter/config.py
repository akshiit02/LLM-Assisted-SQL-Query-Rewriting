"""Configuration loading (YAML + environment variable interpolation)."""

from __future__ import annotations

import os
import re
from dataclasses import MISSING, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


@dataclass
class SQLiteSettings:
    path_template: str = "data/benchmark_sf{scale}.db"


@dataclass
class OracleSettings:
    user: str = ""
    password: str = ""
    dsn: str = "localhost:1521/FREEPDB1"
    # Optional: flush buffer cache / shared pool between runs (needs ALTER SYSTEM)
    flush_buffer_cache: bool = False
    collect_runtime_stats: bool = True  # V$SQL metrics (needs SELECT_CATALOG_ROLE)
    thick_mode: bool = False
    lib_dir: str = ""


@dataclass
class DatabaseSettings:
    backend: str = "sqlite"
    sqlite: SQLiteSettings = field(default_factory=SQLiteSettings)
    oracle: OracleSettings = field(default_factory=OracleSettings)
    seed: int = 42
    load_batch_size: int = 5000


@dataclass
class LLMSettings:
    provider: str = "rule_based"  # rule_based | anthropic | groq | ollama
    model: str = "claude-opus-5-5"
    num_candidates: int = 3
    effort: str = "high"
    max_tokens: int = 32000
    include_plan_in_prompt: bool = True
    include_row_counts: bool = True
    max_repair_rounds: int = 1
    also_run_rule_baseline: bool = True
    cache_dir: str = ".cache/llm"
    use_cache: bool = True
    ollama_url: str = "http://localhost:11434"
    request_timeout_s: float = 600.0
    temperature: Optional[float] = None  # groq / ollama sampling temperature (None = provider default)
    # Groq: on HTTP 429 wait for retry-after and retry; give up if the wait exceeds the cap (daily limits)
    rate_limit_retries: int = 8
    rate_limit_max_wait_s: float = 120.0


@dataclass
class ExperimentSettings:
    name: str = "experiment"
    scales: list[float] = field(default_factory=lambda: [0.1, 1.0])
    index_configs: list[str] = field(default_factory=lambda: ["baseline", "fk_indexes", "tuned"])
    queries: list[str] = field(default_factory=list)  # empty = all
    complexities: list[str] = field(default_factory=list)  # empty = all
    workload_file: str = "workload/queries.yaml"
    # per_query: candidates generated once per query (controlled comparison)
    # per_condition: regenerated for every (scale, index) with that context
    generation_mode: str = "per_query"


@dataclass
class BenchmarkSettings:
    warmup_runs: int = 1
    measured_runs: int = 5
    timeout_s: float = 60.0
    improvement_threshold: float = 0.10
    # Regression guard: a candidate may run at most max(floor, factor x original median),
    # capped by timeout_s, so a catastrophically slow rewrite cannot stall the experiment.
    candidate_timeout_factor: float = 50.0
    candidate_timeout_floor_s: float = 5.0


@dataclass
class EquivalenceSettings:
    float_rel_tolerance: float = 1e-6
    float_abs_tolerance: float = 1e-6
    strict_column_names: bool = True
    max_rows: int = 2_000_000
    diff_sample_size: int = 5


@dataclass
class Settings:
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    equivalence: EquivalenceSettings = field(default_factory=EquivalenceSettings)
    output_dir: str = "results"
    source_path: Optional[str] = None

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _build(cls, data: Optional[dict]):
    data = data or {}
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in cls.__dataclass_fields__.items():
        if name not in data:
            continue
        value = data[name]
        if f.default_factory is not MISSING:
            default_obj = f.default_factory()
            if is_dataclass(default_obj) and isinstance(value, dict):
                value = _build(type(default_obj), value)
        kwargs[name] = value
    return cls(**kwargs)


def load_settings(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> Settings:
    load_dotenv()
    data: dict = {}
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    data = _interpolate(data)
    if overrides:
        data = _deep_merge(data, overrides)
    settings = _build(Settings, data)
    settings.source_path = str(path) if path else None
    settings.experiment.scales = [float(s) for s in settings.experiment.scales]
    return settings


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
