import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlrewriter.config import load_settings  # noqa: E402
from sqlrewriter.db import create_backend  # noqa: E402

SCALE = 0.1


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("db")
    out_dir = tmp_path_factory.mktemp("results")
    cache_dir = tmp_path_factory.mktemp("cache")
    return load_settings(None, {
        "database": {"backend": "sqlite", "sqlite": {"path_template": str(db_dir / "bench_sf{scale}.db")}},
        "llm": {"provider": "rule_based", "cache_dir": str(cache_dir)},
        "benchmark": {"warmup_runs": 0, "measured_runs": 2, "timeout_s": 30},
        "experiment": {"scales": [SCALE], "index_configs": ["fk_indexes"]},
        "output_dir": str(out_dir),
    })


@pytest.fixture(scope="session")
def backend(settings):
    b = create_backend(settings.database)
    b.ensure_dataset(SCALE, settings.database.seed)
    b.apply_index_configuration("fk_indexes")
    yield b
    b.close()
