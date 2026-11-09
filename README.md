# LLM-Assisted Semantics-Preserving SQL Query Rewriting for Query Performance Optimization

**Advanced SQL and Modern Databases**, School of Computing, SRM Institute of Science and Technology
Akshit Sharma (RA2311003011096) · Dhavval (RA2311003011151) · Akshaya KV (RA2311003011155) · Janav Gandhi (RA2311003011156)
Supervisor: Dr. R. Sathya, Department of CTECH

---

This repository implements the complete framework from the project proposal. An LLM generates alternative
formulations of an SQL query. Every candidate is **validated** (safe, syntactically valid, executable) and
**verified** for semantic equivalence by strict result-set comparison. Only then is it **benchmarked** with
Oracle's native tooling (EXPLAIN PLAN, DBMS_XPLAN, V$SQL). A factorial experiment over *query complexity ×
dataset scale × index configuration* answers the five research questions, and the pipeline produces a report
with the best verified rewrite per query, side-by-side execution plans, and an analysis of failed and harmful
rewrites.

It runs **out of the box with no database server and no API key**: a SQLite backend and a rule-based
rewriter are included. Switching to an LLM (Groq, Claude or a local Ollama model) or to Oracle Database is a
config change.

```
 1 SQL workload ─► 2 Schema context ─► 3 LLM rewriter ─► 4 Validation ──NO──► reject
   (20 queries,      (live catalog:      (N candidates,     (safety, syntax,
   4 complexity      columns, keys,      plan-aware          executability)
   levels)           indexes, stats)     prompt)                 │YES
                                                                  ▼
 8 Final output ◄─ 7 Comparative ◄─ 6 Oracle performance ◄─ 5 Semantic equivalence ──NO──► reject
   (best rewrite,     analysis          evaluation (EXPLAIN       (execute both, compare      (+ optional
   plan diff,         (RQ1–RQ5,         PLAN, DBMS_XPLAN, V$SQL,  columns, types, multiset,    repair round
   failure report)    charts)           timings, CPU/IO)          tie-aware order)             with feedback)
```

## Contents

- [Quick start](#quick-start)
- [Using an LLM (Groq)](#using-an-llm-groq)
- [Using Claude and Oracle](#using-claude-and-oracle)
- [How each workflow stage is implemented](#how-each-workflow-stage-is-implemented)
- [Research questions → metrics](#research-questions--metrics)
- [Benchmark schema, data and workload](#benchmark-schema-data-and-workload)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Limitations and threats to validity](#limitations-and-threats-to-validity)
- [References](#references)

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# full demo experiment: 20 queries × 2 scales × 3 index configurations (SQLite + rule-based rewriter)
python -m sqlrewriter run
# -> results/demo_sqlite_rules_<timestamp>/report.html

# one query, printed in the terminal
python -m sqlrewriter rewrite -q Q14 --scale 0.5 --index-config baseline -v

# interactive dashboard
streamlit run app/streamlit_app.py
```

The first run generates the synthetic database files in `data/`, which takes a few seconds.

## Using an LLM (Groq)

[Groq](https://console.groq.com) serves open-weight models with a free tier. This is the quickest way to get
LLM-generated rewrites.

1. Create a free key at <https://console.groq.com/keys>.
2. Put it in `.env` (copy `.env.example`): `GROQ_API_KEY=gsk_...`. Alternatively, set it in the environment
   (PowerShell: `$env:GROQ_API_KEY="gsk_..."`) or paste it into the dashboard sidebar.
3. Run:

```bash
python -m sqlrewriter check   -c config/experiment_groq.yaml --ping       # key + model reachable?
python -m sqlrewriter rewrite -c config/experiment_groq.yaml -q Q14 --scale 0.5 -v
python -m sqlrewriter run     -c config/experiment_groq.yaml               # full study with the LLM
streamlit run app/streamlit_app.py                                        # choose "groq" as the rewriter
```

The Groq provider (`sqlrewriter/llm/groq_provider.py`) uses `openai/gpt-oss-120b` by default, with:
- **strict JSON-schema structured outputs** for models that support them (`openai/gpt-oss-120b`,
  `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`), and JSON-object mode for any other model, e.g.
  `--model llama-3.3-70b-versatile`;
- `llm.effort` mapped to gpt-oss `reasoning_effort` (`low`/`medium`/`high`);
- **free-tier rate-limit handling**: on a 429 it waits for the server's `retry-after` and retries. If the wait
  is longer than `rate_limit_max_wait_s` (the daily limit), it stops calling the API for the rest of the run
  and the remaining queries fall back to the rule-based baseline;
- the same on-disk **response cache** and **repair loop** as the Claude provider.

On the free tier (8K tokens/minute, 200K tokens/day for gpt-oss-120b), one call is about 2.5K prompt tokens
plus the answer, so expect roughly one LLM call per minute. A full 20-query run (about 20–40 calls) fits in
roughly one day's budget. If a run hits the daily limit, re-run it the next day: cached answers cost nothing,
so it continues where it stopped. Use `--queries Q01,Q06,Q11,Q16` for a quick, cheap run.

## Using Claude and Oracle

### Claude as the rewriter

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python -m sqlrewriter check -c config/experiment_claude.yaml --ping
python -m sqlrewriter rewrite -c config/experiment_claude.yaml -q Q17 --scale 0.5
python -m sqlrewriter run -c config/experiment_claude.yaml
```

The Claude provider (`sqlrewriter/llm/anthropic_provider.py`) uses `claude-opus-5-5` by default with:
- **structured outputs**: a JSON schema, so every response parses as `{"candidates": [...]}`;
- **adaptive thinking** with a configurable `effort` (`low` … `max`);
- **server-side refusal fallback** (`fallbacks: "default"`);
- streaming, typed error handling, and an on-disk **response cache** (`.cache/llm`), so a re-run with the same
  prompts is free and reproducible.

The rule-based baseline runs next to the LLM (`also_run_rule_baseline: true`), which gives the "LLM vs
traditional rule-based rewriting" comparison. A local open-weight model can be used instead with
`--provider ollama --model qwen2.5-coder`, or Groq with `--provider groq` (see above).

### Oracle as the database

```bash
docker compose up -d                        # Oracle Database 23ai Free, creates user sqlrw/sqlrw
cp .env.example .env                        # adjust ORACLE_DSN / credentials if needed
python -m sqlrewriter check -c config/experiment_oracle.yaml
python -m sqlrewriter run   -c config/experiment_oracle_rules.yaml   # validate the setup without LLM costs
python -m sqlrewriter run   -c config/experiment_oracle.yaml         # the full study
```

Any existing Oracle 19c+ instance also works. Create the user with `sql/oracle/00_create_user.sql`. The
backend uses python-oracledb in thin mode, so no Oracle Client install is needed. On Oracle it records:

| Metric | Source |
|---|---|
| optimizer cost, cardinality, CPU cost, I/O cost | `EXPLAIN PLAN` → `PLAN_TABLE` |
| formatted estimated plan | `DBMS_XPLAN.DISPLAY` |
| runtime plan actually used | `DBMS_XPLAN.DISPLAY_CURSOR` |
| elapsed time, CPU time, buffer gets, disk reads per execution | `V$SQL` (tagged cursors) |
| wall-clock time (warm-up + N measured runs, median) | client |
| fresh statistics after each load and index change | `DBMS_STATS.GATHER_SCHEMA_STATS` |

`V$` access needs `SELECT_CATALOG_ROLE`; without it the backend falls back to client-side timings.
`flush_buffer_cache: true` enables cold-cache runs (needs `ALTER SYSTEM`). Standalone DDL scripts for DBAs are
in `sql/oracle/`; regenerate them with `python -m sqlrewriter oracle-scripts`.

## How each workflow stage is implemented

| # | Stage (from the proposal) | Module | What it does |
|---|---|---|---|
| 1 | SQL workload | `workload/queries.yaml`, `workload.py`, `complexity.py` | 20 Oracle-SQL queries, 5 per level (basic, intermediate, advanced, complex), each with a known rewrite opportunity. Structural metrics (joins, subqueries, correlation, CTEs, windows, nesting) and a complexity score. |
| 2 | Database & schema context | `context.py`, `db/*.schema_metadata()` | Reads the **live catalog** (columns, types, nullability, PK/FK, indexes, row counts) and renders it for the prompt, with the index configuration and dataset scale. |
| 3 | LLM query rewriter | `llm/groq_provider.py`, `llm/anthropic_provider.py`, `llm/ollama_provider.py`, `llm/rule_based.py`, `llm/prompts.py` | A structured prompt (schema + original SQL + current plan) produces N diverse candidates. A **repair loop** sends rejected candidates back with the verifier's feedback. |
| 4 | Query validation | `validation.py` | **Safety**: exactly one read-only query; DML/DDL/PL-SQL/`FOR UPDATE` blocked, via AST plus a lexical check. **Syntax**: sqlglot parse, with the database parser as final judge. **Executability**: compiled without running (Oracle `cursor.parse`, SQLite `EXPLAIN`). Errors are categorised (invalid identifier, GROUP BY error, …). |
| 5 | Semantic equivalence verification | `equivalence.py` | Executes both queries in a read-only transaction and compares strictly: column count and names, type categories, **multiset** of rows (duplicates matter), a small float tolerance, and **tie-aware order** when the original has `ORDER BY`. Empty-vs-empty results are flagged as vacuous. |
| 6 | Oracle performance evaluation | `db/oracle_backend.py`, `db/base.py::benchmark` | Warm-up + measured runs, median, stdev, plan, cost, CPU/IO metrics, plan-shape signature. A regression guard caps a slow candidate at `max(floor, factor × original)`. |
| 7 | Comparative analysis | `analysis.py` | Funnel rates, speedups (median and geometric mean), verdicts, and breakdowns by complexity, scale × index, source (LLM / rule / repair) and strategy family. Also correlations, cost-vs-time agreement, and failure/harm taxonomy. |
| 8 | Final output | `report.py` | `report.html` (KPIs, RQ answers, 7 figures, per-query cards with side-by-side SQL and plans, all candidates), `report.md`, CSVs, `summary.json`. |

### Verdicts

`improved` means the rewrite is equivalent and at least 10% faster (median). `regressed` means it is
equivalent but at least 10% slower, i.e. harmful. `neutral` is in between. The rejected verdicts are
`rejected_unsafe`, `rejected_syntax`, `rejected_unexecutable`, `rejected_nonequivalent` and `rejected_timeout`.
The pipeline only ever recommends an **improved** rewrite; otherwise it keeps the original query. This is how
semantic equivalence and performance improvement act as *two independent, jointly required success criteria*
(the research gap the proposal addresses).

## Research questions → metrics

| RQ | Question | Measured as |
|---|---|---|
| RQ1 | Can an LLM generate *executable* alternative formulations? | executable rate = passed safety + syntax + executability gates / generated |
| RQ2 | How often are rewrites *semantically equivalent*? | equivalent / executable (and / generated); non-equivalence type breakdown (row count, values, order, columns, types) |
| RQ3 | How much do valid rewrites *improve performance*? | improved / regressed share of valid rewrites; median and geometric-mean speedup; **pipeline speedup** (best verified rewrite per query, else 1.0) |
| RQ4 | How does *query complexity* affect correctness and performance? | all of the above per complexity level; Spearman correlation with the structural complexity score |
| RQ5 | How do *database scale and indexing* affect effectiveness? | pipeline speedup and improved runs per (scale × index configuration); heatmap and scale trend |

The report also covers plan-level causes: whether improving rewrites changed the plan shape, whether the
optimizer-cost ratio agreed with measured time, and which strategy families succeed or cause harm.

## Benchmark schema, data and workload

**Schema** (`schema.py`, 10 tables): `regions`, `nations`, `customers`, `suppliers`, `categories`
(self-referencing hierarchy), `products`, `orders`, `order_items`, `payments`, `reviews`. All tables have
primary and foreign keys.

**Data** (`datagen.py`) is deterministic for a given `(scale, seed)`. Scale 1.0 gives about 5k customers, 2k
products, 50k orders, 150k order lines and 55k payments. The data is designed to make rewriting non-trivial:
- NULLs in nullable columns, which break naive `NOT IN` → `NOT EXISTS` rewrites;
- Zipf-skewed customer and product popularity;
- customers without orders and products without sales (anti-join cases);
- duplicates that punish dropping `DISTINCT` or turning semi-joins into joins.

**Index configurations** (the independent variable for RQ5):

| name | indexes |
|---|---|
| `baseline` | primary keys only |
| `fk_indexes` | + one index per foreign key |
| `tuned` | + composite/covering indexes for the workload's predicates |

**Workload** (`workload/queries.yaml`):

| Level | Queries | Rewrite opportunities / traps |
|---|---|---|
| basic | Q01–Q05 | non-sargable `EXTRACT`, OR-chain, arithmetic on column, redundant `DISTINCT`, `HAVING` used as `WHERE` |
| intermediate | Q06–Q10 | `IN` semi-join, `NOT IN` anti-join, join + `DISTINCT`, correlated scalar subqueries (COUNT trap), `UNION` vs `OR` |
| advanced | Q11–Q15 | correlated aggregate, `EXISTS`+`NOT EXISTS`, duplicated derived table, quadratic top-N-per-group, `HAVING` subquery + top-N |
| complex | Q16–Q20 | multi-CTE + window, correlated running total (ROUND-placement trap), repeated correlated subqueries + CASE, 5-way-join `UNION ALL` roll-up, cohort CTEs + outer join |

Add your own queries to the YAML file. They are written in Oracle SQL and transpiled automatically for SQLite.

## CLI reference

```
python -m sqlrewriter setup     [-c CFG] [--scale 0.1,1] [--index-config fk_indexes] [--force]
python -m sqlrewriter workload  [-c CFG]
python -m sqlrewriter explain   [-c CFG] (-q ID | --sql "..." | --file f.sql) [--scale S] [--index-config X]
python -m sqlrewriter rewrite   [-c CFG] (-q ID | --sql "..." | --file f.sql) [--scale S] [--index-config X]
                                [--provider rule_based|groq|anthropic|ollama] [--model M] [--candidates N]
                                [--no-rule-baseline] [--no-cache] [--runs N] [-v] [--json out.json]
python -m sqlrewriter run       [-c CFG] [--name N] [--scales 0.1,1] [--index-configs a,b] [--queries Q01,Q02]
                                [--complexities basic,complex] [--provider ...] [--model ...] [--runs N]
python -m sqlrewriter report    [RUN_DIR]          # rebuild report (default: latest run)
python -m sqlrewriter check     [-c CFG] [--ping]  # database + LLM connectivity
python -m sqlrewriter oracle-scripts               # write sql/oracle/*.sql
```

After `pip install -e .` the command is also available as `sqlrewriter …`.

## Configuration

Configs are YAML files in `config/`. `${VAR:-default}` values are read from the environment or `.env`.

| file | backend | rewriter |
|---|---|---|
| `experiment.yaml` (default) | SQLite | rule-based |
| `experiment_groq.yaml` | SQLite | Groq (gpt-oss-120b) + rule baseline |
| `experiment_claude.yaml` | SQLite | Claude + rule baseline |
| `experiment_oracle_rules.yaml` | Oracle | rule-based |
| `experiment_oracle.yaml` | Oracle | Claude + rule baseline (**the proposed study**) |
| `experiment_oracle_groq.yaml` | Oracle | Groq (gpt-oss-120b) + rule baseline (the study on the free tier) |

The most important keys:

```yaml
experiment:
  scales: [0.1, 1.0, 5.0]
  index_configs: [baseline, fk_indexes, tuned]
  generation_mode: per_query   # generate candidates once per query (controlled comparison across conditions)
                               # or per_condition (regenerate with each condition's context)
llm:
  provider: anthropic          # rule_based | groq | anthropic | ollama
  model: claude-opus-5-5
  effort: high
  num_candidates: 3
  include_plan_in_prompt: true
  max_repair_rounds: 1
benchmark:
  warmup_runs: 1
  measured_runs: 5
  improvement_threshold: 0.10
equivalence:
  strict_column_names: true
  float_rel_tolerance: 1.0e-6
```

In `per_query` mode, candidates (including repaired ones) are generated under the first condition and then
**re-verified and re-benchmarked under every condition**. Equivalence therefore has to hold on every dataset
scale and index configuration, and the effect of scale and indexing is isolated from generation randomness.

## Project structure

```
├── sqlrewriter/
│   ├── cli.py, __main__.py      command-line interface
│   ├── config.py                YAML + env configuration
│   ├── schema.py                benchmark schema, index configurations, DDL rendering
│   ├── datagen.py               deterministic synthetic data generator
│   ├── workload.py              stage 1 – workload loading
│   ├── complexity.py            structural complexity metrics (RQ4)
│   ├── context.py               stage 2 – schema context for the prompt
│   ├── llm/                     stage 3 – rewriters (Groq, Claude, Ollama, rule-based) + prompts + cache
│   ├── validation.py            stage 4 – safety / syntax / executability gates
│   ├── equivalence.py           stage 5 – strict result-set comparison
│   ├── db/                      stage 6 – Oracle & SQLite backends, benchmarking, plans
│   ├── pipeline.py              stages 3–6 for one query (+ repair loop, verdicts)
│   ├── experiment.py            factorial experiment runner
│   ├── analysis.py              stage 7 – metrics and RQ answers
│   └── report.py                stage 8 – figures, HTML/Markdown report, CSV
├── app/streamlit_app.py         interactive dashboard
├── workload/queries.yaml        the 20-query workload
├── config/*.yaml                experiment configurations
├── sql/oracle/                  standalone Oracle DDL scripts
├── docker-compose.yml, docker/  local Oracle Database Free
├── tests/                       pytest suite (56 tests)
└── results/                     experiment outputs (created on run)
```

## Tests

```bash
python -m pytest -q
```

The tests cover the equivalence checker (bag semantics, float tolerance, tie-aware ordering, columns and
types), the safety gates, every rewrite rule, the data generator, all 20 workload queries executing, and the
full pipeline with a fake LLM that exercises every verdict path and the repair loop. They also check the shape
of the Claude and Groq API requests (with mocked clients), Groq's rate-limit waits and daily-limit cut-off, and run an end-to-end experiment through to the report.

## Limitations and threats to validity

- **Result-based equivalence is evidence, not proof.** Two queries can agree on one database state and differ
  on another. The framework reduces this risk in four ways: re-verifying on every scale and index
  configuration, data deliberately containing NULLs/duplicates/empty groups, conservative ordering checks,
  and flagging vacuous (empty) results. Formal verification, e.g. with the optimizer's memo (Narasayya et al.)
  or an SMT solver, is complementary future work.
- **Timing noise.** Medians over repeated runs after warm-up, plus a 10% threshold, guard against noise.
  Increase `measured_runs` for publication-quality numbers, and use V$SQL CPU/buffer-gets on Oracle as
  noise-free complements.
- **SQLite is a stand-in.** It has no cost-based optimizer cost numbers, and its planner differs from
  Oracle's. Use it for development and demos; the study's conclusions should come from the Oracle backend.
- **LLM non-determinism.** Responses are cached, so re-runs are reproducible. Use `--no-cache` to sample
  fresh candidates.

## References

1. S. Chen, R. Kumar, L. Zhao, "Query Rewriting via LLMs," arXiv:2502.12918, 2025.
2. J. Liu, "GenRewrite: Query Rewriting via Large Language Models," arXiv:2403.09060, 2024.
3. Y. Zhang, H. Wang, P. Li, "E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence and Efficiency," arXiv:2508.09023, 2025.
4. V. Narasayya et al., "Leveraging Query Optimizers to Verify the Soundness of LLM-based Query Rewrites," CIDR 2026.
5. X. Wang et al., "SQLGovernor: An LLM-powered SQL Toolkit for Real World Applications," arXiv:2509.08575, 2025.
6. S. Iyer et al., "Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL," arXiv:2506.09359, 2025.
7. Oracle Corporation, "Oracle Database SQL Tuning Guide," 2024.
