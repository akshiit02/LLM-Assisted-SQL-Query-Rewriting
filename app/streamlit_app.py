"""Interactive dashboard for the SQL rewriting study.

Run with:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlrewriter.analysis import analyze  # noqa: E402
from sqlrewriter.config import load_settings  # noqa: E402
from sqlrewriter.context import build_schema_context  # noqa: E402
from sqlrewriter.db import create_backend  # noqa: E402
from sqlrewriter.experiment import build_rewriters, load_results  # noqa: E402
from sqlrewriter.pipeline import run_single  # noqa: E402
from sqlrewriter.schema import INDEX_CONFIGURATIONS  # noqa: E402
from sqlrewriter.workload import adhoc_query, load_workload  # noqa: E402

st.set_page_config(page_title="LLM SQL Rewriter", layout="wide")

VERDICT_ICON = {"improved": "✅ improved", "neutral": "➖ neutral", "regressed": "🔻 regressed",
                "rejected_nonequivalent": "❌ not equivalent", "rejected_unexecutable": "⚠️ not executable",
                "rejected_syntax": "⚠️ syntax error", "rejected_unsafe": "⛔ unsafe", "rejected_timeout": "⏱️ timeout",
                "error": "❗ error"}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Configuration")
configs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").glob("*.yaml"))
cfg_path = st.sidebar.selectbox("Config file", configs,
                                index=configs.index("config\\experiment.yaml") if "config\\experiment.yaml" in configs
                                else (configs.index("config/experiment.yaml") if "config/experiment.yaml" in configs else 0))
base = load_settings(cfg_path)
backend_name = st.sidebar.selectbox("Database backend", ["sqlite", "oracle"],
                                    index=["sqlite", "oracle"].index(base.database.backend))
providers = ["rule_based", "groq", "anthropic", "ollama"]
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "anthropic": "claude-opus-5-5", "ollama": "qwen2.5-coder"}
KEY_VARS = {"groq": ("GROQ_API_KEY", "https://console.groq.com/keys"),
            "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys")}
provider = st.sidebar.selectbox("Rewriter", providers, index=providers.index(base.llm.provider)
                                if base.llm.provider in providers else 0)
model = st.sidebar.text_input("Model", base.llm.model if provider == base.llm.provider
                              else DEFAULT_MODELS.get(provider, ""), disabled=provider == "rule_based",
                              key=f"model_{provider}")
if provider in KEY_VARS:
    var, url = KEY_VARS[provider]
    if os.environ.get(var):
        st.sidebar.caption(f"🔑 {var} is set")
    else:
        entered = st.sidebar.text_input(var, type="password", help=f"Not found in the environment or .env. "
                                        f"Get a key at {url}. It is kept only in this session.")
        if entered:
            os.environ[var] = entered.strip()
rule_baseline = st.sidebar.checkbox("Also run rule-based baseline", value=provider != "rule_based",
                                    disabled=provider == "rule_based")
n_cand = st.sidebar.slider("LLM candidates", 1, 6, base.llm.num_candidates)
repair_rounds = st.sidebar.slider("Repair rounds", 0, 3, base.llm.max_repair_rounds)
scale = st.sidebar.select_slider("Dataset scale factor", options=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0], value=0.1)
index_config = st.sidebar.selectbox("Index configuration", list(INDEX_CONFIGURATIONS), index=1)
st.sidebar.caption(INDEX_CONFIGURATIONS[index_config].description)
runs = st.sidebar.slider("Measured runs per benchmark", 1, 15, base.benchmark.measured_runs)


def current_settings():
    return load_settings(cfg_path, {
        "database": {"backend": backend_name},
        "llm": {"provider": provider, "model": model, "num_candidates": n_cand,
                "also_run_rule_baseline": rule_baseline, "max_repair_rounds": repair_rounds},
        "benchmark": {"measured_runs": runs},
    })


st.title("LLM-Assisted Semantics-Preserving SQL Query Rewriting")
st.caption("Generate → validate → verify equivalence → benchmark → compare. Only rewrites that return exactly the "
           "same result are ever benchmarked.")

tab_rewrite, tab_results, tab_schema = st.tabs(["🔧 Rewrite a query", "📊 Experiment results", "🗂️ Schema & workload"])

# ------------------------------------------------------------------ rewrite tab
with tab_rewrite:
    s0 = current_settings()
    workload = load_workload(s0.experiment.workload_file)
    options = ["(custom SQL)"] + [f"{q.id} · {q.complexity} · {q.title}" for q in workload]
    choice = st.selectbox("Workload query", options, index=11)
    if choice == "(custom SQL)":
        sql_text = st.text_area("SQL (Oracle dialect – transpiled automatically for SQLite)", height=180,
                                value="SELECT c.customer_id, c.name\nFROM customers c\nWHERE c.customer_id IN "
                                      "(SELECT o.customer_id FROM orders o WHERE o.total_amount > 1000)\n"
                                      "ORDER BY c.customer_id")
        query = adhoc_query(sql_text) if sql_text.strip() else None
    else:
        query = workload[options.index(choice) - 1]
        st.caption(query.description)
        st.code(query.sql, language="sql")

    if st.button("Run pipeline", type="primary", disabled=query is None):
        s = current_settings()
        log_lines: list[str] = []
        status = st.status("Running pipeline…", expanded=True)
        try:
            backend = create_backend(s.database)
            status.write(f"Preparing {backend.display_name} data (scale {scale:g}) …")
            backend.ensure_dataset(scale, s.database.seed, s.database.load_batch_size, progress=status.write)
            backend.apply_index_configuration(index_config)
            rewriters = build_rewriters(s)
            res = run_single(s, backend, rewriters, query, scale, index_config, log=log_lines.append)
            backend.close()
            status.update(label="Pipeline finished", state="complete", expanded=False)
            st.session_state["last_result"] = res
            st.session_state["last_log"] = log_lines
        except Exception as exc:  # show configuration problems nicely
            status.update(label="Pipeline failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")

    res = st.session_state.get("last_result")
    if res is not None:
        op = res.original_performance
        if res.error:
            st.error(res.error)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original median", f"{op.median_ms:.2f} ms" if op else "–")
        c2.metric("Candidates", len(res.outcomes))
        c3.metric("Valid (equivalent)", sum(o.verdict in ("improved", "neutral", "regressed") for o in res.outcomes))
        best = res.best()
        c4.metric("Best speedup", f"{best.speedup:.2f}x" if best else "none")

        rows = []
        for o in res.outcomes:
            note = ""
            if not o.validation.passed:
                note = o.validation.error or ""
            elif o.equivalence and not o.equivalence.equivalent:
                note = o.equivalence.reason
            rows.append({"id": o.candidate.id, "source": o.candidate.source, "strategy": o.candidate.strategy,
                         "verdict": VERDICT_ICON.get(o.verdict, o.verdict),
                         "speedup": round(o.speedup, 3) if o.speedup else None,
                         "median ms": round(o.performance.median_ms, 3) if o.performance else None,
                         "cost ratio": round(o.cost_ratio, 3) if o.cost_ratio else None,
                         "plan changed": o.plan_changed, "note": note[:200]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        shown = best or next((o for o in res.outcomes if o.performance), None)
        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.code(res.backend_sql, language="sql")
            if op:
                st.text(op.plan_text)
        with right:
            if shown:
                st.subheader(f"{'Best rewrite' if best else 'Fastest valid rewrite'} – {shown.candidate.id}")
                st.caption(f"{shown.candidate.strategy} · {shown.verdict} · {shown.speedup:.2f}x")
                st.code(shown.candidate.sql, language="sql")
                st.text(shown.performance.plan_text)
            else:
                st.subheader("No valid rewrite")
                st.write("The pipeline keeps the original query.")

        for o in res.outcomes:
            with st.expander(f"{o.candidate.id} · {o.candidate.strategy} · {VERDICT_ICON.get(o.verdict, o.verdict)}"):
                st.write(o.candidate.rationale or "")
                st.code(o.candidate.sql, language="sql")
                if o.equivalence and not o.equivalence.equivalent:
                    st.warning(o.equivalence.reason)
                    if o.equivalence.only_in_original:
                        st.write("Rows only in the original:", o.equivalence.only_in_original)
                    if o.equivalence.only_in_rewrite:
                        st.write("Rows only in the rewrite:", o.equivalence.only_in_rewrite)
                if not o.validation.passed:
                    st.error(o.validation.error)
        with st.expander("Pipeline log"):
            st.text("\n".join(st.session_state.get("last_log", [])))

# ------------------------------------------------------------------ results tab
with tab_results:
    out_dir = base.resolve_path(base.output_dir)
    run_dirs = sorted((p for p in out_dir.glob("*") if (p / "results.json").exists() or (p / "results.jsonl").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        st.info("No experiment runs yet. Run `python -m sqlrewriter run` to create one.")
    else:
        run_dir = Path(st.selectbox("Experiment run", run_dirs, format_func=lambda p: p.name))
        results, meta = load_results(run_dir)
        a = analyze(results, meta)
        s = a["summary"]
        if s.get("empty"):
            st.warning("This run has no recorded outcomes.")
        else:
            o, p = s["overall"], s["pipeline"]
            k = st.columns(5)
            k[0].metric("Candidates", o["generated"])
            k[1].metric("Executable", f"{100 * (o['executable_rate'] or 0):.1f}%")
            k[2].metric("Equivalent (of exec.)", f"{100 * (o['equivalence_rate_of_executable'] or 0):.1f}%")
            k[3].metric("Improved (of valid)", f"{100 * (o['improved_rate_of_valid'] or 0):.1f}%")
            k[4].metric("Geomean pipeline speedup", f"{(p['geomean_pipeline_speedup'] or 1):.2f}x")
            st.subheader("Research questions")
            for key, v in s["rq_answers"].items():
                st.markdown(f"**{key}** – {v}")
            figs = sorted((run_dir / "figures").glob("*.png")) if (run_dir / "figures").exists() else []
            if figs:
                cols = st.columns(2)
                for i, f in enumerate(figs):
                    cols[i % 2].image(str(f), width="stretch")
            else:
                st.caption("Build figures with `python -m sqlrewriter report`.")
            st.subheader("By complexity")
            st.dataframe(a["by_complexity"], width="stretch", hide_index=True)
            st.subheader("By scale and index configuration")
            st.dataframe(a["runs_by_condition"], width="stretch", hide_index=True)
            st.subheader("All candidate outcomes")
            st.dataframe(a["outcomes"], width="stretch", hide_index=True)
            st.download_button("Download outcomes.csv", a["outcomes"].to_csv(index=False), "outcomes.csv")
            if (run_dir / "report.html").exists():
                st.caption(f"Full HTML report: {run_dir / 'report.html'}")

# ------------------------------------------------------------------ schema tab
with tab_schema:
    s = current_settings()
    try:
        backend = create_backend(s.database)
        backend.ensure_dataset(scale, s.database.seed, s.database.load_batch_size)
        backend.apply_index_configuration(index_config)
        ctx = build_schema_context(backend, scale, index_config)
        backend.close()
        st.subheader("Schema context sent to the LLM")
        st.code(ctx.text, language="text")
    except Exception as exc:
        st.error(f"Cannot read schema: {exc}")
    st.subheader("Workload")
    st.dataframe(pd.DataFrame([{"id": q.id, "complexity": q.complexity, "title": q.title, "tags": ", ".join(q.tags)}
                               for q in load_workload(s.experiment.workload_file)]),
                 width="stretch", hide_index=True)
