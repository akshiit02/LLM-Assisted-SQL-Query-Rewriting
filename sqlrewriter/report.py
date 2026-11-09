"""Stage 8 – Final output: figures, HTML / Markdown report and CSV exports."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import analyze  # noqa: E402
from .experiment import load_results  # noqa: E402
from .models import COMPLEXITY_LEVELS  # noqa: E402

# ---- palette (validated reference palette; see README "Figures") -----------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SOURCE_COLORS = {"llm": SERIES[0], "rule": SERIES[1], "repair": SERIES[2]}
VERDICT_COLORS = {
    "improved": "#0ca30c",
    "neutral": "#b8b7b1",
    "regressed": "#d03b3b",
    "rejected_nonequivalent": "#ec835a",
    "rejected_unexecutable": "#fab219",
    "rejected_syntax": "#f5d27a",
    "rejected_timeout": "#6e6d69",
    "rejected_unsafe": "#2f2f2d",
    "error": "#8c8b86",
}
VERDICT_ORDER = list(VERDICT_COLORS)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w=7.2, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---- figures ------------------------------------------------------------------------

def fig_funnel(df: pd.DataFrame, path: Path) -> Optional[str]:
    stages = [("generated", None), ("executable", "executable"), ("equivalent", "equivalent"),
              ("improved", "improved")]
    sources = [s for s in ("llm", "repair", "rule") if s in set(df["source"])]
    if not sources:
        return None
    fig, ax = _fig(7.2, 0.9 + 0.55 * len(stages) * len(sources) / 2 + 1)
    height = 0.8 / len(sources)
    y = np.arange(len(stages))
    for i, src in enumerate(sources):
        g = df[df["source"] == src]
        vals = [len(g) if col is None else int(g[col].sum()) for _, col in stages]
        pos = y + (i - (len(sources) - 1) / 2) * height
        ax.barh(pos, vals, height=height * 0.86, color=SOURCE_COLORS[src], label=src.upper() if src == "llm" else src)
        for p, v in zip(pos, vals):
            ax.text(v, p, f" {v}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [s for s, _ in stages])
    ax.invert_yaxis()
    _style(ax, "Rewrite funnel: generated -> executable -> equivalent -> improved", "candidates")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def fig_verdicts_by_complexity(df: pd.DataFrame, path: Path) -> Optional[str]:
    levels = [c for c in COMPLEXITY_LEVELS if c in set(df["complexity"])]
    if not levels:
        return None
    fig, ax = _fig(7.2, 1.2 + 0.5 * len(levels))
    left = np.zeros(len(levels))
    present = [v for v in VERDICT_ORDER if v in set(df["verdict"])]
    for v in present:
        shares = []
        for lv in levels:
            g = df[df["complexity"] == lv]
            shares.append(100 * (g["verdict"] == v).mean() if len(g) else 0)
        shares = np.array(shares)
        ax.barh(levels, shares, left=left, color=VERDICT_COLORS[v], edgecolor=SURFACE, linewidth=1.5,
                label=v.replace("rejected_", "rej. ").replace("_", " "))
        for i, s in enumerate(shares):
            if s >= 7:
                ax.text(left[i] + s / 2, i, f"{s:.0f}%", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v in ("regressed", "rejected_unsafe", "rejected_timeout", "improved") else INK)
        left += shares
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    _style(ax, "Candidate verdicts by query complexity", "% of candidates")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    return _save(fig, path)


def fig_speedup_by_complexity(df: pd.DataFrame, path: Path) -> Optional[str]:
    valid = df[df["valid"] & df["speedup"].notna()]
    levels = [c for c in COMPLEXITY_LEVELS if c in set(valid["complexity"])]
    if not levels:
        return None
    fig, ax = _fig(7.2, 3.6)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        vals = valid.loc[valid["complexity"] == lv, "speedup"].clip(lower=1e-3).to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        colors = [VERDICT_COLORS["improved"] if v >= 1.1 else VERDICT_COLORS["regressed"] if v <= 1 / 1.1
                  else VERDICT_COLORS["neutral"] for v in vals]
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, c=colors, edgecolors=SURFACE, linewidths=1.2,
                   zorder=3)
        med = float(np.median(vals))
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.33, med, f"{med:.2f}x", va="center", fontsize=7.5, color=INK)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--", zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(levels)), levels)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, color=VERDICT_COLORS[k], label=k)
               for k in ("improved", "neutral", "regressed")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=3)
    _style(ax, "Speedup of valid (equivalent) rewrites by complexity  (bar = median)", "", "speedup (log scale)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def fig_condition_heatmap(runs_cond: pd.DataFrame, path: Path) -> Optional[str]:
    if runs_cond.empty:
        return None
    pivot = runs_cond.pivot(index="index_config", columns="scale", values="geomean_pipeline_speedup")
    fig, ax = _fig(max(5.6, 1.6 + 1.3 * pivot.shape[1]), 1.2 + 0.6 * pivot.shape[0])
    data = pivot.to_numpy(dtype=float)
    vmax = max(1.05, float(np.nanmax(data)))
    im = ax.imshow(data, cmap=SEQ_BLUE, vmin=1.0, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        if not math.isnan(v):
            ax.text(c, r, f"{v:.2f}x", ha="center", va="center", fontsize=9,
                    color="#ffffff" if (v - 1) / (vmax - 1 + 1e-9) > 0.55 else INK)
    ax.set_xticks(range(pivot.shape[1]), [f"sf={s:g}" for s in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    _style(ax, "Geomean pipeline speedup: scale x index")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    return _save(fig, path)


def fig_scale_trend(runs_cond: pd.DataFrame, path: Path) -> Optional[str]:
    if runs_cond.empty or runs_cond["scale"].nunique() < 2:
        return None
    fig, ax = _fig(7.2, 3.4)
    for i, (idx, g) in enumerate(runs_cond.groupby("index_config", sort=False)):
        g = g.sort_values("scale")
        ax.plot(g["scale"], g["geomean_pipeline_speedup"], color=SERIES[i % 3], linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=idx)
        last = g.iloc[-1]
        ax.text(last["scale"], last["geomean_pipeline_speedup"], f"  {idx}", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(sorted(runs_cond["scale"].unique()), [f"{s:g}" for s in sorted(runs_cond["scale"].unique())])
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    _style(ax, "Pipeline speedup vs dataset scale, per index configuration", "scale factor", "geomean speedup")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def fig_per_query(runs: pd.DataFrame, path: Path) -> Optional[str]:
    ok = runs[runs["error"].isna() & runs["best_valid_speedup"].notna()]
    if ok.empty:
        return None
    from .analysis import geomean

    per_q = ok.groupby("query_id")["best_valid_speedup"].apply(geomean).sort_index()
    fig, ax = _fig(7.2, 0.8 + 0.26 * len(per_q))
    vals = per_q.to_numpy()
    colors = [SERIES[0] if v >= 1 else SERIES[7] for v in vals]
    ax.barh(per_q.index, np.log10(vals), color=colors, height=0.62)
    for i, v in enumerate(vals):
        ax.text(np.log10(v), i, f" {v:.2f}x " if v >= 1 else f" {v:.2f}x ", va="center",
                ha="left" if v >= 1 else "right", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    lim = max(0.3, float(np.max(np.abs(np.log10(vals)))) * 1.25)
    ax.set_xlim(-lim, lim)
    nice = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000]
    ticks = [math.log10(v) for v in nice if -lim <= math.log10(v) <= lim]
    ax.set_xticks(ticks, [f"{10 ** t:g}x" for t in ticks])
    ax.invert_yaxis()
    _style(ax, "Best valid rewrite per query (geomean over conditions)", "speedup (log scale; left of 1x = slower)")
    return _save(fig, path)


def fig_failures(df: pd.DataFrame, path: Path) -> Optional[str]:
    rej = df[df["verdict"].str.startswith("rejected")]
    if rej.empty:
        return None
    labels = []
    for _, r in rej.iterrows():
        if r["verdict"] == "rejected_nonequivalent":
            labels.append(f"not equivalent: {r['equivalence_category']}")
        elif r["verdict"] == "rejected_timeout":
            labels.append("timeout during verification")
        else:
            labels.append(f"{r['verdict'].replace('rejected_', '')}: {r['error_category'] or 'other'}")
    counts = pd.Series(labels).value_counts().sort_values()
    fig, ax = _fig(7.2, 0.8 + 0.3 * len(counts))
    ax.barh(counts.index, counts.to_numpy(), color=SERIES[1], height=0.6)
    for i, v in enumerate(counts.to_numpy()):
        ax.text(v, i, f" {v}", va="center", fontsize=7.5, color=INK2)
    _style(ax, "Why candidates were rejected", "candidates")
    return _save(fig, path)


# ---- report ---------------------------------------------------------------------------

def _fmt(v: Any, kind: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "x":
        return f"{v:.2f}x"
    if kind == "ms":
        return f"{v:,.2f}"
    if isinstance(v, float):
        return f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return html.escape(str(v))


def _query_cards(results: list[dict], runs: pd.DataFrame) -> list[dict]:
    """One card per query: its most instructive condition (largest pipeline speedup, else largest scale)."""
    by_q: dict[str, list[dict]] = {}
    for r in results:
        by_q.setdefault(r["query"]["id"], []).append(r)
    cards = []
    for qid in sorted(by_q):
        rs = by_q[qid]

        def score(r):
            best = r.get("best_candidate_id")
            sp = next((o["speedup"] for o in r["outcomes"] if o["candidate"]["id"] == best), 1.0) if best else 1.0
            return (sp or 1.0, r["condition"]["scale"])

        pick = max(rs, key=score)
        best = next((o for o in pick["outcomes"] if o["candidate"]["id"] == pick.get("best_candidate_id")), None)
        if best is None:  # show the fastest valid rewrite even if it didn't clear the threshold
            valid = [o for o in pick["outcomes"] if o["verdict"] in ("improved", "neutral", "regressed")]
            best = max(valid, key=lambda o: o.get("speedup") or 0) if valid else None
        cond_rows = runs[runs["query_id"] == qid].sort_values(["scale", "index_config"]).to_dict("records")
        cards.append({"query": pick["query"], "run": pick, "best": best, "conditions": cond_rows,
                      "metrics": pick.get("complexity_metrics", {})})
    return cards


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SQL Rewrite Study – {{ meta.name }}</title>
<style>
:root{--bg:#f6f6f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8c8b86;--line:#e4e3df;--accent:#2a78d6;
--good:#0ca30c;--bad:#d03b3b;--warn:#b77900;--code:#f0efec;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#8c8b86;--line:#2e2e2c;--accent:#3987e5;--code:#232322;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8c8b86;--line:#2e2e2c;--accent:#3987e5;--code:#232322;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:18px 0 8px}.sub{color:var(--ink2)}.muted{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--ink2);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;background:var(--surface);position:sticky;top:0}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}pre{background:var(--code);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:12px;line-height:1.45;margin:6px 0;white-space:pre}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.two{grid-template-columns:1fr}}
.fig{background:#fcfcfb;border-radius:8px;padding:6px;border:1px solid var(--line)}.fig img{width:100%;display:block}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;border:1px solid var(--line);margin-right:4px;white-space:nowrap}
.v-improved{color:var(--good);font-weight:600}.v-regressed{color:var(--bad);font-weight:600}.v-neutral{color:var(--ink2)}
.v-rej{color:var(--warn)}details summary{cursor:pointer;color:var(--accent);font-weight:600}
ol.rq li{margin:8px 0}.rq b{color:var(--accent)}
</style></head><body><main>
<p class="muted">Advanced SQL and Modern Databases · SRM Institute of Science and Technology · School of Computing</p>
<h1>LLM-Assisted Semantics-Preserving SQL Query Rewriting</h1>
<p class="sub">Experiment <b>{{ meta.name }}</b> · backend <b>{{ meta.backend }}</b> · rewriters <b>{{ meta.rewriters|join(', ') }}</b>
{% if meta.llm_model %}· model <b>{{ meta.llm_model }}</b>{% endif %} · {{ meta.started_at }} · {{ meta.duration_s }} s</p>

<div class="grid">
 <div class="card kpi"><div class="v">{{ o.generated }}</div><div class="l">candidate rewrites generated</div></div>
 <div class="card kpi"><div class="v">{{ fmt(o.executable_rate,'pct') }}</div><div class="l">executable (RQ1)</div></div>
 <div class="card kpi"><div class="v">{{ fmt(o.equivalence_rate_of_executable,'pct') }}</div><div class="l">equivalent, of executable (RQ2)</div></div>
 <div class="card kpi"><div class="v">{{ fmt(o.improved_rate_of_valid,'pct') }}</div><div class="l">of valid rewrites were faster (RQ3)</div></div>
 <div class="card kpi"><div class="v">{{ fmt(p.geomean_pipeline_speedup,'x') }}</div><div class="l">geomean pipeline speedup ({{ p.runs_with_improvement }}/{{ p.runs }} runs improved)</div></div>
 <div class="card kpi"><div class="v">{{ fmt(o.regressed_rate_of_valid,'pct') }}</div><div class="l">valid but harmful (slower)</div></div>
</div>

<h2>Research questions</h2>
<div class="card"><ol class="rq">
{% for k, v in rq.items() %}<li><b>{{ k }}</b> – {{ v }}</li>{% endfor %}
</ol></div>

<h2>Figures</h2>
<div class="two">
{% for f in figures %}<div class="fig"><img alt="{{ f.alt }}" src="data:image/png;base64,{{ f.b64 }}"></div>{% endfor %}
</div>

<h2>Results by source</h2>
<div class="card scroll"><table><tr><th>source</th><th class="num">generated</th><th class="num">executable</th><th class="num">equivalent</th>
<th class="num">improved</th><th class="num">regressed</th><th class="num">exec rate</th><th class="num">equiv rate (of exec)</th><th class="num">improved (of valid)</th><th class="num">geomean speedup (valid)</th></tr>
{% for r in by_source %}<tr><td>{{ r.source }}</td><td class="num">{{ r.generated }}</td><td class="num">{{ r.executable }}</td><td class="num">{{ r.equivalent }}</td>
<td class="num">{{ r.improved }}</td><td class="num">{{ r.regressed }}</td><td class="num">{{ fmt(r.executable_rate,'pct') }}</td><td class="num">{{ fmt(r.equivalence_rate_of_executable,'pct') }}</td>
<td class="num">{{ fmt(r.improved_rate_of_valid,'pct') }}</td><td class="num">{{ fmt(r.geomean_speedup_valid,'x') }}</td></tr>{% endfor %}
</table></div>

<h2>Results by complexity (RQ4)</h2>
<div class="card scroll"><table><tr><th>complexity</th><th class="num">candidates</th><th class="num">exec rate</th><th class="num">equiv rate (all)</th><th class="num">improved (of valid)</th><th class="num">regressed (of valid)</th><th class="num">median speedup</th><th class="num">geomean speedup</th></tr>
{% for r in by_cx %}<tr><td>{{ r.complexity }}</td><td class="num">{{ r.generated }}</td><td class="num">{{ fmt(r.executable_rate,'pct') }}</td><td class="num">{{ fmt(r.equivalence_rate_overall,'pct') }}</td>
<td class="num">{{ fmt(r.improved_rate_of_valid,'pct') }}</td><td class="num">{{ fmt(r.regressed_rate_of_valid,'pct') }}</td><td class="num">{{ fmt(r.median_speedup_valid,'x') }}</td><td class="num">{{ fmt(r.geomean_speedup_valid,'x') }}</td></tr>{% endfor %}
</table></div>

<h2>Results by scale and index configuration (RQ5)</h2>
<div class="card scroll"><table><tr><th>scale</th><th>index configuration</th><th class="num">query runs</th><th class="num">runs improved</th><th class="num">geomean pipeline speedup</th><th class="num">median original ms</th></tr>
{% for r in runs_cond %}<tr><td>{{ fmt(r.scale) }}</td><td>{{ r.index_config }}</td><td class="num">{{ r.runs }}</td><td class="num">{{ r.improved_runs }}</td><td class="num">{{ fmt(r.geomean_pipeline_speedup,'x') }}</td><td class="num">{{ fmt(r.median_orig_ms,'ms') }}</td></tr>{% endfor %}
</table></div>

<h2>Rewrite strategies</h2>
<div class="card scroll"><table><tr><th>strategy family</th><th class="num">generated</th><th class="num">equivalent</th><th class="num">improved</th><th class="num">regressed</th><th class="num">geomean speedup (valid)</th></tr>
{% for r in by_strategy %}<tr><td>{{ r.strategy_family }}</td><td class="num">{{ r.generated }}</td><td class="num">{{ r.equivalent }}</td><td class="num">{{ r.improved }}</td><td class="num">{{ r.regressed }}</td><td class="num">{{ fmt(r.geomean_speedup_valid,'x') }}</td></tr>{% endfor %}
</table></div>

<h2>Failure and harmful-rewrite analysis</h2>
<div class="card">
<p>Verdicts: {% for k, v in verdicts.items() %}<span class="tag">{{ k }}: {{ v }}</span>{% endfor %}</p>
<p>Validation error categories: {% for k, v in s.error_categories.items() %}<span class="tag">{{ k }}: {{ v }}</span>{% else %}none{% endfor %}</p>
<p>Equivalence failure types: {% for k, v in s.equivalence_failures.items() %}<span class="tag">{{ k }}: {{ v }}</span>{% else %}none{% endfor %}</p>
{% if s.repair.attempts %}<p>Repair loop: {{ s.repair.attempts }} repaired candidates, {{ s.repair.now_valid }} became valid, {{ s.repair.now_improved }} improved.</p>{% endif %}
{% if s.plan_change %}<p>Plan changes: {{ s.plan_change.improved_with_plan_change }} improved rewrites changed the execution plan; {{ s.plan_change.improved_without_plan_change }} improved without a plan-shape change.</p>{% endif %}
{% if s.cost_vs_time %}<p>Optimizer cost vs measured time: Spearman ρ(cost ratio, speedup) = {{ s.cost_vs_time.spearman_cost_ratio_vs_speedup }}; {{ s.cost_vs_time.cheaper_but_slower }} rewrites had a lower estimated cost but ran slower, {{ s.cost_vs_time.costlier_but_faster }} had a higher cost but ran faster.</p>{% endif %}
<details><summary>Harmful rewrites (valid but slower) – {{ harmful|length }}</summary><div class="scroll"><table>
<tr><th>query</th><th>condition</th><th>candidate</th><th>strategy</th><th class="num">speedup</th></tr>
{% for h in harmful %}<tr><td>{{ h.query_id }}</td><td>sf={{ fmt(h.scale) }} / {{ h.index_config }}</td><td>{{ h.candidate_id }} ({{ h.source }})</td><td>{{ h.strategy }}</td><td class="num v-regressed">{{ fmt(h.speedup,'x') }}</td></tr>{% endfor %}
</table></div></details>
<details><summary>Non-equivalent rewrites – {{ nonequiv|length }}</summary><div class="scroll"><table>
<tr><th>query</th><th>candidate</th><th>strategy</th><th>reason</th></tr>
{% for h in nonequiv %}<tr><td>{{ h.query_id }}</td><td>{{ h.candidate_id }} ({{ h.source }})</td><td>{{ h.strategy }}</td><td>{{ h.reason }}</td></tr>{% endfor %}
</table></div></details>
</div>

<h2>Per-query results</h2>
{% for c in cards %}
<div class="card">
<h3>{{ c.query.id }} · {{ c.query.title }} <span class="tag">{{ c.query.complexity }}</span><span class="tag">score {{ c.metrics.score }}</span></h3>
<p class="muted">{{ c.query.description }}</p>
<div class="scroll"><table><tr><th>scale</th><th>index</th><th class="num">original ms</th><th class="num">candidates</th><th class="num">valid</th><th class="num">improved</th><th class="num">best valid speedup</th><th>best</th></tr>
{% for r in c.conditions %}<tr><td>{{ fmt(r.scale) }}</td><td>{{ r.index_config }}</td><td class="num">{{ fmt(r.orig_ms,'ms') }}</td><td class="num">{{ r.n_candidates }}</td><td class="num">{{ r.n_valid }}</td><td class="num">{{ r.n_improved }}</td><td class="num">{{ fmt(r.best_valid_speedup,'x') }}</td><td>{{ r.best_candidate_id or '–' }}</td></tr>{% endfor %}
</table></div>
<p class="muted">Details below: sf={{ fmt(c.run.condition.scale) }}, index configuration {{ c.run.condition.index_config }}.</p>
<div class="two">
 <div><b>Original</b>{% set op = c.run.original_performance %}{% if op %} <span class="muted">median {{ fmt(op.median_ms,'ms') }} ms{% if op.optimizer_cost %} · cost {{ fmt(op.optimizer_cost) }}{% endif %}{% if op.buffer_gets %} · buffer gets {{ fmt(op.buffer_gets) }}{% endif %}{% if op.work_units %} · VM steps {{ fmt(op.work_units) }}{% endif %}</span>{% endif %}
 <pre>{{ c.run.backend_sql }}</pre></div>
 <div>{% if c.best %}{% set bp = c.best.performance %}<b>Best rewrite {{ c.best.candidate.id }}</b> <span class="muted">{{ c.best.candidate.strategy }} · <span class="v-{{ c.best.verdict }}">{{ c.best.verdict }} {{ fmt(c.best.speedup,'x') }}</span>{% if bp %} · median {{ fmt(bp.median_ms,'ms') }} ms{% if bp.optimizer_cost %} · cost {{ fmt(bp.optimizer_cost) }}{% endif %}{% if bp.work_units %} · VM steps {{ fmt(bp.work_units) }}{% endif %}{% endif %}</span>
 <pre>{{ c.best.candidate.sql }}</pre><p class="muted">{{ c.best.candidate.rationale }}</p>{% else %}<b>No valid rewrite</b><p class="muted">The pipeline keeps the original query.</p>{% endif %}</div>
</div>
{% if c.best and c.best.performance %}
<details><summary>Execution plan comparison</summary><div class="two">
<div><pre>{{ c.run.original_performance.plan_text }}</pre></div><div><pre>{{ c.best.performance.plan_text }}</pre></div></div></details>
{% endif %}
<details><summary>All candidates ({{ c.run.outcomes|length }})</summary><div class="scroll"><table>
<tr><th>id</th><th>source</th><th>strategy</th><th>verdict</th><th class="num">speedup</th><th class="num">cost ratio</th><th>notes</th></tr>
{% for o in c.run.outcomes %}<tr><td>{{ o.candidate.id }}</td><td>{{ o.candidate.source }}</td><td>{{ o.candidate.strategy }}</td>
<td class="{% if o.verdict.startswith('rejected') %}v-rej{% else %}v-{{ o.verdict }}{% endif %}">{{ o.verdict }}</td>
<td class="num">{{ fmt(o.speedup,'x') }}</td><td class="num">{{ fmt(o.cost_ratio) }}</td>
<td>{% if not o.validation.passed %}{{ (o.validation.error or '')[:220] }}{% elif o.equivalence and not o.equivalence.equivalent %}{{ o.equivalence.reason }}{% endif %}</td></tr>
{% endfor %}</table></div>
{% for o in c.run.outcomes %}<details><summary>{{ o.candidate.id }} SQL</summary><pre>{{ o.candidate.sql }}</pre></details>{% endfor %}
</details>
</div>
{% endfor %}

<h2>Method</h2>
<div class="card"><p>Each candidate passes four gates: (1) safety – exactly one read-only SELECT; (2) syntax – sqlglot parse,
with the database parser as final judge; (3) executability – compiled by the database without execution; (4) semantic
equivalence – both queries are executed and the result sets compared strictly (columns, type categories, multiset of rows,
tie-aware row order when the original has ORDER BY). Only equivalent rewrites are benchmarked:
{{ bench.warmup_runs }} warm-up + {{ bench.measured_runs }} measured runs, median wall-clock time; a rewrite is
<i>improved</i> when it is at least {{ (bench.improvement_threshold*100)|round|int }}% faster and <i>regressed</i> when it is
equally slower. On Oracle the evaluation also records EXPLAIN PLAN cost/cardinality/CPU/IO, DBMS_XPLAN plans and V$SQL
runtime statistics.</p></div>
<p class="muted">Generated by sqlrewriter {{ meta.tool_version }}.</p>
</main></body></html>
"""


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    results, meta = load_results(run_dir)
    a = analyze(results, meta)
    s = a["summary"]
    if s.get("empty"):
        raise ValueError(f"No candidate outcomes recorded in {run_dir}")
    df, runs = a["outcomes"], a["runs"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    figures = []
    for fn, name, alt, args in [
        (fig_funnel, "funnel.png", "Rewrite funnel by source", (df,)),
        (fig_verdicts_by_complexity, "verdicts_by_complexity.png", "Verdicts by complexity", (df,)),
        (fig_speedup_by_complexity, "speedup_by_complexity.png", "Speedup by complexity", (df,)),
        (fig_per_query, "per_query_speedup.png", "Best speedup per query", (runs,)),
        (fig_condition_heatmap, "condition_heatmap.png", "Speedup by scale and index", (a["runs_by_condition"],)),
        (fig_scale_trend, "scale_trend.png", "Speedup vs scale", (a["runs_by_condition"],)),
        (fig_failures, "failures.png", "Rejection reasons", (df,)),
    ]:
        b64 = fn(*args, fig_dir / name)
        if b64:
            figures.append({"b64": b64, "alt": alt, "file": f"figures/{name}"})

    harmful = df[df["regressed"]].sort_values("speedup").to_dict("records")
    nonequiv = []
    for r in results:
        for o in r["outcomes"]:
            if o["verdict"] == "rejected_nonequivalent":
                nonequiv.append({"query_id": r["query"]["id"], "candidate_id": o["candidate"]["id"],
                                 "source": o["candidate"]["source"], "strategy": o["candidate"]["strategy"],
                                 "reason": (o.get("equivalence") or {}).get("reason", "")})
    # de-duplicate (the same candidate is re-verified under every condition)
    seen, uniq = set(), []
    for n in nonequiv:
        k = (n["query_id"], n["candidate_id"], n["reason"])
        if k not in seen:
            seen.add(k)
            uniq.append(n)

    env = Environment(autoescape=True)
    env.globals["fmt"] = _fmt
    bench = (meta.get("settings") or {}).get("benchmark", {"warmup_runs": "?", "measured_runs": "?",
                                                           "improvement_threshold": 0.1})
    page = env.from_string(HTML_TEMPLATE).render(
        meta={**{"name": run_dir.name, "backend": "?", "rewriters": [], "started_at": "", "duration_s": "",
                 "tool_version": ""}, **meta},
        o=s["overall"], p=s["pipeline"], rq=s["rq_answers"], s=s, figures=figures,
        by_source=s["by_source"], by_cx=a["by_complexity"].to_dict("records"),
        runs_cond=a["runs_by_condition"].to_dict("records"), by_strategy=a["by_strategy"].to_dict("records"),
        verdicts=s["verdict_counts"], harmful=harmful, nonequiv=uniq,
        cards=_query_cards(results, runs), bench=bench,
    )
    report_path = run_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")

    df.to_csv(run_dir / "outcomes.csv", index=False)
    runs.to_csv(run_dir / "runs.csv", index=False)
    a["by_complexity"].to_csv(run_dir / "by_complexity.csv", index=False)
    a["runs_by_condition"].to_csv(run_dir / "by_condition.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(s), indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(_markdown(s, a, figures), encoding="utf-8")
    return report_path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    return obj


def _markdown(s: dict, a: dict, figures: list[dict]) -> str:
    o, p = s["overall"], s["pipeline"]
    lines = [
        f"# SQL rewrite study – {s['meta'].get('name', '')}",
        "",
        f"Backend: **{s['meta'].get('backend')}** · rewriters: **{', '.join(s['meta'].get('rewriters', []))}**",
        "",
        "## Headline numbers",
        "",
        "| metric | value |",
        "|---|---|",
        f"| candidates generated | {o['generated']} |",
        f"| executable | {o['executable']} ({_fmt(o['executable_rate'], 'pct')}) |",
        f"| semantically equivalent | {o['equivalent']} ({_fmt(o['equivalence_rate_of_executable'], 'pct')} of executable) |",
        f"| improved (of valid) | {o['improved']} ({_fmt(o['improved_rate_of_valid'], 'pct')}) |",
        f"| regressed / harmful (of valid) | {o['regressed']} ({_fmt(o['regressed_rate_of_valid'], 'pct')}) |",
        f"| geomean speedup of valid rewrites | {_fmt(o['geomean_speedup_valid'], 'x')} |",
        f"| query runs improved by the pipeline | {p['runs_with_improvement']} / {p['runs']} |",
        f"| geomean pipeline speedup | {_fmt(p['geomean_pipeline_speedup'], 'x')} |",
        "",
        "## Research questions",
        "",
    ]
    for k, v in s["rq_answers"].items():
        lines += [f"**{k}.** {v}", ""]
    lines += ["## Figures", ""]
    for f in figures:
        lines.append(f"![{f['alt']}]({f['file']})")
    lines.append("")
    return "\n".join(lines)


def latest_run_dir(output_dir: Path) -> Optional[Path]:
    runs = sorted((p for p in output_dir.glob("*") if (p / "results.jsonl").exists() or (p / "results.json").exists()),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None
