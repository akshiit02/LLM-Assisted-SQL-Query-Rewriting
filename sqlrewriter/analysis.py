"""Stage 7 - Comparative analysis and answers to the research questions.

Works on the JSON records written by the experiment runner, so reports can be
regenerated at any time without re-running queries.

Terminology
-----------
* **executable**  - passed the safety/syntax gates and ran without error
* **equivalent**  - executable *and* its result set matched the original
* **valid**       - equivalent and benchmarked (verdict improved/neutral/regressed)
* **improved**    - valid and at least ``improvement_threshold`` faster (median)
* **regressed**   - valid but slower by the same margin (a *harmful* rewrite)
* **pipeline speedup** - per (query, condition), the best valid rewrite's
  speedup, or 1.0 when no improving rewrite exists (the pipeline keeps the
  original).  Because unsafe/slow rewrites are filtered, this is never < 1.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from .models import COMPLEXITY_LEVELS

REJECTED = ("rejected_unsafe", "rejected_syntax", "rejected_unexecutable", "rejected_timeout",
            "rejected_nonequivalent")
VALID = ("improved", "neutral", "regressed")

STRATEGY_FAMILIES = [
    ("sargable predicate", r"sarg|range|date|extract|arithmetic|function on|index"),
    ("window function", r"window|over\s*\(|rank|row_number|partition|running|lag|lead|analytic"),
    ("decorrelation / join", r"decorrel|unnest|scalar|lateral|join|group(ed)? derived|outer join"),
    ("semi/anti-join (EXISTS/IN)", r"exists|semi|anti|\bin\b|not in|in_to|in list"),
    ("CTE / factoring", r"cte|with clause|factor|materiali|subquer(y|ies)_to|hoist|common"),
    ("set operation", r"union|intersect|minus|except|\bor\b|or_to"),
    ("aggregation", r"aggregat|group|having|rollup|grouping sets|distinct|single pass|conditional"),
    ("predicate pushdown / merge", r"push|merge|inline|flatten"),
]


def strategy_family(strategy: str) -> str:
    s = (strategy or "").lower().replace("_", " ")
    for family, pattern in STRATEGY_FAMILIES:
        if re.search(pattern, s):
            return family
    return "other"


def _get(d: Optional[dict], *path, default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def outcomes_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        q, cond = r["query"], r["condition"]
        orig = r.get("original_performance") or {}
        for o in r.get("outcomes", []):
            c, v, eq, perf = o["candidate"], o["validation"], o.get("equivalence"), o.get("performance") or {}
            verdict = o["verdict"]
            executable = verdict not in ("rejected_unsafe", "rejected_syntax", "rejected_unexecutable")
            rows.append({
                "query_id": q["id"],
                "title": q.get("title", ""),
                "complexity": q["complexity"],
                "complexity_score": _get(r, "complexity_metrics", "score"),
                "scale": cond["scale"],
                "index_config": cond["index_config"],
                "backend": cond["backend"],
                "candidate_id": c["id"],
                "source": c["source"],
                "generator": c.get("generator", ""),
                "strategy": c.get("strategy", ""),
                "strategy_family": strategy_family(c.get("strategy", "")),
                "parent_id": c.get("parent_id"),
                "verdict": verdict,
                "validation_stage": v.get("stage"),
                "error_category": v.get("error_category"),
                "error": v.get("error") if not v.get("passed") else None,
                "executable": executable,
                "equivalent": bool(eq and eq.get("equivalent")),
                "equivalence_category": eq.get("category") if eq else None,
                "vacuous": bool(eq and eq.get("vacuous")),
                "valid": verdict in VALID,
                "improved": verdict == "improved",
                "regressed": verdict == "regressed",
                "speedup": o.get("speedup"),
                "cost_ratio": o.get("cost_ratio"),
                "plan_changed": o.get("plan_changed"),
                "orig_ms": orig.get("median_ms"),
                "cand_ms": perf.get("median_ms"),
                "orig_cost": orig.get("optimizer_cost"),
                "cand_cost": perf.get("optimizer_cost"),
                "orig_buffer_gets": orig.get("buffer_gets"),
                "cand_buffer_gets": perf.get("buffer_gets"),
                "orig_work": orig.get("work_units"),
                "cand_work": perf.get("work_units"),
            })
    return pd.DataFrame(rows)


def runs_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        q, cond = r["query"], r["condition"]
        orig = r.get("original_performance") or {}
        outs = r.get("outcomes", [])
        valid = [o for o in outs if o["verdict"] in VALID and o.get("speedup")]
        best = max((o["speedup"] for o in valid), default=None)
        best_improving = max((o["speedup"] for o in valid if o["verdict"] == "improved"), default=None)
        rows.append({
            "query_id": q["id"],
            "title": q.get("title", ""),
            "complexity": q["complexity"],
            "complexity_score": _get(r, "complexity_metrics", "score"),
            "scale": cond["scale"],
            "index_config": cond["index_config"],
            "orig_ms": orig.get("median_ms"),
            "orig_cost": orig.get("optimizer_cost"),
            "orig_rows": orig.get("rows"),
            "n_candidates": len(outs),
            "n_valid": len(valid),
            "n_improved": sum(o["verdict"] == "improved" for o in outs),
            "n_regressed": sum(o["verdict"] == "regressed" for o in outs),
            "best_valid_speedup": best,
            "pipeline_speedup": best_improving if best_improving else 1.0,
            "best_candidate_id": r.get("best_candidate_id"),
            "error": r.get("error"),
            "llm_calls": r.get("llm_calls", 0),
            "llm_seconds": r.get("llm_seconds", 0.0),
        })
    return pd.DataFrame(rows)


def geomean(values) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0]
    if not vals:
        return None
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def funnel(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    return {
        "generated": n,
        "executable": int(df["executable"].sum()) if n else 0,
        "equivalent": int(df["equivalent"].sum()) if n else 0,
        "valid": int(df["valid"].sum()) if n else 0,
        "improved": int(df["improved"].sum()) if n else 0,
        "regressed": int(df["regressed"].sum()) if n else 0,
        "executable_rate": _rate(int(df["executable"].sum()), n) if n else None,
        "equivalence_rate_of_executable": _rate(int(df["equivalent"].sum()), int(df["executable"].sum())) if n else None,
        "equivalence_rate_overall": _rate(int(df["equivalent"].sum()), n) if n else None,
        "improved_rate_of_valid": _rate(int(df["improved"].sum()), int(df["valid"].sum())) if n else None,
        "regressed_rate_of_valid": _rate(int(df["regressed"].sum()), int(df["valid"].sum())) if n else None,
        "median_speedup_valid": float(df.loc[df["valid"], "speedup"].median()) if n and df["valid"].any() else None,
        "geomean_speedup_valid": geomean(df.loc[df["valid"], "speedup"]) if n else None,
        "max_speedup": float(df.loc[df["valid"], "speedup"].max()) if n and df["valid"].any() else None,
    }


def grouped_funnel(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        f = funnel(g)
        rows.append({**dict(zip(by, key)), **f})
    out = pd.DataFrame(rows)
    if "complexity" in by:
        out["complexity"] = pd.Categorical(out["complexity"], COMPLEXITY_LEVELS, ordered=True)
        out = out.sort_values(by)
    return out.reset_index(drop=True)


def analyze(results: list[dict], meta: Optional[dict] = None) -> dict[str, Any]:
    df = outcomes_frame(results)
    runs = runs_frame(results)
    meta = meta or {}
    summary: dict[str, Any] = {"meta": meta, "n_runs": len(runs), "n_candidates": len(df)}
    if df.empty:
        summary["empty"] = True
        return {"summary": summary, "outcomes": df, "runs": runs}

    summary["overall"] = funnel(df)
    summary["by_source"] = grouped_funnel(df, ["source"]).to_dict("records")
    by_cx = grouped_funnel(df, ["complexity"])
    by_scale_idx = grouped_funnel(df, ["scale", "index_config"])

    # pipeline-level (best rewrite per query/condition)
    ok_runs = runs[runs["error"].isna()]
    pipe = {
        "runs": len(ok_runs),
        "runs_with_valid_rewrite": int((ok_runs["n_valid"] > 0).sum()),
        "runs_with_improvement": int((ok_runs["n_improved"] > 0).sum()),
        "geomean_pipeline_speedup": geomean(ok_runs["pipeline_speedup"]),
        "median_pipeline_speedup": float(ok_runs["pipeline_speedup"].median()) if len(ok_runs) else None,
        "max_pipeline_speedup": float(ok_runs["pipeline_speedup"].max()) if len(ok_runs) else None,
    }
    summary["pipeline"] = pipe

    runs_cx = ok_runs.groupby("complexity").agg(
        runs=("query_id", "size"), improved_runs=("n_improved", lambda s: int((s > 0).sum())),
        geomean_pipeline_speedup=("pipeline_speedup", geomean)).reindex(COMPLEXITY_LEVELS).dropna(how="all")
    runs_cond = ok_runs.groupby(["scale", "index_config"]).agg(
        runs=("query_id", "size"), improved_runs=("n_improved", lambda s: int((s > 0).sum())),
        geomean_pipeline_speedup=("pipeline_speedup", geomean),
        median_orig_ms=("orig_ms", "median")).reset_index()

    # correlation of complexity with rewrite success (RQ4)
    corr = {}
    if ok_runs["complexity_score"].notna().sum() > 2:
        corr["score_vs_pipeline_speedup_spearman"] = _safe_corr(ok_runs["complexity_score"],
                                                                ok_runs["pipeline_speedup"])
        per_query = df.groupby("query_id").agg(score=("complexity_score", "first"),
                                               equiv_rate=("equivalent", "mean"),
                                               exec_rate=("executable", "mean"))
        corr["score_vs_equivalence_rate_spearman"] = _safe_corr(per_query["score"], per_query["equiv_rate"])
        corr["score_vs_executable_rate_spearman"] = _safe_corr(per_query["score"], per_query["exec_rate"])
    summary["complexity_correlation"] = corr

    # failure analysis
    rejected = df[df["verdict"].isin(REJECTED)]
    summary["verdict_counts"] = df["verdict"].value_counts().to_dict()
    summary["error_categories"] = rejected["error_category"].dropna().value_counts().to_dict()
    summary["equivalence_failures"] = (df[df["verdict"] == "rejected_nonequivalent"]["equivalence_category"]
                                       .value_counts().to_dict())
    summary["vacuous_equivalences"] = int(df["vacuous"].sum())

    # repair effectiveness
    rep = df[df["source"] == "repair"]
    summary["repair"] = {"attempts": len(rep), "now_valid": int(rep["valid"].sum()),
                         "now_improved": int(rep["improved"].sum())}

    # strategy families
    strat = grouped_funnel(df, ["strategy_family"]).sort_values("generated", ascending=False)

    # plan change vs improvement (why rewrites help)
    valid = df[df["valid"]]
    if not valid.empty and valid["plan_changed"].notna().any():
        changed = valid["plan_changed"].map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)
        summary["plan_change"] = {
            "valid_with_plan_change": int(changed.sum()),
            "improved_with_plan_change": int((valid["improved"] & changed).sum()),
            "improved_without_plan_change": int((valid["improved"] & ~changed).sum()),
        }
    if valid["cost_ratio"].notna().any():
        vv = valid[valid["cost_ratio"].notna() & valid["speedup"].notna()]
        summary["cost_vs_time"] = {
            "n": len(vv),
            "spearman_cost_ratio_vs_speedup": _safe_corr(vv["cost_ratio"], vv["speedup"]),
            "cheaper_but_slower": int(((vv["cost_ratio"] < 1) & (vv["speedup"] < 1)).sum()),
            "costlier_but_faster": int(((vv["cost_ratio"] > 1) & (vv["speedup"] > 1)).sum()),
        }

    summary["rq_answers"] = research_question_answers(summary, by_cx, runs_cond)
    return {
        "summary": summary,
        "outcomes": df,
        "runs": runs,
        "by_complexity": by_cx,
        "by_condition": by_scale_idx,
        "runs_by_complexity": runs_cx.reset_index(),
        "runs_by_condition": runs_cond,
        "by_strategy": strat,
    }


def _safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    try:
        a, b = pd.Series(a, dtype=float).reset_index(drop=True), pd.Series(b, dtype=float).reset_index(drop=True)
        if a.nunique() < 2 or b.nunique() < 2:
            return None
        v = a.rank().corr(b.rank())
        return None if v is None or np.isnan(v) else round(float(v), 3)
    except Exception:
        return None


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _x(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}x"


def research_question_answers(summary: dict, by_cx: pd.DataFrame, runs_cond: pd.DataFrame) -> dict[str, str]:
    o, p = summary["overall"], summary["pipeline"]
    ans = {}
    ans["RQ1"] = (f"{o['executable']} of {o['generated']} generated candidates ({_pct(o['executable_rate'])}) "
                  f"passed the safety, syntax and executability gates.")
    ans["RQ2"] = (f"{o['equivalent']} candidates were semantically equivalent: {_pct(o['equivalence_rate_of_executable'])} "
                  f"of executable candidates and {_pct(o['equivalence_rate_overall'])} of all candidates. "
                  f"Vacuous (empty-result) equivalences: {summary.get('vacuous_equivalences', 0)}.")
    ans["RQ3"] = (f"Of {o['valid']} valid rewrites, {_pct(o['improved_rate_of_valid'])} improved performance by at "
                  f"least the threshold and {_pct(o['regressed_rate_of_valid'])} were harmful (slower). "
                  f"Geometric-mean speedup of valid rewrites: {_x(o['geomean_speedup_valid'])} "
                  f"(max {_x(o['max_speedup'])}). Selecting the best verified rewrite per query and condition "
                  f"improved {p['runs_with_improvement']} of {p['runs']} query runs, for a geometric-mean "
                  f"pipeline speedup of {_x(p['geomean_pipeline_speedup'])}.")
    if not by_cx.empty:
        parts = [f"{r['complexity']}: exec {_pct(r['executable_rate'])}, equiv {_pct(r['equivalence_rate_overall'])}, "
                 f"improved {_pct(r['improved_rate_of_valid'])} of valid"
                 for r in by_cx.to_dict("records")]
        c = summary.get("complexity_correlation", {})
        ans["RQ4"] = ("By complexity level - " + "; ".join(parts) + ". Spearman correlation of the structural "
                      f"complexity score with equivalence rate: {c.get('score_vs_equivalence_rate_spearman', 'n/a')}, "
                      f"with pipeline speedup: {c.get('score_vs_pipeline_speedup_spearman', 'n/a')}.")
    if not runs_cond.empty:
        parts = [f"sf={r['scale']:g}/{r['index_config']}: {int(r['improved_runs'])}/{int(r['runs'])} runs improved, "
                 f"geomean {_x(r['geomean_pipeline_speedup'])}" for r in runs_cond.to_dict("records")]
        ans["RQ5"] = "By database scale and index configuration - " + "; ".join(parts) + "."
    return ans
