#!/usr/bin/env python3
"""
Evaluate accuracy of what the site SHOWS USERS against reality.
=================================================================
There are two datasets:
  - data/gurugram_bootstrap.csv  -- TomTom's *historical model*
    (congestion_idx = 1 - noTraffic/historic). This is a day-of-week x hour
    average with no date sensitivity, and it is what the site serves for
    the ~95% of (corridor, day, hour) cells that have never been directly
    observed. We call this "served" below.
  - data/gurugram_observed.csv   -- reality as actually measured by our own
    collector (congestion_idx = 1 - noTraffic/live), growing every ~40 min
    via CI. We call this "observed" below.

Both use the same free_flow numerator, so congestion_idx values from the
two files are directly comparable cell-for-cell.

This script joins them on (corridor_id, day_of_week, hour) and reports:
  1. Point-error metrics (MAE / RMSE / bias / error distribution),
     overall and per corridor and per road class.
  2. Label agreement -- the metric users actually experience -- with a
     full confusion matrix and a directional (understate/overstate) split.
  3. Advice-level accuracy: does the served ranking of hours within a
     corridor/day actually match the observed ranking? (Only computed
     where hour-diversity in the observed data makes it meaningful.)
  4. Coverage: how many of the 8x7x24 (now 13x7x24) cells have any
     observed data at all, with sample sizes attached to every figure and
     an explicit refusal to present tiny-n results as trustworthy.

Nothing here is tuned to look good. If the served numbers are inaccurate,
this script says so and quantifies it.

Design note -- IMPORTANT, read before changing thresholds/corridors:
  Label thresholds and label_for() are imported directly from
  backend/app.py (the single source of truth referenced by
  docs/api_contract.md). Corridor metadata is imported directly from
  corridors.py. Neither is duplicated here. If those files change, this
  script's behavior changes with them automatically -- that's intentional.

Usage:
    python3 tools/evaluate_accuracy.py [--out docs/accuracy_report.md]

Re-runnable as data accumulates: writes a fresh
docs/accuracy_report.md each run (current snapshot) and appends one row
to docs/accuracy_history.csv (append-only run log, so accuracy over time
can be tracked/plotted later without re-deriving old numbers).
"""

from __future__ import annotations  # py3.9 compat: allows `dict | None` in annotations

import argparse
import contextlib
import csv
import datetime
import importlib.util
import io
import math
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

BOOTSTRAP_CSV = os.path.join(REPO_ROOT, "data", "gurugram_bootstrap.csv")
OBSERVED_CSV = os.path.join(REPO_ROOT, "data", "gurugram_observed.csv")
REPORT_MD = os.path.join(REPO_ROOT, "docs", "accuracy_report.md")
HISTORY_CSV = os.path.join(REPO_ROOT, "docs", "accuracy_history.csv")

LABEL_ORDER = ["Free", "Moderate", "Heavy", "Severe"]
LABEL_RANK = {l: i for i, l in enumerate(LABEL_ORDER)}

# Confidence tiers for sample size -- deliberately conservative. These are
# NOT statistical magic numbers, just an honest line in the sand so the
# report can never present a tiny-n figure as if it were solid.
N_MODERATE = 100   # >= this many comparisons: "preliminary but usable" at best
N_LOW = 30          # >= this many: "low confidence"; below: "insufficient"


def confidence_tier(n: int) -> str:
    if n >= N_MODERATE:
        return "LOW-MODERATE (n still small for a product-level claim)"
    if n >= N_LOW:
        return "LOW (preliminary only)"
    return "INSUFFICIENT (do not draw conclusions)"


# ─────────────────────────────────────────────────────────────────────────
# Import single sources of truth (do not duplicate their values here)
# ─────────────────────────────────────────────────────────────────────────

def load_backend_constants():
    """Import LABEL_THRESHOLDS / label_for from backend/app.py without
    duplicating them, and without letting its startup prints (model
    loading, grid precompute) clutter this script's own output."""
    spec = importlib.util.spec_from_file_location(
        "gtp_backend_app_readonly", os.path.join(REPO_ROOT, "backend", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod.LABEL_THRESHOLDS, mod.label_for


def load_corridors():
    import corridors as corridors_mod
    return {c["id"]: c for c in corridors_mod.CORRIDORS}


# ─────────────────────────────────────────────────────────────────────────
# Data loading / join
# ─────────────────────────────────────────────────────────────────────────

def load_data():
    boot = pd.read_csv(BOOTSTRAP_CSV)
    obs = pd.read_csv(OBSERVED_CSV)
    return boot, obs


def merge_served_observed(boot: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """One row per OBSERVED measurement, joined to the served (bootstrap)
    value for that exact (corridor_id, day_of_week, hour) cell. Every
    observed row is an independent real-world measurement -- we do NOT
    average duplicate sweeps of the same cell away here, because the
    point-level metrics below are meant to answer "how well does the
    single served number predict any given real instance of this hour?",
    and averaging would hide real variance instead of measuring it."""
    key = ["corridor_id", "day_of_week", "hour"]
    boot_slim = boot[key + ["congestion_idx"]].rename(columns={"congestion_idx": "idx_served"})
    obs_slim = obs[key + ["road_class", "congestion_idx", "collected_at"]].rename(
        columns={"congestion_idx": "idx_observed"}
    )
    merged = obs_slim.merge(boot_slim, on=key, how="left")
    unmatched = merged["idx_served"].isna().sum()
    merged = merged.dropna(subset=["idx_served"])
    merged["error"] = merged["idx_served"] - merged["idx_observed"]  # + = we understate reality is worse... see below
    merged["abs_error"] = merged["error"].abs()
    return merged, unmatched


# ─────────────────────────────────────────────────────────────────────────
# 1. Point-error metrics
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, stat_fn, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for a statistic (mean of abs error, mean of
    signed error, etc). Cheap, dependency-light (no scipy requirement),
    and honest about small-n: intervals will simply come out wide."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def point_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    err = df["error"].to_numpy()
    abs_err = df["abs_error"].to_numpy()
    mae = float(np.mean(abs_err)) if n else float("nan")
    rmse = float(np.sqrt(np.mean(err ** 2))) if n else float("nan")
    bias = float(np.mean(err)) if n else float("nan")
    mae_ci = bootstrap_ci(abs_err, np.mean) if n >= 2 else (float("nan"),) * 2
    bias_ci = bootstrap_ci(err, np.mean) if n >= 2 else (float("nan"),) * 2
    pct = {}
    if n:
        pct["p50_abs"] = float(np.percentile(abs_err, 50))
        pct["p90_abs"] = float(np.percentile(abs_err, 90))
        pct["max_abs"] = float(np.max(abs_err))
        pct["p10_signed"] = float(np.percentile(err, 10))
        pct["p50_signed"] = float(np.percentile(err, 50))
        pct["p90_signed"] = float(np.percentile(err, 90))
    return {
        "n": n, "mae": mae, "rmse": rmse, "bias": bias,
        "mae_ci95": mae_ci, "bias_ci95": bias_ci,
        "tier": confidence_tier(n), **pct,
    }


def per_group_metrics(df: pd.DataFrame, group_col: str, names: dict | None = None) -> list:
    out = []
    for key, g in df.groupby(group_col):
        m = point_metrics(g)
        m[group_col] = key
        if names:
            m["name"] = names.get(key, str(key))
        out.append(m)
    out.sort(key=lambda m: -m["n"])
    return out


# ─────────────────────────────────────────────────────────────────────────
# 2. Label agreement / confusion matrix / directional split
# ─────────────────────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


def label_metrics(df: pd.DataFrame, label_for) -> dict:
    served_labels = df["idx_served"].apply(label_for)
    observed_labels = df["idx_observed"].apply(label_for)
    n = len(df)

    matrix = {s: {o: 0 for o in LABEL_ORDER} for s in LABEL_ORDER}
    for s, o in zip(served_labels, observed_labels):
        matrix[s][o] += 1

    agree = int(sum(matrix[l][l] for l in LABEL_ORDER))
    agree_ci = wilson_ci(agree, n) if n else (float("nan"), float("nan"))

    served_rank = served_labels.map(LABEL_RANK)
    observed_rank = observed_labels.map(LABEL_RANK)
    understate = int((served_rank < observed_rank).sum())  # told user it's BETTER than reality -- dangerous
    overstate = int((served_rank > observed_rank).sum())   # told user it's WORSE than reality -- merely annoying
    exact = n - understate - overstate

    return {
        "n": n, "matrix": matrix, "agree": agree, "agree_pct": agree / n if n else float("nan"),
        "agree_ci95": agree_ci, "tier": confidence_tier(n),
        "understate": understate, "understate_pct": understate / n if n else float("nan"),
        "overstate": overstate, "overstate_pct": overstate / n if n else float("nan"),
        "exact": exact, "exact_pct": exact / n if n else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. Advice-level accuracy: pairwise hour-ranking concordance
# ─────────────────────────────────────────────────────────────────────────

def advice_metrics(boot: pd.DataFrame, obs: pd.DataFrame, corridor_names: dict) -> dict:
    """For each (corridor, day) where we have >=2 DISTINCT observed hours,
    check whether the served (bootstrap) ranking of those hours agrees
    with the observed ranking -- i.e. if the site says hour A is clearer
    than hour B, was it actually clearer? This is what "leave at X, avoid
    Y" advice depends on, at a granularity our current coverage can
    actually test (full best/worst-window text checks would need many
    more distinct hours per corridor/day than we have).

    Duplicate sweeps of the same (corridor,day,hour) cell are averaged to
    one observed value per hour first -- this stage is about ranking
    hours against each other, not about measurement noise within an hour.
    """
    key = ["corridor_id", "day_of_week", "hour"]
    obs_hourly = obs.groupby(key)["congestion_idx"].mean().reset_index()
    boot_slim = boot[key + ["congestion_idx"]].rename(columns={"congestion_idx": "idx_served"})
    joined = obs_hourly.merge(boot_slim, on=key, how="left").rename(columns={"congestion_idx": "idx_observed"})

    groups = []
    total_pairs = 0
    concordant_pairs = 0
    tied_pairs = 0
    best_hits = 0
    worst_hits = 0
    n_groups = 0

    for (cid, day), g in joined.groupby(["corridor_id", "day_of_week"]):
        hours = g["hour"].tolist()
        if len(hours) < 2:
            continue
        n_groups += 1
        served = dict(zip(g["hour"], g["idx_served"]))
        observed = dict(zip(g["hour"], g["idx_observed"]))

        g_concordant = 0
        g_total = 0
        for i in range(len(hours)):
            for j in range(i + 1, len(hours)):
                h1, h2 = hours[i], hours[j]
                s_diff = served[h1] - served[h2]
                o_diff = observed[h1] - observed[h2]
                if s_diff == 0 or o_diff == 0:
                    tied_pairs += 1
                    continue
                total_pairs += 1
                g_total += 1
                if (s_diff > 0) == (o_diff > 0):
                    concordant_pairs += 1
                    g_concordant += 1

        best_served = min(hours, key=lambda h: served[h])
        best_observed = min(hours, key=lambda h: observed[h])
        worst_served = max(hours, key=lambda h: served[h])
        worst_observed = max(hours, key=lambda h: observed[h])
        best_hit = best_served == best_observed
        worst_hit = worst_served == worst_observed
        best_hits += int(best_hit)
        worst_hits += int(worst_hit)

        groups.append({
            "corridor_id": cid, "corridor_name": corridor_names.get(cid, str(cid)),
            "day": day, "n_hours": len(hours), "hours": sorted(hours),
            "pairwise_concordance": (g_concordant / g_total) if g_total else float("nan"),
            "n_pairs": g_total,
            "best_hit": best_hit, "worst_hit": worst_hit,
        })

    return {
        "n_groups": n_groups,
        "total_pairs": total_pairs,
        "tied_pairs": tied_pairs,
        "concordant_pairs": concordant_pairs,
        "pairwise_concordance": (concordant_pairs / total_pairs) if total_pairs else float("nan"),
        "best_hit_rate": (best_hits / n_groups) if n_groups else float("nan"),
        "worst_hit_rate": (worst_hits / n_groups) if n_groups else float("nan"),
        "groups": groups,
        "tier": confidence_tier(n_groups),
    }


# ─────────────────────────────────────────────────────────────────────────
# 4. Coverage
# ─────────────────────────────────────────────────────────────────────────

def coverage_metrics(corridors: dict, obs: pd.DataFrame) -> dict:
    total_cells = len(corridors) * 7 * 24
    covered_cells = obs[["corridor_id", "day_of_week", "hour"]].drop_duplicates().shape[0]
    days_covered = sorted(obs["day_of_week"].unique().tolist())
    return {
        "n_corridors": len(corridors),
        "total_cells": total_cells,
        "covered_cells": covered_cells,
        "coverage_pct": covered_cells / total_cells if total_cells else 0,
        "n_observed_rows": len(obs),
        "days_covered": days_covered,
        "days_missing": [d for d in range(7) if d not in days_covered],
        "first_collected": obs["collected_at"].min() if len(obs) else None,
        "last_collected": obs["collected_at"].max() if len(obs) else None,
    }


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def fmt_pct(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100*x:.{d}f}%"


def render_report(coverage, overall, per_corridor, per_class, labels, advice,
                   label_thresholds, corridors, unmatched, run_date) -> str:
    lines = []
    a = lines.append

    a(f"# Accuracy Report — served numbers vs. measured reality")
    a("")
    a(f"_Generated {run_date} by `tools/evaluate_accuracy.py`. Re-run this script as "
      f"`data/gurugram_observed.csv` grows; it overwrites this file with the latest snapshot "
      f"and appends one row to `docs/accuracy_history.csv` so trend-over-time can be tracked._")
    a("")

    # ---- Headline callout, honesty gate ----
    a("## Headline")
    a("")
    tier = overall["tier"]
    if coverage["covered_cells"] < 200:
        a(f"**Coverage is still very thin: {coverage['covered_cells']} of {coverage['total_cells']} "
          f"cells ({fmt_pct(coverage['coverage_pct'])}) have ever been observed, from "
          f"{coverage['n_observed_rows']} collected rows spanning "
          f"{coverage['first_collected']} to {coverage['last_collected']}.** "
          f"The numbers below are real (not fabricated, not tuned), but treat every headline "
          f"figure here as **{tier}** until coverage grows. See the Coverage section for exactly "
          f"how much more data would change that.")
    a("")
    a(f"- Point error (served vs. observed `congestion_index`, n={overall['n']}): "
      f"**MAE = {fmt(overall['mae'])}**, **RMSE = {fmt(overall['rmse'])}**, "
      f"**bias = {fmt(overall['bias'])}** "
      f"(95% CI {fmt(overall['bias_ci95'][0])} to {fmt(overall['bias_ci95'][1])}).")
    sign_word = "UNDERSTATES" if overall["bias"] < 0 else "OVERSTATES"
    a(f"  A negative bias means the site systematically **{sign_word}** real congestion "
      f"(served value below what was actually measured).")
    a(f"- Label agreement (what users actually see, n={labels['n']}): "
      f"**{fmt_pct(labels['agree_pct'])}** exact match "
      f"(95% CI {fmt_pct(labels['agree_ci95'][0])}–{fmt_pct(labels['agree_ci95'][1])}). "
      f"**{fmt_pct(labels['understate_pct'])}** of the time the site showed a label "
      f"*better* than reality (the dangerous direction), "
      f"**{fmt_pct(labels['overstate_pct'])}** of the time *worse* than reality (merely annoying).")
    if advice["n_groups"]:
        a(f"- Advice-level hour ranking (n={advice['n_groups']} corridor/day groups, "
          f"{advice['total_pairs']} comparable hour-pairs): pairwise concordance "
          f"**{fmt_pct(advice['pairwise_concordance'])}**, best-hour-hit rate "
          f"**{fmt_pct(advice['best_hit_rate'])}**, worst-hour-hit rate "
          f"**{fmt_pct(advice['worst_hit_rate'])}**.")
    else:
        a("- Advice-level hour ranking: **not computed** — no corridor/day currently has 2+ "
          "distinct observed hours (see Advice-level section).")
    a("")

    # ---- What "served" means here ----
    a("## What this compares")
    a("")
    a("- **Served** = `data/gurugram_bootstrap.csv`'s `congestion_idx` for a "
      "(corridor, day-of-week, hour) cell — TomTom's historical-model average "
      "(`1 - noTraffic/historic`), with no date sensitivity. This is what "
      "`backend/app.py` serves for any cell that has not (yet) been directly observed — "
      f"today that is {fmt_pct(1 - coverage['coverage_pct'])} of all cells.")
    a("- **Observed** = `data/gurugram_observed.csv`'s `congestion_idx` for the same cell — "
      "a real measurement collected by CI (`1 - noTraffic/live`) at some actual date/time "
      "that fell into that (corridor, day-of-week, hour) bucket.")
    a("- Both use the same `free_flow` numerator, so the two `congestion_idx` values are "
      "directly comparable — this is *not* comparing two different quantities.")
    a("- **One caveat, for completeness:** the live backend's `load_measured_grid()` "
      "(`backend/app.py`) actually overwrites a cell with the *freshest matching "
      "observation* once one exists for that exact cell, so a handful of cells are, right "
      "now, serving the observed value verbatim (self-matching by construction). This "
      "evaluation deliberately measures the underlying **bootstrap** model instead, "
      "because that is what is served for the overwhelming majority of cells (everything "
      "not yet observed), and it is what was being served for every one of these "
      "comparison rows at the moment they were actually collected.")
    if unmatched:
        a(f"- {unmatched} observed row(s) had no matching bootstrap cell and were excluded.")
    a("")

    # ---- Coverage ----
    a("## Coverage")
    a("")
    a(f"- Corridors: {coverage['n_corridors']} (13, after the 5 added 2026-08-17)")
    a(f"- Total cells (corridors x 7 days x 24 hours): {coverage['total_cells']}")
    a(f"- Cells with at least one observation: {coverage['covered_cells']} "
      f"({fmt_pct(coverage['coverage_pct'])})")
    a(f"- Observed rows collected so far: {coverage['n_observed_rows']}")
    days_str = ", ".join(DAY_NAMES[d] for d in coverage["days_covered"]) or "none"
    missing_str = ", ".join(DAY_NAMES[d] for d in coverage["days_missing"]) or "none"
    a(f"- Days of week with any coverage: {days_str}")
    a(f"- Days of week with ZERO coverage: {missing_str}")
    a(f"- Collection window: {coverage['first_collected']} → {coverage['last_collected']}")
    a("")
    a("At the current CI cadence (~40 min/sweep, 13 corridors x 1 hour-of-day per sweep), "
      "reaching even 500 covered cells (~23% coverage, still thin) needs roughly "
      f"{max(0, 500 - coverage['covered_cells'])} more distinct (corridor, day, hour) cells "
      "to be hit — coverage grows slower than row count because the same popular "
      "hours get re-sampled before new ones are reached. Full 2184-cell coverage "
      "(every hour of every day) requires the collector to run across all 7 days, which "
      "it has not yet done (see missing days above).")
    a("")

    # ---- Point-error metrics ----
    a("## 1. Point-error metrics (served vs. observed `congestion_index`)")
    a("")
    a(f"Overall, n={overall['n']} — confidence tier: **{overall['tier']}**")
    a("")
    a("| metric | value | 95% CI |")
    a("|---|---|---|")
    a(f"| MAE | {fmt(overall['mae'])} | {fmt(overall['mae_ci95'][0])} – {fmt(overall['mae_ci95'][1])} |")
    a(f"| RMSE | {fmt(overall['rmse'])} | — |")
    a(f"| Bias (mean signed error, served − observed) | {fmt(overall['bias'])} | "
      f"{fmt(overall['bias_ci95'][0])} – {fmt(overall['bias_ci95'][1])} |")
    a(f"| p50 abs error | {fmt(overall.get('p50_abs'))} | — |")
    a(f"| p90 abs error | {fmt(overall.get('p90_abs'))} | — |")
    a(f"| max abs error | {fmt(overall.get('max_abs'))} | — |")
    a(f"| p10 / p50 / p90 signed error | {fmt(overall.get('p10_signed'))} / "
      f"{fmt(overall.get('p50_signed'))} / {fmt(overall.get('p90_signed'))} | — |")
    a("")
    a("### Per corridor")
    a("")
    a("| corridor | n | MAE | RMSE | bias | tier |")
    a("|---|---|---|---|---|---|")
    for m in per_corridor:
        a(f"| {m['name']} | {m['n']} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['bias'])} | {m['tier']} |")
    a("")
    a("### Per road class")
    a("")
    a("| road class | n | MAE | RMSE | bias | tier |")
    a("|---|---|---|---|---|---|")
    for m in per_class:
        a(f"| {m['road_class']} | {m['n']} | {fmt(m['mae'])} | {fmt(m['rmse'])} | {fmt(m['bias'])} | {m['tier']} |")
    a("")

    # ---- Label agreement ----
    a("## 2. Label agreement (what the user actually sees)")
    a("")
    thr_str = ", ".join(f"{name} < {t}" for t, name in label_thresholds) + ", Severe ≥ " + str(label_thresholds[-1][0])
    a(f"Thresholds (from `backend/app.py`, matching `docs/api_contract.md`): {thr_str}")
    a("")
    a(f"n={labels['n']} — confidence tier: **{labels['tier']}**")
    a("")
    a(f"- Exact label match: **{fmt_pct(labels['agree_pct'])}** "
      f"(95% CI {fmt_pct(labels['agree_ci95'][0])}–{fmt_pct(labels['agree_ci95'][1])})")
    a(f"- Understated (served label better than observed — **dangerous**, user leaves "
      f"at a time we called clear/moderate but was actually worse): "
      f"{labels['understate']} / {labels['n']} = **{fmt_pct(labels['understate_pct'])}**")
    a(f"- Overstated (served label worse than observed — merely annoying): "
      f"{labels['overstate']} / {labels['n']} = **{fmt_pct(labels['overstate_pct'])}**")
    a("")
    a("### Confusion matrix (rows = served label, columns = observed label)")
    a("")
    header = "| served \\ observed | " + " | ".join(LABEL_ORDER) + " | row total |"
    a(header)
    a("|" + "---|" * (len(LABEL_ORDER) + 2))
    for s in LABEL_ORDER:
        row = labels["matrix"][s]
        total = sum(row.values())
        a(f"| **{s}** | " + " | ".join(str(row[o]) for o in LABEL_ORDER) + f" | {total} |")
    col_totals = [sum(labels["matrix"][s][o] for s in LABEL_ORDER) for o in LABEL_ORDER]
    a("| **col total** | " + " | ".join(str(t) for t in col_totals) + f" | {labels['n']} |")
    a("")

    # ---- Advice-level ----
    a("## 3. Advice-level accuracy (hour ranking within a corridor/day)")
    a("")
    a("The site's core claim is \"leave at hour X, avoid hour Y.\" That claim only makes "
      "sense to check where we have observed data at 2+ distinct hours for the same "
      "corridor and day-of-week, so we can ask: did the served ranking of those hours "
      "match the observed ranking?")
    a("")
    if advice["n_groups"] == 0:
        a("**Not computed: zero corridor/day groups currently have 2+ distinct observed "
          "hours.** Coverage is too thin for this check to mean anything yet.")
    else:
        a(f"n={advice['n_groups']} corridor/day groups, {advice['total_pairs']} comparable "
          f"hour-pairs ({advice['tied_pairs']} tied pairs excluded) — confidence tier: "
          f"**{advice['tier']}**")
        a("")
        a(f"- Pairwise concordance (served says A vs B in the same order reality did): "
          f"**{fmt_pct(advice['pairwise_concordance'])}**")
        a(f"- Best-hour-hit rate (served's best hour among observed hours = observed's "
          f"actual best hour): **{fmt_pct(advice['best_hit_rate'])}** "
          f"({sum(1 for g in advice['groups'] if g['best_hit'])}/{advice['n_groups']})")
        a(f"- Worst-hour-hit rate: **{fmt_pct(advice['worst_hit_rate'])}** "
          f"({sum(1 for g in advice['groups'] if g['worst_hit'])}/{advice['n_groups']})")
        a("")
        a("**Important limitation:** every group below comes from Monday "
          "(`day_of_week=0`) — that is the only day with enough distinct observed hours "
          "to rank. This says nothing yet about weekday-vs-weekend or other days.")
        a("")
        a("| corridor | day | hours observed | n pairs | pairwise concordance | best-hour hit | worst-hour hit |")
        a("|---|---|---|---|---|---|---|")
        for g in sorted(advice["groups"], key=lambda x: x["corridor_id"]):
            a(f"| {g['corridor_name']} | {DAY_NAMES[g['day']]} | "
              f"{', '.join(str(h) for h in g['hours'])} | {g['n_pairs']} | "
              f"{fmt_pct(g['pairwise_concordance'])} | "
              f"{'yes' if g['best_hit'] else 'no'} | {'yes' if g['worst_hit'] else 'no'} |")
    a("")

    return "\n".join(lines)


METHODOLOGY = """
## Evaluation framework — formulas and why

This section is meant to be defensible to a teacher, not just a wall of numbers.

**Why label agreement matters more than MAE for this product.**
Nobody using this site reads `congestion_index = 0.237`. They read "Heavy" and decide
whether to leave now. MAE and RMSE describe how far off the underlying number is, but
they don't directly answer the question a user cares about: *did the label I saw match
what actually happened?* Two cells can have identical MAE (e.g. both off by 0.05) and
land in completely different places for the user — one might sit at a threshold
boundary and flip label, the other might sit safely mid-band and not flip at all. Label
agreement (`served_label == observed_label`) is therefore the primary product-facing
metric; MAE/RMSE are the secondary, model-facing ones underneath it.

**Why bias direction matters more than magnitude.**
`bias = mean(served − observed)`. A model with high MAE but zero bias is *noisy but
fair* — errors go both ways and average out; a user who follows its advice over many
trips comes out roughly even. A model with the same MAE but strong *negative* bias is
*consistently lying in the dangerous direction* — it tells users conditions are better
than they are, so they leave expecting a clear road and hit one that isn't. That is a
worse failure mode than noise, even at equal MAE, because it's not self-correcting: a
noisy-but-unbiased signal degrades gracefully with repeated use (regression to the
mean), a biased one doesn't. That's why the report calls out the *sign* of bias in the
headline, not just its magnitude, and why the label-level directional split
(understate % vs overstate %) is reported separately from the raw label-match %.

**Why a confusion matrix, not just an accuracy percentage.**
A single "62% match" figure hides *which* mistakes are being made. A model that
confuses Free/Moderate constantly but never confuses Free/Severe is far more usable
than one with the same overall accuracy that occasionally calls a Severe cell Free.
The full matrix makes that visible; the collapsed directional summary (understate vs
overstate) is derived from it, not a replacement for it.

**Why advice-level (pairwise ranking) accuracy, and why it's separate from label
accuracy.** The site's actual promise is comparative — "leave at X, not Y" — not
absolute. A model could have the labels systematically shifted by one band (e.g. always
one notch more congested than reality) and still give perfect *advice*, because what
matters for "when should I go" is whether hour X is really better than hour Y, not
whether either hour's absolute label is exactly right. Pairwise concordance
(`sign(served[h1]-served[h2]) == sign(observed[h1]-observed[h2])`, summed over all
comparable hour-pairs within a corridor/day) tests exactly that, independent of any
absolute-value bias. Best/worst-hour-hit rate is the same idea at its strictest: does
the specific hour the site would recommend as "best" among the hours we've actually
observed really turn out to be the best one?

**Why every figure carries an n and a confidence tier, and why headline claims are
refused below a threshold.** With ~115 observed rows against 2184 possible cells, any
single figure computed here is a small-sample estimate of a much larger population.
Reporting a bare percentage without n invites over-trusting it — exactly the kind of
overconfident presentation this project has a documented history of (see
`data-integrity-history` in project memory: a fabricated R²=0.83, a hand-typed lookup
table served as if it were model output). This script instead (a) reports n next to
every metric, (b) computes a bootstrap 95% CI for MAE/bias and a Wilson 95% CI for
label-agreement proportion so the *uncertainty* is visible, not just the point
estimate, and (c) explicitly tags every group below n=100 as **LOW** and below n=30 as
**INSUFFICIENT**, refusing to call insufficient-n numbers a "headline" result.

**Limitations, stated plainly.**
- Coverage is currently ~5% of cells and concentrated on one day of the week
  (Monday) and a handful of hours (0, 1, 8, 9, 10, 11) plus one Sunday evening hour
  (20). Sample sizes for other days/hours/corridors are effectively zero; this report
  says nothing about them yet.
- Per-corridor breakdowns have n≈7–10 each — individually far below even the "LOW"
  tier. They are reported for transparency (per the task's honesty requirement), not as
  standalone conclusions.
- "Observed" itself is a single live snapshot per collection sweep, not a
  time-averaged ground truth — it has its own measurement noise (traffic incidents,
  weather, one-off events). Comparing a historical-average "served" value against a
  single noisy "observed" point will always show some spread that isn't purely served-
  model error; the bootstrap CIs partly account for this by widening with n, but they
  cannot separate served-model bias from observed-measurement noise with the data
  available today.
- All observed rows to date come from a ~15-hour collection window across two
  calendar days (2026-08-16 to 2026-08-17), not from repeated sampling of the same
  cells across many weeks — so this evaluation cannot yet say anything about
  week-to-week stability of the historical-average model, only about this specific
  window.

**Proposal: run this in CI.** This script is safe to run unattended (read-only against
the two CSVs, no network calls, no side effects on `backend/*`, `data/*`, or the
model). A natural next step is a weekly scheduled GitHub Actions job that runs it and
commits the refreshed `docs/accuracy_report.md` / `docs/accuracy_history.csv`, so
accuracy is tracked automatically as `data/gurugram_observed.csv` grows — proposed
here, not wired in, since `.github/*` is owned by another workstream.
"""


def render_history_row(coverage, overall, labels, advice, run_date) -> dict:
    return {
        "date": run_date,
        "n_observed_rows": coverage["n_observed_rows"],
        "covered_cells": coverage["covered_cells"],
        "coverage_pct": round(coverage["coverage_pct"], 4),
        "point_n": overall["n"],
        "mae": round(overall["mae"], 4) if overall["n"] else "",
        "rmse": round(overall["rmse"], 4) if overall["n"] else "",
        "bias": round(overall["bias"], 4) if overall["n"] else "",
        "label_n": labels["n"],
        "label_agreement_pct": round(labels["agree_pct"], 4) if labels["n"] else "",
        "understate_pct": round(labels["understate_pct"], 4) if labels["n"] else "",
        "overstate_pct": round(labels["overstate_pct"], 4) if labels["n"] else "",
        "advice_n_groups": advice["n_groups"],
        "advice_pairwise_concordance": round(advice["pairwise_concordance"], 4) if advice["total_pairs"] else "",
    }


def append_history(row: dict):
    fieldnames = list(row.keys())
    file_exists = os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=REPORT_MD, help="Output path for the markdown report")
    parser.add_argument("--no-history", action="store_true", help="Skip appending to docs/accuracy_history.csv")
    args = parser.parse_args()

    label_thresholds, label_for = load_backend_constants()
    corridors = load_corridors()
    corridor_names = {cid: c["name"] for cid, c in corridors.items()}
    corridor_class = {cid: c["road_class"] for cid, c in corridors.items()}

    boot, obs = load_data()
    merged, unmatched = merge_served_observed(boot, obs)

    run_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    overall = point_metrics(merged)
    per_corridor = per_group_metrics(merged, "corridor_id", corridor_names)
    merged["road_class_grp"] = merged["corridor_id"].map(corridor_class)
    per_class = per_group_metrics(merged, "road_class_grp")
    for m in per_class:
        m["road_class"] = m.pop("road_class_grp")

    labels = label_metrics(merged, label_for)
    advice = advice_metrics(boot, obs, corridor_names)
    coverage = coverage_metrics(corridors, obs)

    report = render_report(coverage, overall, per_corridor, per_class, labels, advice,
                            label_thresholds, corridors, unmatched, run_date)
    report += "\n" + METHODOLOGY.strip() + "\n"

    with open(args.out, "w") as f:
        f.write(report)
    print(f"Wrote {args.out}")

    if not args.no_history:
        row = render_history_row(coverage, overall, labels, advice, run_date)
        append_history(row)
        print(f"Appended run to {HISTORY_CSV}")

    # Console summary for the human running this
    print()
    print(f"n_observed_rows={coverage['n_observed_rows']}  covered_cells={coverage['covered_cells']}"
          f"/{coverage['total_cells']} ({100*coverage['coverage_pct']:.1f}%)")
    print(f"Point (n={overall['n']}, tier={overall['tier']}): "
          f"MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} bias={overall['bias']:.4f}")
    print(f"Label agreement (n={labels['n']}, tier={labels['tier']}): "
          f"{100*labels['agree_pct']:.1f}% match, "
          f"{100*labels['understate_pct']:.1f}% understated, "
          f"{100*labels['overstate_pct']:.1f}% overstated")
    if advice["n_groups"]:
        print(f"Advice ranking (n_groups={advice['n_groups']}, tier={advice['tier']}): "
              f"{100*advice['pairwise_concordance']:.1f}% pairwise concordance")
    else:
        print("Advice ranking: not computed (no corridor/day with 2+ distinct observed hours)")


if __name__ == "__main__":
    main()
