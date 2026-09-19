#!/usr/bin/env python3
"""
Standalone evaluation of the shipped residual forecast model.

Rebuilds the same training table as model/forecast_model.py, applies the
same time-based holdout (last 20% of distinct days), and scores the
*shipped* models/forecast_residual_gbt.joblib artifact on the holdout
test rows — no Flask, no website.

Usage:
    .venv_forecast/bin/python tools/evaluate_forecast.py
    .venv_forecast/bin/python tools/evaluate_forecast.py --no-write
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model.forecast_model import (  # noqa: E402
    DWARKA_CORRIDOR_ID,
    FEATURE_COLS,
    TEST_HOLDOUT_FRACTION,
    build_training_table,
)

MODEL_FILE = os.path.join(REPO_ROOT, "models", "forecast_residual_gbt.joblib")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "forecast_eval.md")

# Same thresholds as backend/app.py LABEL_THRESHOLDS / docs/api_contract.md
LABEL_THRESHOLDS = (
    (0.091, "Free"),
    (0.200, "Moderate"),
    (0.310, "Heavy"),
)


def label_for(idx: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if idx < threshold:
            return label
    return "Severe"


def load_shipped_payload():
    if not os.path.exists(MODEL_FILE):
        print(f"[FATAL] {MODEL_FILE} not found — run model/forecast_model.py train first.")
        sys.exit(1)

    try:
        payload = joblib.load(MODEL_FILE)
    except Exception as exc:
        print(f"[FATAL] could not load {MODEL_FILE}: {exc}")
        sys.exit(1)

    if not isinstance(payload, dict) or "model" not in payload or "features" not in payload:
        print(f"[FATAL] {MODEL_FILE} is malformed (expected dict with model + features).")
        sys.exit(1)

    skill = payload.get("skill_score")
    if skill is None:
        print("[FATAL] shipped forecast artifact has no skill_score — refusing to evaluate.")
        sys.exit(1)
    try:
        skill = float(skill)
    except (TypeError, ValueError):
        print("[FATAL] shipped forecast skill_score is not numeric.")
        sys.exit(1)
    if skill <= 0:
        print(f"[FATAL] shipped forecast holdout skill={skill:.4f} <= 0 — model must not be served.")
        sys.exit(1)

    return payload


def time_holdout_split(df: pd.DataFrame):
    distinct_days = sorted(df["date"].unique())
    n_test_days = max(1, round(len(distinct_days) * TEST_HOLDOUT_FRACTION))
    test_days = set(distinct_days[-n_test_days:])
    train_df = df[~df["date"].isin(test_days)].copy()
    test_df = df[df["date"].isin(test_days)].copy()
    meta = {
        "distinct_days": len(distinct_days),
        "n_test_days": n_test_days,
        "train_day_span": (str(distinct_days[0]), str(distinct_days[-n_test_days - 1])),
        "test_day_span": (str(distinct_days[-n_test_days]), str(distinct_days[-1])),
    }
    return train_df, test_df, meta


def predict_congestion(model, dwarka_override, df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS]
    pred_resid = model.predict(X).astype(float)
    if dwarka_override is not None:
        mask = df["corridor_id"].values == DWARKA_CORRIDOR_ID
        if mask.any():
            pred_resid = pred_resid.copy()
            pred_resid[mask] = dwarka_override.predict(X.loc[mask]).astype(float)
    return np.clip(df["baseline_idx"].values + pred_resid, 0.0, 1.0)


def label_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_labels = [label_for(float(v)) for v in y_true]
    pred_labels = [label_for(float(v)) for v in y_pred]
    n = len(true_labels)
    agree = sum(t == p for t, p in zip(true_labels, pred_labels))
    return {
        "n": n,
        "agree": agree,
        "agree_pct": (agree / n) if n else float("nan"),
    }


def hour_ranking_concordance(test_df: pd.DataFrame, pred_col: str) -> dict:
    """Pairwise hour-ranking concordance on test rows, grouped by corridor+date."""
    total_pairs = 0
    concordant_pairs = 0
    tied_pairs = 0
    n_groups = 0

    for (_, _date), g in test_df.groupby(["corridor_id", "date"]):
        hours = sorted(g["hour"].unique())
        if len(hours) < 2:
            continue
        n_groups += 1
        pred = dict(zip(g["hour"], g[pred_col]))
        obs = dict(zip(g["hour"], g["congestion_idx"]))
        for i in range(len(hours)):
            for j in range(i + 1, len(hours)):
                h1, h2 = hours[i], hours[j]
                s_diff = pred[h1] - pred[h2]
                o_diff = obs[h1] - obs[h2]
                if s_diff == 0 or o_diff == 0:
                    tied_pairs += 1
                    continue
                total_pairs += 1
                if (s_diff > 0) == (o_diff > 0):
                    concordant_pairs += 1

    return {
        "n_groups": n_groups,
        "total_pairs": total_pairs,
        "tied_pairs": tied_pairs,
        "concordant_pairs": concordant_pairs,
        "pairwise_concordance": (concordant_pairs / total_pairs) if total_pairs else float("nan"),
    }


def format_report(results: dict) -> str:
    lines = [
        "# Forecast model evaluation",
        "",
        f"_Generated {results['generated_at']} by `tools/evaluate_forecast.py`._",
        "",
        "## Holdout split (time-based, last 20% of distinct days)",
        "",
        f"- Train rows: **{results['n_train']}** ({results['train_day_span'][0]} .. {results['train_day_span'][1]})",
        f"- Test rows: **{results['n_test']}** ({results['test_day_span'][0]} .. {results['test_day_span'][1]})",
        f"- Distinct calendar days in table: **{results['distinct_days']}**",
        "",
        "## Shipped artifact",
        "",
        f"- Path: `{results['model_path']}`",
        f"- Model type: **{results['model_type']}**",
        f"- Dwarka corridor override (id {DWARKA_CORRIDOR_ID}): **{results['dwarka_override']}**",
        f"- Trained at (artifact): {results['trained_at']}",
        f"- Artifact skill_score: {results['artifact_skill']:.4f}",
        "",
        "## Test-set metrics (recomputed this run)",
        "",
        f"- Baseline MAE (bootstrap vs observed): **{results['baseline_mae']:.4f}**",
        f"- Model MAE (baseline + residual vs observed): **{results['model_mae']:.4f}**",
        f"- Skill = 1 - model_MAE/baseline_MAE: **{results['skill']:.4f}**",
        f"- Label agreement (exact): **{100 * results['label_agreement_pct']:.1f}%** (n={results['n_test']})",
    ]
    if results["ranking_total_pairs"]:
        lines.append(
            f"- Hour-ranking pairwise concordance (corridor+date groups, n={results['ranking_n_groups']}): "
            f"**{100 * results['ranking_concordance']:.1f}%** "
            f"({results['ranking_concordant']}/{results['ranking_total_pairs']} pairs)"
        )
    else:
        lines.append("- Hour-ranking concordance: not computed (no corridor+date group with 2+ hours on test)")

    if results["feature_importances"]:
        lines.extend(["", "## Feature importances (shipped model)", ""])
        for name, imp in results["feature_importances"]:
            lines.append(f"- `{name}`: {imp:.4f}")

    lines.append("")
    return "\n".join(lines)


def evaluate(write_md: bool = True) -> dict:
    payload = load_shipped_payload()
    model = payload["model"]
    dwarka_override = payload.get("dwarka_override")
    model_type = payload.get("model_type") or type(model).__name__

    df = build_training_table()
    if df is None or df.empty:
        print("[FATAL] build_training_table() returned no rows.")
        sys.exit(1)

    train_df, test_df, split_meta = time_holdout_split(df)
    if len(test_df) < 1:
        print("[FATAL] holdout split produced an empty test set.")
        sys.exit(1)

    y_obs = test_df["congestion_idx"].values
    baseline_pred = test_df["baseline_idx"].values
    model_pred = predict_congestion(model, dwarka_override, test_df)

    baseline_mae = mean_absolute_error(y_obs, baseline_pred)
    model_mae = mean_absolute_error(y_obs, model_pred)
    skill = 1 - (model_mae / baseline_mae) if baseline_mae > 0 else float("nan")

    labels = label_agreement(y_obs, model_pred)
    ranking = hour_ranking_concordance(
        test_df.assign(_model_pred=model_pred),
        pred_col="_model_pred",
    )

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
        )

    results = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_path": os.path.relpath(MODEL_FILE, REPO_ROOT),
        "model_type": model_type,
        "dwarka_override": "yes" if dwarka_override is not None else "no",
        "trained_at": payload.get("trained_at", "unknown"),
        "artifact_skill": float(payload["skill_score"]),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "distinct_days": split_meta["distinct_days"],
        "train_day_span": split_meta["train_day_span"],
        "test_day_span": split_meta["test_day_span"],
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "skill": skill,
        "label_agreement_pct": labels["agree_pct"],
        "ranking_n_groups": ranking["n_groups"],
        "ranking_total_pairs": ranking["total_pairs"],
        "ranking_concordant": ranking["concordant_pairs"],
        "ranking_concordance": ranking["pairwise_concordance"],
        "feature_importances": importances,
    }

    print("=" * 70)
    print("FORECAST MODEL EVALUATION (shipped artifact, time holdout)")
    print("=" * 70)
    print(f"  train: {results['n_train']} rows  ({results['train_day_span'][0]} .. {results['train_day_span'][1]})")
    print(f"  test:  {results['n_test']} rows  ({results['test_day_span'][0]} .. {results['test_day_span'][1]})")
    print(f"  shipped: {results['model_type']}  dwarka_override={results['dwarka_override']}")
    print(f"  baseline MAE: {baseline_mae:.4f}")
    print(f"  model MAE:    {model_mae:.4f}")
    print(f"  skill:        {skill:.4f}")
    print(f"  label agreement: {100 * labels['agree_pct']:.1f}%")
    if ranking["total_pairs"]:
        print(f"  hour-ranking concordance: {100 * ranking['pairwise_concordance']:.1f}% "
              f"(n_groups={ranking['n_groups']}, pairs={ranking['concordant_pairs']}/{ranking['total_pairs']})")
    if importances:
        print("  feature importances (top 5):")
        for name, imp in importances[:5]:
            print(f"    {name:28s} {imp:.4f}")
    print("=" * 70)

    if write_md:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            fh.write(format_report(results))
        print(f"\nWrote {OUT_PATH}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate shipped forecast_residual_gbt.joblib.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print metrics only; do not write docs/forecast_eval.md",
    )
    args = parser.parse_args()
    evaluate(write_md=not args.no_write)


if __name__ == "__main__":
    main()
