# Forecast model evaluation

_Generated 2026-09-19T18:59:05+00:00 by `tools/evaluate_forecast.py`._

## Holdout split (time-based, last 20% of distinct days)

- Train rows: **12916** (2026-08-16 .. 2026-09-12)
- Test rows: **1789** (2026-09-13 .. 2026-09-19)
- Distinct calendar days in table: **35**

## Shipped artifact

- Path: `models/forecast_residual_gbt.joblib`
- Model type: **GBT**
- Dwarka corridor override (id 4): **yes**
- Trained at (artifact): 2026-09-18T18:22:26.528988
- Artifact skill_score: 0.6125

## Test-set metrics (recomputed this run)

- Baseline MAE (bootstrap vs observed): **0.0997**
- Model MAE (baseline + residual vs observed): **0.0441**
- Skill = 1 - model_MAE/baseline_MAE: **0.5575**
- Label agreement (exact): **78.6%** (n=1789)
- Hour-ranking pairwise concordance (corridor+date groups, n=91): **91.1%** (3092/3393 pairs)

## Feature importances (shipped model)

- `hour_sin`: 0.5182
- `lag_prior_idx`: 0.2105
- `hour_cos`: 0.1153
- `road_class_enc`: 0.0395
- `lag_week_idx`: 0.0309
- `incident_total_delay_s`: 0.0308
- `nearest_incident_m`: 0.0211
- `temperature_c`: 0.0144
- `corridor_id`: 0.0067
- `is_weekend`: 0.0054
- `incident_count`: 0.0025
- `days_to_nearest_holiday`: 0.0022
- `is_festival_period_i`: 0.0008
- `rain_last_3h`: 0.0004
- `visibility_m`: 0.0004
- `incident_max_magnitude`: 0.0002
- `is_month_end_i`: 0.0002
- `has_road_closure_i`: 0.0001
- `precipitation_mm`: 0.0001
- `incident_data_known`: 0.0001
- `has_jam_i`: 0.0001
- `is_holiday_i`: 0.0001
- `is_raining_i`: 0.0000
- `low_visibility_i`: 0.0000
