# Forecast model evaluation

_Generated 2026-09-28T10:08:18+00:00 by `tools/evaluate_forecast.py`._

## Holdout split (time-based, last 20% of distinct days)

- Train rows: **14797** (2026-08-16 .. 2026-09-19)
- Test rows: **2066** (2026-09-20 .. 2026-09-28)
- Distinct calendar days in table: **44**

## Shipped artifact

- Path: `models/forecast_residual_gbt.joblib`
- Model type: **GBT**
- Dwarka corridor override (id 4): **yes**
- Trained at (artifact): 2026-09-28T09:20:43.239254
- Artifact skill_score: 0.5550

## Test-set metrics (recomputed this run)

- Baseline MAE (bootstrap vs observed): **0.0970**
- Model MAE (baseline + residual vs observed): **0.0418**
- Skill = 1 - model_MAE/baseline_MAE: **0.5686**
- Label agreement (exact): **82.2%** (n=2066)
- Hour-ranking pairwise concordance (corridor+date groups, n=114): **91.7%** (3368/3674 pairs)

## Feature importances (shipped model)

- `hour_sin`: 0.5404
- `lag_prior_idx`: 0.1784
- `hour_cos`: 0.1103
- `incident_total_delay_s`: 0.0437
- `road_class_enc`: 0.0433
- `lag_week_idx`: 0.0264
- `nearest_incident_m`: 0.0191
- `temperature_c`: 0.0162
- `corridor_id`: 0.0091
- `is_weekend`: 0.0082
- `incident_count`: 0.0025
- `days_to_nearest_holiday`: 0.0011
- `rain_last_3h`: 0.0003
- `incident_max_magnitude`: 0.0003
- `visibility_m`: 0.0002
- `precipitation_mm`: 0.0001
- `has_road_closure_i`: 0.0001
- `is_festival_period_i`: 0.0001
- `incident_data_known`: 0.0001
- `has_jam_i`: 0.0001
- `is_holiday_i`: 0.0000
- `is_raining_i`: 0.0000
- `low_visibility_i`: 0.0000
- `is_month_end_i`: 0.0000
