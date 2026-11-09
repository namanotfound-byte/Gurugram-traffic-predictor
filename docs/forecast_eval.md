# Forecast model evaluation

_Generated 2026-11-09T10:54:27+00:00 by `tools/evaluate_forecast.py`._

## Holdout split (time-based, last 20% of distinct days)

- Train rows: **17980** (2026-08-16 .. 2026-10-03)
- Test rows: **2218** (2026-10-04 .. 2026-10-15)
- Distinct calendar days in table: **61**

## Shipped artifact

- Path: `models/forecast_residual_gbt.joblib`
- Model type: **GBT**
- Dwarka corridor override (id 4): **yes**
- Trained at (artifact): 2026-11-09T10:53:27.965295
- Production fit rows: **20198** (all rows after holdout validation)
- Artifact skill_score: 0.5424

## Test-set metrics (recomputed this run)

- Baseline MAE (bootstrap vs observed): **0.1007**
- Model MAE (baseline + residual vs observed): **0.0461**
- Skill = 1 - model_MAE/baseline_MAE: **0.5424**
- Label agreement (exact): **77.9%** (n=2218)
- Hour-ranking pairwise concordance (corridor+date groups, n=154): **88.1%** (2278/2585 pairs)

## Feature importances (shipped model)

- `hour_sin`: 0.5471
- `lag_prior_idx`: 0.1603
- `hour_cos`: 0.1174
- `incident_total_delay_s`: 0.0556
- `road_class_enc`: 0.0492
- `lag_week_idx`: 0.0258
- `nearest_incident_m`: 0.0148
- `is_weekend`: 0.0101
- `corridor_id`: 0.0099
- `temperature_c`: 0.0054
- `incident_count`: 0.0025
- `days_to_nearest_holiday`: 0.0010
- `visibility_m`: 0.0003
- `incident_max_magnitude`: 0.0002
- `incident_data_known`: 0.0002
- `has_road_closure_i`: 0.0001
- `is_holiday_i`: 0.0001
- `rain_last_3h`: 0.0001
- `is_festival_period_i`: 0.0000
- `precipitation_mm`: 0.0000
- `is_raining_i`: 0.0000
- `low_visibility_i`: 0.0000
- `is_month_end_i`: 0.0000
- `has_jam_i`: 0.0000
