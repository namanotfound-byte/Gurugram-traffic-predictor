# Gurugram Traffic Congestion Predictor

**Naman Arora — Class 12**  
Computer Science Project

Repository: https://github.com/namanotfound-byte/Gurugram-traffic-predictor  
Live site: https://gurugram-traffic-predictor.vercel.app

---

## 1. What it does

If you have to drive one of Gurugram’s main roads, what time should you leave?

Thirteen corridors × 7 weekdays × 24 hours = **2,184 slots**. The map colours each road, opens on the typical-week hour that saves the most time, and shows Best time / Avoid plus You save versus the quietest hour. A trip planner applies the same residual forecast to the overlapping part of a real TomTom route.

Congestion index:

```
congestion_index = 1 − (free-flow travel time ÷ expected travel time)
```

| Label | Index | Meaning |
|---|---|---|
| Free | below 0.091 | under 1.10× free-flow |
| Moderate | below 0.200 | 1.10× to 1.25× |
| Heavy | below 0.310 | 1.25× to 1.45× |
| Severe | 0.310 and above | over 1.45× |

## 2. Why it is built this way

An earlier version invented its own traffic and scored itself against it. That was deleted. Everything now is measured TomTom Routing data. Missing model → HTTP 503, no fake table.

## 3. Data

- **Baseline:** one TomTom Routing sweep of all 2,184 slots (`departAt` future + `computeTravelTimeFor=all`). Date-blind.
- **Live CSV:** **24,490** rows, **16 Aug 2026 20:00 – 15 Oct 2026 16:00**, **61 days**, 13 corridors.
- **Training rows:** **20,198** after dropping 3,119 wrong-length routes and 1,173 rows with no baseline cell. Training table: 2,313 rainy, 16,562 incident-affected, 6 holidays.
- Collection ran on hourly GitHub Actions (4 rounds per hour), then **stopped** after the 15 Oct train so the submitted snapshot would not keep moving.

## 4. What is served

Not a replay of the latest CSV cell. Each slot is `clip(baseline + residual, 0, 1)`, origin `residual_adjusted`. Static copy: `frontend/data/bundle.json`. Trip planner place/route APIs are Vercel functions.

`backend/test_api.py`: **77 passed, 0 failed** (15 Oct 2026).

## 5. Model

`models/forecast_residual_gbt.joblib` — GradientBoostingRegressor on the residual (live − baseline). Trained 15 Oct 2026 23:14. MLP holdout skill **−1.07** (not shipped). Dwarka has an override estimator.

Importances (shipped GBT): `hour_sin` 0.5469, `lag_prior_idx` 0.1604, `hour_cos` 0.1175, `incident_total_delay_s` 0.0554, `road_class_enc` 0.0492. Time of day ≈ 66%.

## 6. Accuracy (re-run, not copied from an old README)

Time holdout, last 20% of days:

| | |
|---|---|
| Train | 17,980 rows (16 Aug – 3 Oct) |
| Test | 2,218 rows (4 Oct – 15 Oct) |
| Baseline MAE | 0.1007 |
| Model MAE | 0.0461 |
| Skill | **0.5417** |
| Label agreement | **77.9%** |
| Hour-ranking concordance | **88.1%** |
| Production refit | 20,198 rows |

An older bootstrap-vs-observed diagnostic on small n looked like 89% ranking; at full sample it was ~57% ranking / ~43% labels. That was the date-blind formula. The residual model is what the map uses now.

Typical-week extra delay tops out around **13 minutes** (KMP, Monday 08:00 vs 22:00). That is not a 30-minute jam day.

## 7–8. Decisions and limits

Skill gate; one `corridors.py`; 300 m incident buffer; 503 not a fake table. Limits: typical week not live-this-minute; rain flags weak in this GBT; Dwarka is a special case; 3,119 rows lost to wrong TomTom length; collection stopped 15 Oct.

## Technical summary

Python + Flask + MapLibre; scikit-learn GBT residual; TomTom Routing + Incidents, Open-Meteo; host **Vercel**. Run: `./run.sh` (not `file://`).

*Figures from `gurugram_observed.csv`, the joblib artifact, `evaluate_forecast.py --no-write`, `forecast_model.py readiness`, and pytest, 15–16 Oct 2026.*
