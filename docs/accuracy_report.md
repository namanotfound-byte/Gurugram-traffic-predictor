# Accuracy Report — served numbers vs. measured reality

_Generated 2026-09-18 12:41 UTC by `tools/evaluate_accuracy.py`. Re-run this script as `data/gurugram_observed.csv` grows; it overwrites this file with the latest snapshot and appends one row to `docs/accuracy_history.csv` so trend-over-time can be tracked._

## Headline


- Point error (served vs. observed `congestion_index`, n=15234): **MAE = 0.116**, **RMSE = 0.153**, **bias = 0.003** (95% CI 0.000 to 0.005).
  A negative bias means the site systematically **OVERSTATES** real congestion (served value below what was actually measured).
- Label agreement (what users actually see, n=15234): **43.0%** exact match (95% CI 42.2%–43.7%). **30.8%** of the time the site showed a label *better* than reality (the dangerous direction), **26.3%** of the time *worse* than reality (merely annoying).
- Advice-level hour ranking (n=91 corridor/day groups, 23704 comparable hour-pairs): pairwise concordance **57.2%**, best-hour-hit rate **13.2%**, worst-hour-hit rate **0.0%**.

## What this compares

- **Served** = `data/gurugram_bootstrap.csv`'s `congestion_idx` for a (corridor, day-of-week, hour) cell — TomTom's historical-model average (`1 - noTraffic/historic`), with no date sensitivity. This is what `backend/app.py` serves for any cell that has not (yet) been directly observed — today that is 0.0% of all cells.
- **Observed** = `data/gurugram_observed.csv`'s `congestion_idx` for the same cell — a real measurement collected by CI (`1 - noTraffic/live`) at some actual date/time that fell into that (corridor, day-of-week, hour) bucket.
- Both use the same `free_flow` numerator, so the two `congestion_idx` values are directly comparable — this is *not* comparing two different quantities.
- **One caveat, for completeness:** the live backend's `load_measured_grid()` (`backend/app.py`) actually overwrites a cell with the *freshest matching observation* once one exists for that exact cell, so a handful of cells are, right now, serving the observed value verbatim (self-matching by construction). This evaluation deliberately measures the underlying **bootstrap** model instead, because that is what is served for the overwhelming majority of cells (everything not yet observed), and it is what was being served for every one of these comparison rows at the moment they were actually collected.

## Coverage

- Corridors: 13 (13, after the 5 added 2026-08-17)
- Total cells (corridors x 7 days x 24 hours): 2184
- Cells with at least one observation: 2184 (100.0%)
- Observed rows collected so far: 15234
- Days of week with any coverage: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
- Days of week with ZERO coverage: none
- Collection window: 2026-08-16T20:00:00 → 2026-09-11T15:30:00

At the current CI cadence (~40 min/sweep, 13 corridors x 1 hour-of-day per sweep), reaching even 500 covered cells (~23% coverage, still thin) needs roughly 0 more distinct (corridor, day, hour) cells to be hit — coverage grows slower than row count because the same popular hours get re-sampled before new ones are reached. Full 2184-cell coverage (every hour of every day) requires the collector to run across all 7 days, which it has not yet done (see missing days above).

## 1. Point-error metrics (served vs. observed `congestion_index`)

Overall, n=15234 — confidence tier: **LOW-MODERATE (n still small for a product-level claim)**

| metric | value | 95% CI |
|---|---|---|
| MAE | 0.116 | 0.115 – 0.118 |
| RMSE | 0.153 | — |
| Bias (mean signed error, served − observed) | 0.003 | 0.000 – 0.005 |
| p50 abs error | 0.087 | — |
| p90 abs error | 0.262 | — |
| max abs error | 0.663 | — |
| p10 / p50 / p90 signed error | -0.190 / 0.001 / 0.198 | — |

### Per corridor

| corridor | n | MAE | RMSE | bias | tier |
|---|---|---|---|---|---|
| NH-48 Delhi-Gurgaon Expressway | 1173 | 0.117 | 0.139 | -0.022 | LOW-MODERATE (n still small for a product-level claim) |
| MG Road | 1173 | 0.175 | 0.220 | 0.000 | LOW-MODERATE (n still small for a product-level claim) |
| Golf Course Road | 1173 | 0.131 | 0.162 | 0.001 | LOW-MODERATE (n still small for a product-level claim) |
| Sohna Road | 1173 | 0.155 | 0.197 | 0.002 | LOW-MODERATE (n still small for a product-level claim) |
| Dwarka Expressway | 1173 | 0.041 | 0.055 | -0.005 | LOW-MODERATE (n still small for a product-level claim) |
| Golf Course Extension Road | 1173 | 0.110 | 0.139 | 0.028 | LOW-MODERATE (n still small for a product-level claim) |
| Mehrauli-Gurgaon Road | 1173 | 0.148 | 0.189 | 0.021 | LOW-MODERATE (n still small for a product-level claim) |
| Southern Peripheral Road | 1173 | 0.136 | 0.167 | -0.011 | LOW-MODERATE (n still small for a product-level claim) |
| KMP Expressway (Western Peripheral Expressway) | 1170 | 0.119 | 0.145 | 0.044 | LOW-MODERATE (n still small for a product-level claim) |
| Delhi-Mumbai Expressway | 1170 | 0.066 | 0.079 | 0.007 | LOW-MODERATE (n still small for a product-level claim) |
| NH-352W (Gurugram-Sohna-Alwar Road) | 1170 | 0.103 | 0.126 | -0.026 | LOW-MODERATE (n still small for a product-level claim) |
| Old Delhi-Gurgaon Road | 1170 | 0.151 | 0.184 | -0.017 | LOW-MODERATE (n still small for a product-level claim) |
| Pataudi Road | 1170 | 0.062 | 0.078 | 0.014 | LOW-MODERATE (n still small for a product-level claim) |

### Per road class

| road class | n | MAE | RMSE | bias | tier |
|---|---|---|---|---|---|
| arterial | 7038 | 0.143 | 0.181 | 0.007 | LOW-MODERATE (n still small for a product-level claim) |
| highway | 4683 | 0.108 | 0.137 | -0.013 | LOW-MODERATE (n still small for a product-level claim) |
| expressway | 3513 | 0.075 | 0.101 | 0.015 | LOW-MODERATE (n still small for a product-level claim) |

## 2. Label agreement (what the user actually sees)

Thresholds (from `backend/app.py`, matching `docs/api_contract.md`): Free < 0.091, Moderate < 0.2, Heavy < 0.31, Severe ≥ 0.31

n=15234 — confidence tier: **LOW-MODERATE (n still small for a product-level claim)**

- Exact label match: **43.0%** (95% CI 42.2%–43.7%)
- Understated (served label better than observed — **dangerous**, user leaves at a time we called clear/moderate but was actually worse): 4691 / 15234 = **30.8%**
- Overstated (served label worse than observed — merely annoying): 3999 / 15234 = **26.3%**

### Confusion matrix (rows = served label, columns = observed label)

| served \ observed | Free | Moderate | Heavy | Severe | row total |
|---|---|---|---|---|---|
| **Free** | 4518 | 1868 | 827 | 317 | 7530 |
| **Moderate** | 1536 | 1442 | 821 | 560 | 4359 |
| **Heavy** | 1183 | 739 | 548 | 298 | 2768 |
| **Severe** | 425 | 60 | 56 | 36 | 577 |
| **col total** | 7662 | 4109 | 2252 | 1211 | 15234 |

## 3. Advice-level accuracy (hour ranking within a corridor/day)

The site's core claim is "leave at hour X, avoid hour Y." That claim only makes sense to check where we have observed data at 2+ distinct hours for the same corridor and day-of-week, so we can ask: did the served ranking of those hours match the observed ranking?

n=91 corridor/day groups, 23704 comparable hour-pairs (1412 tied pairs excluded) — confidence tier: **LOW (preliminary only)**

- Pairwise concordance (served says A vs B in the same order reality did): **57.2%**
- Best-hour-hit rate (served's best hour among observed hours = observed's actual best hour): **13.2%** (12/91)
- Worst-hour-hit rate: **0.0%** (0/91)

**Important limitation:** every group below comes from Monday (`day_of_week=0`) — that is the only day with enough distinct observed hours to rank. This says nothing yet about weekday-vs-weekend or other days.

| corridor | day | hours observed | n pairs | pairwise concordance | best-hour hit | worst-hour hit |
|---|---|---|---|---|---|---|
| NH-48 Delhi-Gurgaon Expressway | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 57.9% | no | no |
| NH-48 Delhi-Gurgaon Expressway | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 61.3% | no | no |
| NH-48 Delhi-Gurgaon Expressway | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.5% | no | no |
| NH-48 Delhi-Gurgaon Expressway | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 57.7% | no | no |
| NH-48 Delhi-Gurgaon Expressway | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 58.6% | yes | no |
| NH-48 Delhi-Gurgaon Expressway | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.5% | no | no |
| NH-48 Delhi-Gurgaon Expressway | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 61.2% | no | no |
| MG Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 56.7% | no | no |
| MG Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 47.9% | no | no |
| MG Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 55.0% | no | no |
| MG Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.9% | no | no |
| MG Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 57.5% | yes | no |
| MG Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 60.0% | no | no |
| MG Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 65.4% | yes | no |
| Golf Course Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 59.8% | yes | no |
| Golf Course Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 55.6% | no | no |
| Golf Course Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 58.8% | no | no |
| Golf Course Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 58.2% | no | no |
| Golf Course Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.9% | yes | no |
| Golf Course Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 57.3% | no | no |
| Golf Course Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 62.3% | no | no |
| Sohna Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 56.3% | no | no |
| Sohna Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.8% | no | no |
| Sohna Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 55.6% | no | no |
| Sohna Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 55.0% | no | no |
| Sohna Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.4% | yes | no |
| Sohna Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 61.5% | no | no |
| Sohna Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 67.0% | no | no |
| Dwarka Expressway | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 58.7% | no | no |
| Dwarka Expressway | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 258 | 55.8% | no | no |
| Dwarka Expressway | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 258 | 57.0% | no | no |
| Dwarka Expressway | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 258 | 57.8% | no | no |
| Dwarka Expressway | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 257 | 59.1% | no | no |
| Dwarka Expressway | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 59.6% | no | no |
| Dwarka Expressway | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 56.8% | no | no |
| Golf Course Extension Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 56.7% | no | no |
| Golf Course Extension Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 53.6% | no | no |
| Golf Course Extension Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 56.9% | no | no |
| Golf Course Extension Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 61.7% | no | no |
| Golf Course Extension Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 56.7% | yes | no |
| Golf Course Extension Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 58.6% | no | no |
| Golf Course Extension Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 63.1% | no | no |
| Mehrauli-Gurgaon Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.5% | no | no |
| Mehrauli-Gurgaon Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.1% | no | no |
| Mehrauli-Gurgaon Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 54.2% | no | no |
| Mehrauli-Gurgaon Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.4% | no | no |
| Mehrauli-Gurgaon Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 51.7% | no | no |
| Mehrauli-Gurgaon Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 61.5% | no | no |
| Mehrauli-Gurgaon Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 63.8% | no | no |
| Southern Peripheral Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.9% | no | no |
| Southern Peripheral Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.0% | no | no |
| Southern Peripheral Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 53.6% | no | no |
| Southern Peripheral Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 51.0% | no | no |
| Southern Peripheral Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 53.3% | yes | no |
| Southern Peripheral Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 56.2% | no | no |
| Southern Peripheral Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 62.8% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 62.8% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 44.1% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.9% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 45.4% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 59.0% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 52.9% | no | no |
| KMP Expressway (Western Peripheral Expressway) | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 53.1% | no | no |
| Delhi-Mumbai Expressway | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 58.5% | no | no |
| Delhi-Mumbai Expressway | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 59.2% | no | no |
| Delhi-Mumbai Expressway | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 57.5% | no | no |
| Delhi-Mumbai Expressway | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 59.8% | no | no |
| Delhi-Mumbai Expressway | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.9% | yes | no |
| Delhi-Mumbai Expressway | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.9% | no | no |
| Delhi-Mumbai Expressway | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 63.6% | yes | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 55.6% | no | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 55.6% | no | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 59.4% | no | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 52.9% | no | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 57.1% | yes | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.4% | no | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 62.8% | no | no |
| Old Delhi-Gurgaon Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 51.7% | no | no |
| Old Delhi-Gurgaon Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 47.9% | no | no |
| Old Delhi-Gurgaon Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 50.6% | no | no |
| Old Delhi-Gurgaon Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 54.0% | no | no |
| Old Delhi-Gurgaon Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 51.3% | no | no |
| Old Delhi-Gurgaon Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 57.1% | no | no |
| Old Delhi-Gurgaon Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 62.2% | no | no |
| Pataudi Road | Monday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 259 | 56.4% | no | no |
| Pataudi Road | Tuesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 58.2% | no | no |
| Pataudi Road | Wednesday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 62.8% | no | no |
| Pataudi Road | Thursday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 60.5% | no | no |
| Pataudi Road | Friday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 62.8% | yes | no |
| Pataudi Road | Saturday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 261 | 58.6% | no | no |
| Pataudi Road | Sunday | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 | 260 | 65.4% | no | no |

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
