# Accuracy Report — served numbers vs. measured reality

_Generated 2026-08-17 12:39 UTC by `tools/evaluate_accuracy.py`. Re-run this script as `data/gurugram_observed.csv` grows; it overwrites this file with the latest snapshot and appends one row to `docs/accuracy_history.csv` so trend-over-time can be tracked._

## Headline

**Coverage is still very thin: 76 of 2184 cells (3.5%) have ever been observed, from 115 collected rows spanning 2026-08-16T20:00:00 to 2026-08-17T11:45:00.** The numbers below are real (not fabricated, not tuned), but treat every headline figure here as **LOW-MODERATE (n still small for a product-level claim)** until coverage grows. See the Coverage section for exactly how much more data would change that.

- Point error (served vs. observed `congestion_index`, n=115): **MAE = 0.057**, **RMSE = 0.075**, **bias = -0.017** (95% CI -0.031 to -0.004).
  A negative bias means the site systematically **UNDERSTATES** real congestion (served value below what was actually measured).
- Label agreement (what users actually see, n=115): **58.3%** exact match (95% CI 49.1%–66.9%). **28.7%** of the time the site showed a label *better* than reality (the dangerous direction), **13.0%** of the time *worse* than reality (merely annoying).
- Advice-level hour ranking (n=13 corridor/day groups, 142 comparable hour-pairs): pairwise concordance **89.4%**, best-hour-hit rate **76.9%**, worst-hour-hit rate **76.9%**.

## What this compares

- **Served** = `data/gurugram_bootstrap.csv`'s `congestion_idx` for a (corridor, day-of-week, hour) cell — TomTom's historical-model average (`1 - noTraffic/historic`), with no date sensitivity. This is what `backend/app.py` serves for any cell that has not (yet) been directly observed — today that is 96.5% of all cells.
- **Observed** = `data/gurugram_observed.csv`'s `congestion_idx` for the same cell — a real measurement collected by CI (`1 - noTraffic/live`) at some actual date/time that fell into that (corridor, day-of-week, hour) bucket.
- Both use the same `free_flow` numerator, so the two `congestion_idx` values are directly comparable — this is *not* comparing two different quantities.
- **One caveat, for completeness:** the live backend's `load_measured_grid()` (`backend/app.py`) actually overwrites a cell with the *freshest matching observation* once one exists for that exact cell, so a handful of cells are, right now, serving the observed value verbatim (self-matching by construction). This evaluation deliberately measures the underlying **bootstrap** model instead, because that is what is served for the overwhelming majority of cells (everything not yet observed), and it is what was being served for every one of these comparison rows at the moment they were actually collected.

## Coverage

- Corridors: 13 (13, after the 5 added 2026-08-17)
- Total cells (corridors x 7 days x 24 hours): 2184
- Cells with at least one observation: 76 (3.5%)
- Observed rows collected so far: 115
- Days of week with any coverage: Monday, Sunday
- Days of week with ZERO coverage: Tuesday, Wednesday, Thursday, Friday, Saturday
- Collection window: 2026-08-16T20:00:00 → 2026-08-17T11:45:00

At the current CI cadence (~40 min/sweep, 13 corridors x 1 hour-of-day per sweep), reaching even 500 covered cells (~23% coverage, still thin) needs roughly 424 more distinct (corridor, day, hour) cells to be hit — coverage grows slower than row count because the same popular hours get re-sampled before new ones are reached. Full 2184-cell coverage (every hour of every day) requires the collector to run across all 7 days, which it has not yet done (see missing days above).

## 1. Point-error metrics (served vs. observed `congestion_index`)

Overall, n=115 — confidence tier: **LOW-MODERATE (n still small for a product-level claim)**

| metric | value | 95% CI |
|---|---|---|
| MAE | 0.057 | 0.048 – 0.066 |
| RMSE | 0.075 | — |
| Bias (mean signed error, served − observed) | -0.017 | -0.031 – -0.004 |
| p50 abs error | 0.045 | — |
| p90 abs error | 0.137 | — |
| max abs error | 0.201 | — |
| p10 / p50 / p90 signed error | -0.127 / -0.016 / 0.065 | — |

### Per corridor

| corridor | n | MAE | RMSE | bias | tier |
|---|---|---|---|---|---|
| NH-48 Delhi-Gurgaon Expressway | 10 | 0.078 | 0.089 | -0.071 | INSUFFICIENT (do not draw conclusions) |
| MG Road | 10 | 0.041 | 0.051 | 0.029 | INSUFFICIENT (do not draw conclusions) |
| Golf Course Road | 10 | 0.070 | 0.083 | -0.048 | INSUFFICIENT (do not draw conclusions) |
| Sohna Road | 10 | 0.038 | 0.042 | -0.012 | INSUFFICIENT (do not draw conclusions) |
| Dwarka Expressway | 10 | 0.022 | 0.024 | -0.011 | INSUFFICIENT (do not draw conclusions) |
| Golf Course Extension Road | 10 | 0.033 | 0.040 | 0.013 | INSUFFICIENT (do not draw conclusions) |
| Mehrauli-Gurgaon Road | 10 | 0.044 | 0.052 | 0.041 | INSUFFICIENT (do not draw conclusions) |
| Southern Peripheral Road | 10 | 0.095 | 0.110 | -0.037 | INSUFFICIENT (do not draw conclusions) |
| KMP Expressway (Western Peripheral Expressway) | 7 | 0.086 | 0.115 | 0.083 | INSUFFICIENT (do not draw conclusions) |
| Delhi-Mumbai Expressway | 7 | 0.032 | 0.037 | -0.032 | INSUFFICIENT (do not draw conclusions) |
| NH-352W (Gurugram-Sohna-Alwar Road) | 7 | 0.095 | 0.099 | -0.095 | INSUFFICIENT (do not draw conclusions) |
| Old Delhi-Gurgaon Road | 7 | 0.100 | 0.126 | -0.095 | INSUFFICIENT (do not draw conclusions) |
| Pataudi Road | 7 | 0.020 | 0.021 | -0.003 | INSUFFICIENT (do not draw conclusions) |

### Per road class

| road class | n | MAE | RMSE | bias | tier |
|---|---|---|---|---|---|
| arterial | 60 | 0.053 | 0.068 | -0.002 | LOW (preliminary only) |
| highway | 31 | 0.074 | 0.092 | -0.067 | LOW (preliminary only) |
| expressway | 24 | 0.044 | 0.067 | 0.010 | INSUFFICIENT (do not draw conclusions) |

## 2. Label agreement (what the user actually sees)

Thresholds (from `backend/app.py`, matching `docs/api_contract.md`): Free < 0.091, Moderate < 0.2, Heavy < 0.31, Severe ≥ 0.31

n=115 — confidence tier: **LOW-MODERATE (n still small for a product-level claim)**

- Exact label match: **58.3%** (95% CI 49.1%–66.9%)
- Understated (served label better than observed — **dangerous**, user leaves at a time we called clear/moderate but was actually worse): 33 / 115 = **28.7%**
- Overstated (served label worse than observed — merely annoying): 15 / 115 = **13.0%**

### Confusion matrix (rows = served label, columns = observed label)

| served \ observed | Free | Moderate | Heavy | Severe | row total |
|---|---|---|---|---|---|
| **Free** | 27 | 6 | 0 | 0 | 33 |
| **Moderate** | 2 | 29 | 18 | 7 | 56 |
| **Heavy** | 0 | 9 | 11 | 2 | 22 |
| **Severe** | 0 | 2 | 2 | 0 | 4 |
| **col total** | 29 | 46 | 31 | 9 | 115 |

## 3. Advice-level accuracy (hour ranking within a corridor/day)

The site's core claim is "leave at hour X, avoid hour Y." That claim only makes sense to check where we have observed data at 2+ distinct hours for the same corridor and day-of-week, so we can ask: did the served ranking of those hours match the observed ranking?

n=13 corridor/day groups, 142 comparable hour-pairs (8 tied pairs excluded) — confidence tier: **INSUFFICIENT (do not draw conclusions)**

- Pairwise concordance (served says A vs B in the same order reality did): **89.4%**
- Best-hour-hit rate (served's best hour among observed hours = observed's actual best hour): **76.9%** (10/13)
- Worst-hour-hit rate: **76.9%** (10/13)

**Important limitation:** every group below comes from Monday (`day_of_week=0`) — that is the only day with enough distinct observed hours to rank. This says nothing yet about weekday-vs-weekend or other days.

| corridor | day | hours observed | n pairs | pairwise concordance | best-hour hit | worst-hour hit |
|---|---|---|---|---|---|---|
| NH-48 Delhi-Gurgaon Expressway | Monday | 0, 1, 8, 9, 10, 11 | 14 | 85.7% | yes | no |
| MG Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 100.0% | yes | yes |
| Golf Course Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 92.9% | yes | yes |
| Sohna Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 100.0% | yes | yes |
| Dwarka Expressway | Monday | 0, 1, 8, 9, 10, 11 | 14 | 100.0% | no | yes |
| Golf Course Extension Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 92.9% | yes | yes |
| Mehrauli-Gurgaon Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 92.9% | yes | yes |
| Southern Peripheral Road | Monday | 0, 1, 8, 9, 10, 11 | 14 | 100.0% | no | yes |
| KMP Expressway (Western Peripheral Expressway) | Monday | 8, 9, 10, 11 | 6 | 16.7% | no | no |
| Delhi-Mumbai Expressway | Monday | 8, 9, 10, 11 | 6 | 50.0% | yes | no |
| NH-352W (Gurugram-Sohna-Alwar Road) | Monday | 8, 9, 10, 11 | 6 | 83.3% | yes | yes |
| Old Delhi-Gurgaon Road | Monday | 8, 9, 10, 11 | 6 | 83.3% | yes | yes |
| Pataudi Road | Monday | 8, 9, 10, 11 | 6 | 100.0% | yes | yes |

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
