# CLAUDE.md — orientation for Claude Code sessions in this repo

## ⚠️ FIRST — check whether this machine is set up yet

Before anything else, check whether `.venv_backend/bin/python` exists.

**If it does, skip this whole section** — the machine is set up, carry on with
whatever was asked.

**If it does not**, this is a freshly transferred copy on a new machine, and the
user is very likely to have forgotten the setup steps. Do not wait to be asked
and do not just run `setup.sh` silently. Walk them through it, in this order,
and **stop and wait** at each step marked WAIT, because those need a human and
you cannot do them.

Open by telling them, in your own words: this looks like a fresh copy, here is
what needs to happen, it takes about five minutes, and two steps need them.

### Step 1 — Command Line Tools (WAIT for them)

Run `git --version`. If it prints a version, this step is already done — say so
and move on.

If it errors, or if `python3 -c 'print(1)'` fails, tell them:

> Your Mac needs Apple's Command Line Tools before anything here can run. On a
> new Mac `/usr/bin/python3` exists but is only a stub. Run this yourself and
> click through the installer window that appears:
>
>     xcode-select --install
>
> It takes a few minutes. Tell me when it has finished.

Then **stop and wait for them to confirm.** Do not attempt to click through the
installer or work around it, and do not move to step 2 until `git --version` and
`python3 -c 'print(1)'` both succeed.

### Step 2 — Get out of ~/Downloads (WAIT if it applies)

Check the working directory. If the project sits under `~/Downloads`, tell them
to move it to `~/Documents` or `~/Developer` and restart you there — macOS
quarantines files downloaded from the internet and Gatekeeper will block the
scripts. If Gatekeeper complains anyway, `bash setup.sh` works where
`./setup.sh` is refused.

### Step 3 — Run the setup (you do this)

Run `./setup.sh`. Warn them first that it downloads roughly 1 GB of Python
packages and takes two to three minutes, so they want to be on wifi. Report what
it prints.

### Step 4 — Pull the latest data (you do this)

Run `git pull`. The GitHub Actions collector commits new data on its own
schedule, so this copy will already be behind. If the pull is blocked by
`frontend/data/bundle.json`, that file is regenerated daily and is safe to
discard: `git checkout -- frontend/data/bundle.json`, then pull again.

Note that `frontend/index.html` may have uncommitted changes (a Day/Night toggle
redesign). Check `git diff` before doing anything that would discard it, and
tell them it is there rather than deciding for them.

### Step 5 — Verify (you do this)

Run `.venv_backend/bin/python -m pytest backend/test_api.py -q`.

**Expect 76 passed, 3 failed.** Tell them this is the expected result and not
something the move broke — the three failures are stale fixtures, explained
under "Known issues" below. Do not try to fix them unless asked.

Then run `./run.sh` and confirm the site comes up.

### Step 6 — GitHub push access (WAIT, and only if they need it)

Everything works read-only without this, so only raise it if they want to push.
Tell them:

> To push, GitHub needs a personal access token — your password will not work.
> Go to https://github.com/settings/tokens → Generate new token (classic), tick
> the **repo** scope, and copy it. When git asks for a username enter
> `namanotfound-byte`, and paste the token as the password. macOS will remember
> it after that.

You cannot do this part for them.

### Finally

Tell them setup is done, restate that 3 failing tests are expected, and point at
the "Known issues" list below for what is worth doing next. Then ask what they
want to work on.

---


Read this before doing anything. It is short on purpose; `README.md` has the
full picture and `PROJECT_EXPLAINER.md` has the file-by-file walkthrough.

## What this is

A congestion predictor for 13 Gurugram road corridors, built as a Class 12 CS
project by Naman Arora. For each corridor it holds a congestion index for every
hour of every day of the week (13 × 7 × 24 = 2,184 slots) and tells the user
when to leave. Flask API + static MapLibre frontend, deployed on GitHub Pages,
data collected autonomously by GitHub Actions.

## Hard rules — these are not style preferences

1. **Never quote a metric from the README, the docs, or a comment without
   re-deriving it.** This repo has a documented history of invented numbers
   presented as measured: a fabricated R²=0.83 from synthetic data, a
   hand-typed `PROFILES` lookup table served as if it were model output, and a
   train/serve label-encoding mismatch that silently mislabelled every road
   class. All three were pre-existing and all three looked fine. Re-run
   `tools/evaluate_accuracy.py` or the training script and use what it prints.

2. **Fail loudly rather than falling back to something plausible.** Model
   endpoints return HTTP 503 when no model is loaded. Do not add a default
   table, a cached guess, or an interpolated value. Tests enforce this.

3. **Provenance stays visible all the way to the UI.** Every served value
   carries `observed` / `bootstrap` / `synthetic`. This is frozen in
   `docs/api_contract.md`.

4. **No synthetic data.** The original project generated its own traffic and
   scored itself against it. That was deleted. Everything now is measured.

5. **The project is examined viva-style.** Naman has to justify every file and
   import to a teacher. Prefer a choice that is easy to explain out loud over a
   clever one that is not, and when you reject an approach, record *why* — the
   reasoning is part of the deliverable.

## State as of 2026-08-31 (all figures re-derived on this date)

- **Data:** 2,184-slot bootstrap grid, complete. Observed set is **11,438 rows**
  over 16 days (16–31 Aug), all 7 weekdays, **100% grid coverage**. 1,841 rainy
  rows, 9,612 carrying incident data.
- **Accuracy has been revised down.** The README and the live site still quote
  89.4% ranking concordance / 58.3% label agreement — those were measured on
  **115 rows**. Against all 11,438: **41.7% label agreement, 57.1% ranking
  concordance, MAE 0.117, worst-hour hit rate 0.0%**. The README has not been
  updated yet. `PROJECT_DESCRIPTION.md` carries the corrected figures.
- **Model:** GradientBoostingRegressor, 2,038 training rows. In-sample R² 0.786.
  Leave-one-corridor-out CV mean R² **−0.353 ± 3.519**; Dwarka Expressway is the
  outlier fold at **−12.42** (real construction traffic, not a bug to average
  away). Time-of-day carries ~80% of feature importance.
- **Residual model** (`model/forecast_model.py`) is deliberately gated and will
  refuse to train until its data thresholds are met. That is by design. Run
  `python model/forecast_model.py readiness` to see where it stands.
- **Tests: 76 pass, 3 fail.** Stale fixtures, not broken behaviour — see below.

## Known issues, in priority order

1. **Three failing tests.** `test_served_value_matches_the_csv_exactly_not_the_model`,
   `test_unstable_cell_confidence_materially_lower_than_stable`, and
   `test_happy_path`. Written when the observed set was ~100 rows and nearly
   every cell was served from the baseline; the collector has since made every
   cell observed. Fix the assertions, not the application.
2. **No workflow runs the test suite**, which is why #1 went unnoticed for two
   weeks. Adding a `pytest` step to CI is the obvious fix.
3. **The published accuracy numbers are stale** (see above). The README and the
   `accuracy` block in `frontend/data/bundle.json` both need regenerating.
4. **Intro animation is broken** — `frontend/intro.{css,js}` draws corridor
   polylines via `stroke-dashoffset` but renders in reverse (erases instead of
   draws). Two suspects: inverted `L→0` direction, and the initial offset being
   applied after first paint (needs set offset → force reflow → enable
   transition → animate). Not yet wired into `index.html`. Wanted: ~4s, all 13
   corridors at once, plays every time, skippable.
5. **`frontend/index.html` may have uncommitted changes** — a Day/Night toggle
   redesign moving the time ranges below the buttons. Check `git diff` first.

## Gotchas that have already cost time

- **The TomTom key is Routing-only.** Flow Segment and Search returned 403.
  Both live and historical traffic come from Routing alone: `departAt=<future>`
  + `computeTravelTimeFor=all` for the historical model, `departAt=now` for
  live, `noTrafficTravelTimeInSeconds` as the free-flow baseline either way.
  Re-test entitlement before reaching for any other TomTom API. Geocoding uses
  OSM Nominatim instead. Free tier is 2,500 non-tile requests/day; collection
  already uses ~1,344.
- **CI commits constantly.** The data workflow pushes on its own schedule, so
  the remote drifts ahead. `git pull` before starting. If it blocks on
  `frontend/data/bundle.json`, that file is regenerated daily —
  `git checkout -- frontend/data/bundle.json` and pull again.
- **Never open the frontend as a `file://` URL.** Browsers block it from
  fetching the map geometry and data. Use `./run.sh` or any HTTP server.
- **`corridors.py` is the single source of truth** for corridor coordinates and
  road class. Nothing may redefine them locally.
- **Don't feed `corridor_id` to the model.** It was tested and made
  leave-one-corridor-out R² worse (−1.234 vs −0.353), and the deployed
  `predict_raw()` contract is road-class-based.

## Commands

```bash
./setup.sh                                              # one-time, rebuilds venvs
./run.sh                                                # backend + frontend + browser
.venv_backend/bin/python -m pytest backend/test_api.py -q
.venv_backend/bin/python tools/evaluate_accuracy.py     # regenerates docs/accuracy_report.md
.venv_forecast/bin/python tools/evaluate_forecast.py    # holdout eval → docs/forecast_eval.md
.venv_forecast/bin/python model/forecast_model.py readiness
.venv_forecast/bin/python model/forecast_model.py train # retrains + writes forecast artifact
```

Note `setup.sh` builds against whatever `python3` is on the machine. Apple's
stock Python is 3.9, while CI uses 3.11 — the pinned-version gap produces
`InconsistentVersionWarning` noise when unpickling the CI-trained model. It is
noise, not the cause of the failing tests.

## Historical context

`.claude/memory-from-old-mac/` holds the memory files from the machine this
project was developed on, carried over so the reasoning behind past decisions
isn't lost. **Parts are stale** — `project_overview.md` in particular still
describes 8 corridors and synthetic data, both long superseded. Treat it as a
record of how the project got here, not as a description of what it is now.
