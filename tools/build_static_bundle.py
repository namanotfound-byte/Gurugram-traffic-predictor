#!/usr/bin/env python3
"""
Generate frontend/data/bundle.json — a static snapshot of everything the
Flask API (backend/app.py) serves, so the live GitHub Pages site needs no
server at all.
=====================================================================
Why this is correct, not a shortcut: backend/app.py precomputes its full
13x7x24 (corridor x day x hour) grid ONCE at startup from measured CSVs
plus a gap-filling model, and that grid only changes when data is
re-collected or the model is retrained (see backend/app.py's module
docstring and docs/api_contract.md). /now and /best-time are both just
different views over that same static grid, evaluated against whatever
"now" the browser's clock says. So a static JSON snapshot of the grid is
enough to reconstruct every endpoint's behaviour client-side — nothing is
actually dynamic on the server side, so nothing needs a server.

CRITICAL — this script does NOT reimplement backend/app.py's computation.
It imports backend/app.py as a module (exactly like backend/test_api.py
does) and reads its already-precomputed, already-tested GRID, CORRIDORS,
FREE_FLOW_MINUTES, LABEL_THRESHOLDS, and advice_payload_for(). If the two
ever diverged, the static site would silently disagree with the live API
— importing guarantees they cannot.

Usage:
    python3 tools/build_static_bundle.py

Reads models/traffic_gbt.joblib + data/*.csv (via backend/app.py's own
startup logic). Writes frontend/data/bundle.json.

Bundle shape (see the "conventions" block written into the bundle itself
for the authoritative, machine-readable version of this):

{
  "generated_at": "<ISO 8601 UTC>",
  "model_version": "<str>",
  "provenance": "observed" | "bootstrap" | "synthetic",
  "trained_rows": <int|null>,
  "measured_cells": <int>, "inferred_cells": <int>,
  "accuracy": {  // same object GET /health serves under "accuracy" -- see
                 // docs/accuracy_report.md. Surfaces the site's real strength
                 // (ranking hours, ~89% concordance) vs its real weakness
                 // (exact labels, ~58% agreement) -- see "label honesty" in
                 // docs/api_contract.md.
    "label_agreement_pct", "hour_ranking_concordance_pct", "sample_size",
    "as_of", "note"
  },
  "corridors": [
    {"id", "name", "sub", "road_class", "start": [lat,lon], "end": [lat,lon],
     "length_km", "free_flow_minutes"}, ...  // same shape as GET /corridors,
                                              // + free_flow_minutes added
  ],
  "grid": {
    // indexed grid.<field>[corridor_index][day 0-6][hour 0-23].
    // corridor_index == corridors[i]["id"] (ids are 0..N-1 in array order).
    "congestion_index": [[[...24 floats...] x7 days] x13 corridors],
    "confidence":       [[[...24 floats...] x7 days] x13 corridors],
    "origin":           [[[...24 strings...] x7 days] x13 corridors]
                         // "observed" | "bootstrap" | "model_inferred"
  },
  "advice": [
    // one object per (corridor, day) = 13*7 = 91, same shape GET /advice
    // returns (minus the top-level-only "provenance" field). Added
    // 2026-08-17: whole-day best-vs-worst saving in minutes AND percent, so
    // a short corridor's small minute figure still reads as compelling and
    // a narrow-window UI can lead with the corridor-wide number -- see
    // "best-time saving fields" in docs/api_contract.md.
    {"corridor_id", "day", "profile": [24 floats], "best_windows": [...],
     "worst_windows": [...], "best_hour", "peak_hour",
     "best_hour_delay_minutes", "peak_delay_minutes", "peak_delay_pct",
     "whole_day_saving_minutes", "whole_day_saving_pct",
     "summary", "confidence"}
  ],
  "conventions": {
    "label_thresholds": [[0.091,"Free"],[0.2,"Moderate"],[0.31,"Heavy"]],
    "label_else": "Severe",
    "minutes_formula": "typical = free_flow_minutes / (1 - min(idx,0.97)); "
                        "delay = typical - free_flow_minutes; both round(.,1)",
    "day_convention": "0=Monday .. 6=Sunday (Python weekday()), IST hours 0-23"
  }
}

Frontend wiring (a follow-up task): fetch this file once, then reimplement
_trend_for/_verdict_for/_now_text (for "/now") and the min/max-over-window
scan (for "/best-time") in JS against grid.congestion_index — both are
small, already-documented functions in backend/app.py.
"""
import datetime
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
OUT_PATH = os.path.join(ROOT_DIR, "frontend", "data", "bundle.json")

sys.path.insert(0, BACKEND_DIR)


def _fresh_backend_app():
    """(Re)import backend/app.py fresh, exactly like test_api.py's
    _fresh_app(), so its startup logic (model load + grid precompute) runs
    against the CURRENT state of models/traffic_gbt.joblib and data/*.csv
    — never a stale cached import."""
    if "app" in sys.modules:
        del sys.modules["app"]
    import app as app_module
    return app_module


def build_bundle():
    app_module = _fresh_backend_app()

    if not app_module.MODEL_READY:
        raise RuntimeError(
            "backend/app.py has no usable model loaded (models/traffic_gbt.joblib "
            "missing or malformed) -- cannot build a bundle with no data."
        )

    CORRIDORS = app_module.CORRIDORS
    GRID = app_module.GRID

    # ── corridors (+ free_flow_minutes) ─────────────────────────────────────
    # Read from app_module.FREE_FLOW_MINUTES (2dp), NOT from
    # GRID[...]["free_flow_minutes"] (already rounded to 1dp for display).
    # /predict internally computes typical_minutes/delay_minutes from the
    # 2dp value and only rounds to 1dp at the very end (minutes_from_index);
    # if the bundle instead started from the 1dp display value, applying the
    # same formula client-side would double-round and drift up to 0.1min
    # (~6s) from the live API on some cells. Verified empirically against a
    # running backend/app.py instance -- using the 2dp source here makes the
    # client-side minutes_formula match the API exactly, this rounding was
    # the only source of any mismatch found.
    FREE_FLOW_MINUTES = app_module.FREE_FLOW_MINUTES
    corridors_out = []
    for c in CORRIDORS:
        cid = c["id"]
        ff_minutes = FREE_FLOW_MINUTES[cid]
        corridors_out.append({
            "id": cid,
            "name": c["name"],
            "sub": c["sub"],
            "road_class": c["road_class"],
            "start": list(c["start"]),
            "end": list(c["end"]),
            "length_km": c["verified_km"],
            "free_flow_minutes": ff_minutes,
        })

    # ── grid: columnar (parallel-array) layout, indexed [corridor][day][hour]
    #    -- read straight from the already-precomputed, already-tested GRID
    #    dict; nothing here recomputes a value app.py already computed. ────
    congestion_index, confidence, origin = [], [], []
    measured_cells = 0
    inferred_cells = 0
    for c in CORRIDORS:
        cid = c["id"]
        c_idx, c_conf, c_origin = [], [], []
        for day in range(7):
            d_idx, d_conf, d_origin = [], [], []
            for hour in range(24):
                cell = GRID[(cid, day, hour)]
                d_idx.append(cell["congestion_index"])
                d_conf.append(cell["confidence"])
                d_origin.append(cell["origin"])
                if cell["origin"] == "model_inferred":
                    inferred_cells += 1
                else:
                    measured_cells += 1
            c_idx.append(d_idx)
            c_conf.append(d_conf)
            c_origin.append(d_origin)
        congestion_index.append(c_idx)
        confidence.append(c_conf)
        origin.append(c_origin)

    # ── advice: one object per (corridor, day), reusing
    #    advice_payload_for() verbatim -- the exact function GET /advice and
    #    GET /advice/all call. Zero reimplementation. ───────────────────────
    advice_out = []
    for c in CORRIDORS:
        for day in range(7):
            payload = app_module.advice_payload_for(c["id"], day)
            payload["day"] = day
            advice_out.append(payload)

    bundle = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_version": app_module.MODEL_VERSION,
        "provenance": app_module.MODEL_PROVENANCE,
        "trained_rows": app_module.TRAINED_ROWS,
        "measured_cells": measured_cells,
        "inferred_cells": inferred_cells,
        # Same object GET /health serves under "accuracy" -- read verbatim
        # from app_module.ACCURACY_SUMMARY, never re-typed here, so the
        # static site and the live API can never drift on this figure.
        "accuracy": app_module.ACCURACY_SUMMARY,
        "corridors": corridors_out,
        "grid": {
            "congestion_index": congestion_index,
            "confidence": confidence,
            "origin": origin,
        },
        "advice": advice_out,
        "conventions": {
            "label_thresholds": [list(t) for t in app_module.LABEL_THRESHOLDS],
            "label_else": "Severe",
            "minutes_formula": (
                "typical = free_flow_minutes / (1 - min(congestion_index, 0.97)); "
                "delay = typical - free_flow_minutes; both round(., 1)"
            ),
            "day_convention": "0=Monday .. 6=Sunday (Python weekday()), IST hours 0-23",
            "grid_axes": "grid.<field>[corridor_index][day][hour]; corridor_index == corridors[i].id",
        },
    }
    return bundle, app_module


def main():
    bundle, app_module = build_bundle()

    n_corridors = len(bundle["corridors"])
    n_grid_cells = n_corridors * 7 * 24
    n_advice = len(bundle["advice"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))

    size_bytes = os.path.getsize(OUT_PATH)
    print(f"[build_static_bundle] wrote {OUT_PATH}")
    print(f"[build_static_bundle] corridors={n_corridors} grid_cells={n_grid_cells} "
          f"(measured={bundle['measured_cells']}, inferred={bundle['inferred_cells']}) "
          f"advice_objects={n_advice}")
    print(f"[build_static_bundle] model_version={bundle['model_version']} "
          f"provenance={bundle['provenance']} trained_rows={bundle['trained_rows']}")
    print(f"[build_static_bundle] size={size_bytes} bytes ({size_bytes / 1024:.1f} KB)")

    if size_bytes > 1_000_000:
        print("[build_static_bundle] WARNING: bundle exceeds ~1MB; "
              "consider reducing float precision before committing.")


if __name__ == "__main__":
    main()
