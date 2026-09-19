"""
Tests for the v2 Flask API (backend/app.py).

Covers: every endpoint's happy path, wrap-past-midnight window/best-time
behavior, invalid-input 400s, and the no-model 503 path.

Run:
  pip install -r requirements.txt
  pytest backend/test_api.py -v
"""

import os
import shutil
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BACKEND_DIR, "..")
FORECAST_MODEL_PATH = os.path.join(ROOT_DIR, "models", "forecast_residual_gbt.joblib")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

# The frozen single source of truth (corridors.py) -- derive every
# corridor-count/id expectation from it instead of hardcoding numbers, so
# the suite doesn't rot every time a corridor is added or removed.
from corridors import CORRIDORS  # noqa: E402

N_CORRIDORS = len(CORRIDORS)
VALID_CORRIDOR_IDS = {c["id"] for c in CORRIDORS}
INVALID_CORRIDOR_ID = max(VALID_CORRIDOR_IDS) + 1  # guaranteed out of range


def _fresh_app():
    """(Re)import app.py fresh so its module-level startup logic (forecast
    model load + grid precompute) reruns against the current state of
    models/forecast_residual_gbt.joblib. Needed because the app builds its
    grid once at import time, not per-request."""
    if "app" in sys.modules:
        del sys.modules["app"]
    import app as app_module
    return app_module


@pytest.fixture()
def client():
    app_module = _fresh_app()
    app_module.app.testing = True
    with app_module.app.test_client() as c:
        yield c


# ── label thresholds (recalibrated 2026-08-16 against real bootstrap data) ─

class TestLabelThresholds:
    """Free < 0.091, Moderate < 0.200, Heavy < 0.310, else Severe.
    Sanity-checked against real NH-48 Friday values from the completed
    bootstrap sweep (orchestrator-provided)."""

    @pytest.mark.parametrize("idx,expected", [
        (0.0, "Free"),
        (0.090, "Free"),
        (0.091, "Moderate"),
        (0.199, "Moderate"),
        (0.200, "Heavy"),
        (0.309, "Heavy"),
        (0.310, "Severe"),
        (1.0, "Severe"),
        # real NH-48 Friday cells (bootstrap data), from the orchestrator
        (0.101, "Moderate"),
        (0.157, "Moderate"),
        (0.305, "Heavy"),
        (0.273, "Heavy"),
        (0.001, "Free"),
    ])
    def test_boundaries(self, client, idx, expected):
        app_module = _fresh_app()
        assert app_module.label_for(idx) == expected

    def test_thresholds_constant_matches_contract(self, client):
        """The thresholds live in one named constant (LABEL_THRESHOLDS), not
        duplicated as magic numbers per endpoint."""
        app_module = _fresh_app()
        assert app_module.LABEL_THRESHOLDS == (
            (0.091, "Free"), (0.200, "Moderate"), (0.310, "Heavy"),
        )


# ── /health, /corridors ─────────────────────────────────────────────────

class TestHealthAndCorridors:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.get_json()
        assert body["status"] == "ok"
        assert body["corridors"] == N_CORRIDORS
        for key in ("model_version", "provenance", "trained_rows"):
            assert key in body

    def test_health_surfaces_accuracy_summary(self, client):
        """Bootstrap-vs-observed accuracy (ranking and label agreement)
        must be discoverable in the API, not just buried in
        docs/accuracy_report.md -- see docs/api_contract.md 'label honesty'."""
        r = client.get("/health")
        body = r.get_json()
        assert "accuracy" in body
        acc = body["accuracy"]
        for key in ("label_agreement_pct", "hour_ranking_concordance_pct", "sample_size", "note"):
            assert key in acc
        assert acc["hour_ranking_concordance_pct"] > acc["label_agreement_pct"], (
            "the whole point of surfacing this: ranking is measurably stronger than "
            "absolute labels, and the note should be built on that being true"
        )

    def test_corridors_static_list_not_hardcoded_shape(self, client):
        r = client.get("/corridors")
        assert r.status_code == 200
        body = r.get_json()
        assert len(body["corridors"]) == N_CORRIDORS
        c0 = body["corridors"][0]
        for key in ("id", "name", "sub", "road_class", "start", "end", "length_km"):
            assert key in c0
        assert isinstance(c0["start"], list) and len(c0["start"]) == 2
        assert isinstance(c0["end"], list) and len(c0["end"]) == 2


# ── /predict ─────────────────────────────────────────────────────────────

class TestConfidenceIsHonest:
    """Confidence must never be a flat constant -- it reflects whether the
    served value is a real measurement (and whether that measurement's
    route was stable) or a model-inferred gap-fill. It must NOT be driven
    by metrics["cv_r2"], which is leave-one-corridor-out and (at the time
    this was written) was dominated by the two road classes (expressway,
    highway) that had only one member each -- not representative of
    confidence in a served value."""

    def test_confidence_in_valid_range(self, client):
        for cid in sorted(VALID_CORRIDOR_IDS):
            r = client.get(f"/predict?corridor={cid}&day=1&hour=8")
            conf = r.get_json()["confidence"]
            assert 0.0 <= conf <= 1.0

    def test_strict_ordering_observed_gt_stable_gt_unstable_gt_forecast(self, client):
        app_module = _fresh_app()
        observed = app_module.compute_confidence(
            "observed", None,
            {"origin": "observed", "congestion_idx": 0.1, "route_stable": True},
        )
        measured_stable = app_module.compute_confidence(
            "observed", None,
            {"origin": "bootstrap", "congestion_idx": 0.1, "route_stable": True},
        )
        measured_unstable = app_module.compute_confidence(
            "observed", None,
            {"origin": "bootstrap", "congestion_idx": 0.1, "route_stable": False},
        )
        forecast = app_module.compute_confidence(
            "observed", None, None,
            cell_origin="residual_adjusted", forecast_skill=0.5,
        )

        assert observed > measured_stable > forecast > measured_unstable
        assert measured_stable >= 0.9, "a stable measured cell must read as high confidence (0.9+)"


class TestForecastGrid:
    """The weekly grid must serve residual forecasts (baseline + predicted
    residual), not replay the latest observed CSV cell."""

    def test_full_grid_is_residual_forecast_when_model_loaded(self, client):
        app_module = _fresh_app()
        if not app_module.GRID_READY:
            pytest.skip("forecast model not loaded in this environment")
        assert len(app_module.GRID) == N_CORRIDORS * 7 * 24
        origins = {cell["origin"] for cell in app_module.GRID.values()}
        assert "observed" not in origins, "observed CSV must not be served as the map value"
        assert "residual_adjusted" in origins

    def test_served_value_is_forecast_not_latest_observed(self, client):
        """Mehrauli-Gurgaon Rd (corridor 6), Thursday (day=3) 19:00: the
        observed CSV may read 0.0, but the served value must come from the
        residual forecast path (origin residual_adjusted or bootstrap fallback),
        not a copy of that observed row."""
        app_module = _fresh_app()
        if not app_module.GRID_READY:
            pytest.skip("forecast model not loaded")
        cid, day, hour = 6, 3, 19
        cell = app_module.GRID[(cid, day, hour)]
        r = client.get(f"/predict?corridor={cid}&day={day}&hour={hour}")
        assert r.status_code == 200
        body = r.get_json()
        assert body["congestion_index"] == pytest.approx(cell["congestion_index"], abs=0.001)
        assert cell["origin"] in ("residual_adjusted", "bootstrap")
        if app_module.MEASURED_GRID.get((cid, day, hour), {}).get("origin") == "observed":
            obs_idx = app_module.MEASURED_GRID[(cid, day, hour)]["congestion_idx"]
            if cell["origin"] == "residual_adjusted":
                # Forecast may differ from the last live sample — that is intended.
                pass
            else:
                assert body["congestion_index"] == pytest.approx(
                    app_module.BASELINE_GRID[(cid, day, hour)], abs=0.001,
                )
                assert abs(body["congestion_index"] - obs_idx) >= 0 or cell["origin"] == "bootstrap"

    def test_residual_adjusted_confidence_tier(self, client):
        app_module = _fresh_app()
        if not app_module.FORECAST_SKILL or app_module.FORECAST_SKILL <= 0:
            pytest.skip("forecast skill not positive")
        conf = app_module.compute_confidence(
            "bootstrap", None, None,
            cell_origin="residual_adjusted", forecast_skill=app_module.FORECAST_SKILL,
        )
        assert 0.55 <= conf <= 0.85
        assert conf < app_module.CONFIDENCE_OBSERVED


class TestRouteUnstableConfidence:
    def test_unstable_cell_confidence_materially_lower_than_stable(self, client):
        """Measured-grid tier logic (for evaluation paths): unstable bootstrap
        must read lower than stable bootstrap or observed."""
        app_module = _fresh_app()
        conf_unstable = app_module.compute_confidence(
            app_module.MODEL_PROVENANCE, None,
            {"origin": "bootstrap", "route_stable": False, "congestion_idx": 0.2},
        )
        conf_stable = app_module.compute_confidence(
            app_module.MODEL_PROVENANCE, None,
            {"origin": "bootstrap", "route_stable": True, "congestion_idx": 0.2},
        )
        conf_observed = app_module.compute_confidence(
            app_module.MODEL_PROVENANCE, None,
            {"origin": "observed", "route_stable": True, "congestion_idx": 0.2},
        )
        assert conf_unstable == app_module.CONFIDENCE_MEASURED_UNSTABLE
        assert conf_stable == app_module.CONFIDENCE_MEASURED_STABLE
        assert conf_observed == app_module.CONFIDENCE_OBSERVED
        assert conf_unstable < conf_stable < conf_observed

    def test_served_forecast_confidence_is_not_flat_0_09(self, client):
        """Regression guard: served forecast cells must not read the old
        uniform 0.09 cv_r2 artifact."""
        app_module = _fresh_app()
        if not app_module.GRID_READY:
            pytest.skip("forecast model not loaded")
        r = client.get("/predict?corridor=0&day=1&hour=8")
        conf = r.get_json()["confidence"]
        assert conf >= 0.5, f"expected forecast cell confidence well above 0.09, got {conf}"


class TestPredict:
    def test_happy_path(self, client):
        r = client.get("/predict?corridor=0&day=1&hour=8")
        assert r.status_code == 200
        body = r.get_json()
        for key in ("corridor_id", "day", "hour", "congestion_index", "label",
                    "delay_minutes", "typical_minutes", "free_flow_minutes",
                    "provenance", "confidence", "model_version"):
            assert key in body
        assert 0.0 <= body["congestion_index"] <= 1.0
        assert body["label"] in ("Free", "Moderate", "Heavy", "Severe")
        assert body["provenance"] in ("observed", "bootstrap")
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["typical_minutes"] >= body["free_flow_minutes"]
        # typical and delay are each rounded to 1dp independently in minutes_from_index
        assert body["delay_minutes"] == pytest.approx(
            body["typical_minutes"] - body["free_flow_minutes"], abs=0.15)

    def test_bad_corridor_high(self, client):
        r = client.get("/predict?corridor=99&day=1&hour=8")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_bad_corridor_negative(self, client):
        r = client.get("/predict?corridor=-1&day=1&hour=8")
        assert r.status_code == 400

    def test_bad_day(self, client):
        r = client.get("/predict?corridor=0&day=7&hour=8")
        assert r.status_code == 400

    def test_bad_hour(self, client):
        r = client.get("/predict?corridor=0&day=1&hour=24")
        assert r.status_code == 400

    def test_negative_hour(self, client):
        r = client.get("/predict?corridor=0&day=1&hour=-1")
        assert r.status_code == 400

    def test_non_integer_param(self, client):
        r = client.get("/predict?corridor=abc&day=1&hour=8")
        assert r.status_code == 400

    def test_missing_param(self, client):
        r = client.get("/predict?day=1&hour=8")
        assert r.status_code == 400

    def test_never_500_on_garbage(self, client):
        for qs in ["corridor=&day=&hour=", "corridor=0.5&day=1&hour=8",
                   "corridor=0&day=1&hour=1e10", "corridor=%00&day=1&hour=8"]:
            r = client.get(f"/predict?{qs}")
            assert r.status_code in (400, 200), f"got {r.status_code} for {qs}"


# ── /advice ──────────────────────────────────────────────────────────────

class TestAdvice:
    def test_happy_path(self, client):
        r = client.get("/advice?corridor=0&day=1")
        assert r.status_code == 200
        body = r.get_json()
        assert body["corridor_id"] == 0
        assert body["day"] == 1
        assert len(body["profile"]) == 24
        assert all(0.0 <= v <= 1.0 for v in body["profile"])
        assert 0 <= body["best_hour"] <= 23
        assert 0 <= body["peak_hour"] <= 23
        assert isinstance(body["summary"], str) and len(body["summary"]) > 0
        assert "provenance" in body and "confidence" in body
        for w in body["best_windows"] + body["worst_windows"]:
            for key in ("start_hour", "end_hour", "avg_index", "label", "text"):
                assert key in w

    def test_best_windows_are_merged_not_hourly(self, client):
        """The whole point of window detection: don't emit ~12 one-hour
        windows, merge into a handful of meaningful blocks."""
        r = client.get("/advice?corridor=0&day=1")
        body = r.get_json()
        assert len(body["best_windows"]) <= 4
        assert len(body["worst_windows"]) <= 4

    def test_bad_corridor(self, client):
        r = client.get("/advice?corridor=-1&day=1")
        assert r.status_code == 400

    def test_bad_day(self, client):
        r = client.get("/advice?corridor=0&day=9")
        assert r.status_code == 400

    def test_missing_corridor(self, client):
        r = client.get("/advice?day=1")
        assert r.status_code == 400


class TestAdviceSavingFields:
    """Whole-day best-vs-worst saving (minutes AND percentage) must be a
    first-class part of /advice, not buried behind a narrow window -- see
    docs/api_contract.md 'best-time saving fields' for the rationale. All
    figures must be exactly re-derivable from the GRID cells already served,
    never invented."""

    def test_saving_fields_present_and_consistent(self, client):
        app_module = _fresh_app()
        for cid in sorted(VALID_CORRIDOR_IDS):
            body = app_module.advice_payload_for(cid, 1)
            for key in ("best_hour_delay_minutes", "peak_delay_minutes", "peak_delay_pct",
                        "whole_day_saving_minutes", "whole_day_saving_pct"):
                assert key in body, f"missing {key} for corridor {cid}"

            best_cell = app_module.GRID[(cid, 1, body["best_hour"])]
            peak_cell = app_module.GRID[(cid, 1, body["peak_hour"])]
            assert body["best_hour_delay_minutes"] == pytest.approx(best_cell["delay_minutes"], abs=1e-6)
            assert body["peak_delay_minutes"] == pytest.approx(peak_cell["delay_minutes"], abs=1e-6)
            expected_saving = round(peak_cell["delay_minutes"] - best_cell["delay_minutes"], 1)
            assert body["whole_day_saving_minutes"] == pytest.approx(expected_saving, abs=0.05)
            assert body["whole_day_saving_minutes"] >= -1e-6, "best hour can never be worse than the peak hour"

    def test_route_endpoint_exposes_same_fields(self, client):
        r = client.get("/advice?corridor=1&day=4")
        body = r.get_json()
        assert body["whole_day_saving_minutes"] >= 0
        assert 0.0 <= body["whole_day_saving_pct"] <= 500.0  # sane upper bound, not a tight spec
        assert 0.0 <= body["peak_delay_pct"] <= 500.0

    def test_summary_mentions_percentage_when_saving_is_material(self, client):
        """MG Road Friday (corridor 1, day 4) has a real ~5 min / ~40%+ swing
        between its best and worst hour (see docs/accuracy-adjacent project
        memory) -- the summary must surface both the minute and percent
        figures, not just minutes."""
        r = client.get("/advice?corridor=1&day=4")
        body = r.get_json()
        if body["whole_day_saving_minutes"] >= 0.5:
            assert "%" in body["summary"]
            assert "min" in body["summary"]

    def test_day_night_periods_present_and_well_formed(self, client):
        """Added 2026-08-17: the whole-day best hour is midnight for nearly
        every corridor (roads are empty overnight), which made the site's
        'best time' figure identical and useless across all corridors for a
        daytime traveller. day_period/night_period give each its own
        corridor-specific best/worst hour, confined to DAY_HOURS/NIGHT_HOURS."""
        app_module = _fresh_app()
        for cid in sorted(VALID_CORRIDOR_IDS):
            body = app_module.advice_payload_for(cid, 1)
            for period_key, hours in (("day_period", app_module.DAY_HOURS),
                                       ("night_period", app_module.NIGHT_HOURS)):
                p = body[period_key]
                for key in ("period", "start_hour", "end_hour", "best_hour", "worst_hour",
                            "best_hour_delay_minutes", "worst_hour_delay_minutes",
                            "worst_hour_delay_pct", "saving_minutes", "saving_pct",
                            "best_windows", "worst_windows", "summary", "confidence"):
                    assert key in p, f"{period_key} missing {key} for corridor {cid}"
                assert p["best_hour"] in hours
                assert p["worst_hour"] in hours
                assert p["start_hour"] == hours[0]
                assert p["end_hour"] == hours[-1]
                assert p["saving_minutes"] >= -1e-6

    def test_day_hours_and_night_hours_partition_the_clock(self, client):
        app_module = _fresh_app()
        assert sorted(app_module.DAY_HOURS + app_module.NIGHT_HOURS) == list(range(24))
        assert set(app_module.DAY_HOURS) & set(app_module.NIGHT_HOURS) == set()
        assert app_module.DAY_HOURS == list(range(6, 22))
        assert app_module.NIGHT_HOURS == list(range(22, 24)) + list(range(0, 6))

    def test_night_period_does_not_dominate_day_period(self, client):
        """The whole point of the split: on Monday the whole-day best_hour
        is midnight (a night hour) for every corridor in this dataset, yet
        day_period's best_hour must always come from within DAY_HOURS --
        never silently equal to the whole-day midnight pick."""
        app_module = _fresh_app()
        for cid in sorted(VALID_CORRIDOR_IDS):
            body = app_module.advice_payload_for(cid, 1)
            assert body["day_period"]["best_hour"] in app_module.DAY_HOURS
            if body["best_hour"] not in app_module.DAY_HOURS:
                assert body["day_period"]["best_hour"] != body["best_hour"]

    def test_low_confidence_advice_gets_caveat_text(self, client):
        app_module = _fresh_app()
        summary = app_module.build_advice_summary(
            [{"start_hour": 0, "end_hour": 5, "avg_index": 0.01, "label": "Free", "text": "Clear before 6 AM"}],
            18, 5.0, 5.0, 40.0, 70.0, confidence=0.3,
        )
        assert "Limited data" in summary

    def test_no_caveat_when_confidence_is_healthy(self, client):
        app_module = _fresh_app()
        summary = app_module.build_advice_summary(
            [{"start_hour": 0, "end_hour": 5, "avg_index": 0.01, "label": "Free", "text": "Clear before 6 AM"}],
            18, 5.0, 5.0, 40.0, 70.0, confidence=0.9,
        )
        assert "Limited data" not in summary


# ── /advice/all ──────────────────────────────────────────────────────────

class TestAdviceAll:
    def test_happy_path(self, client):
        r = client.get("/advice/all?day=2")
        assert r.status_code == 200
        body = r.get_json()
        assert body["day"] == 2
        assert "provenance" in body and "model_version" in body
        assert len(body["corridors"]) == N_CORRIDORS
        ids = {c["corridor_id"] for c in body["corridors"]}
        assert ids == VALID_CORRIDOR_IDS
        for c in body["corridors"]:
            assert len(c["profile"]) == 24
            assert "day" not in c        # day is top-level only, per contract
            assert "provenance" not in c  # provenance is top-level only, per contract
            for key in ("best_windows", "worst_windows", "best_hour", "peak_hour",
                        "summary", "confidence"):
                assert key in c

    def test_missing_day(self, client):
        r = client.get("/advice/all")
        assert r.status_code == 400

    def test_bad_day(self, client):
        r = client.get("/advice/all?day=7")
        assert r.status_code == 400


# ── window-detection internals (wrap-past-midnight) ─────────────────────

class TestWindowDetectionWrap:
    def test_merge_runs_wraps_midnight(self, client):
        app_module = _fresh_app()
        flags = [True] * 6 + [False] * 16 + [True] * 2  # hours 0-5 True, 22-23 True
        runs = app_module._merge_runs(flags)
        assert runs == [(22, 5)]

    def test_merge_runs_all_true(self, client):
        app_module = _fresh_app()
        assert app_module._merge_runs([True] * 24) == [(0, 23)]

    def test_merge_runs_all_false(self, client):
        app_module = _fresh_app()
        assert app_module._merge_runs([False] * 24) == []

    def test_window_hours_expands_wrap_correctly(self, client):
        app_module = _fresh_app()
        assert app_module._window_hours(22, 5) == [22, 23, 0, 1, 2, 3, 4, 5]
        assert app_module._window_hours(8, 12) == [8, 9, 10, 11, 12]

    def test_find_windows_produces_single_wrapping_best_window(self, client):
        app_module = _fresh_app()
        low_hours = set(range(22, 24)) | set(range(0, 6))   # 22,23,0,1,2,3,4,5
        peak_hours = {17, 18, 19, 20}
        profile = [
            0.9 if h in peak_hours else (0.05 if h in low_hours else 0.4)
            for h in range(24)
        ]
        best, worst = app_module.find_windows(profile)
        wrap_best = [w for w in best if w["start_hour"] > w["end_hour"]]
        assert len(wrap_best) == 1
        w = wrap_best[0]
        assert w["start_hour"] == 22
        assert w["end_hour"] == 5
        assert w["label"] == "Free"
        # merged into one window, not eight separate one-hour windows
        assert len(best) < len(low_hours)

        assert any(w["start_hour"] <= 18 <= w["end_hour"] for w in worst)


# ── /best-time ────────────────────────────────────────────────────────────

class TestBestTime:
    def test_happy_path(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=8&latest=12")
        assert r.status_code == 200
        body = r.get_json()
        assert body["earliest"] == 8 and body["latest"] == 12
        assert 8 <= body["recommended_hour"] <= 12
        assert "summary" in body and isinstance(body["alternatives"], list)
        for alt in body["alternatives"]:
            for key in ("hour", "congestion_index", "delay_minutes"):
                assert key in alt
            assert 8 <= alt["hour"] <= 12

    def test_wrap_past_midnight(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=22&latest=5")
        assert r.status_code == 200
        body = r.get_json()
        valid_hours = set(range(22, 24)) | set(range(0, 6))
        assert body["recommended_hour"] in valid_hours
        for alt in body["alternatives"]:
            assert alt["hour"] in valid_hours

    def test_single_hour_window(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=9&latest=9")
        assert r.status_code == 200
        body = r.get_json()
        assert body["recommended_hour"] == 9
        assert body["saving_vs_worst_minutes"] == 0

    def test_bad_earliest(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=30&latest=5")
        assert r.status_code == 400

    def test_bad_latest(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=8&latest=-2")
        assert r.status_code == 400

    def test_missing_latest(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=8")
        assert r.status_code == 400

    def test_bad_corridor(self, client):
        r = client.get(f"/best-time?corridor={INVALID_CORRIDOR_ID}&day=1&earliest=8&latest=12")
        assert r.status_code == 400


class TestBestTimeWholeDayAndWindowConstraint:
    """/best-time must lead with the corridor's whole-day best-vs-worst
    saving, and explicitly flag when the requested window excludes the real
    best hour, so a small window-limited number reads as 'within your
    window' rather than 'this site barely helps'. See docs/api_contract.md
    'best-time saving fields'."""

    def test_new_fields_present(self, client):
        r = client.get("/best-time?corridor=1&day=4&earliest=7&latest=11")
        body = r.get_json()
        for key in ("saving_vs_worst_pct", "window_constrained", "whole_day_best_hour",
                    "whole_day_worst_hour", "whole_day_saving_minutes", "whole_day_saving_pct"):
            assert key in body, f"missing {key}"

    def test_whole_day_fields_match_full_scan(self, client):
        """MG Road (corridor 1), Friday (day 4): a narrow 7-11 AM window
        should report a materially smaller saving than the true whole-day
        best (midnight) vs worst (7 PM) swing, and must be flagged as
        window_constrained."""
        app_module = _fresh_app()
        r = client.get("/best-time?corridor=1&day=4&earliest=7&latest=11")
        body = r.get_json()

        day_idx = [app_module.GRID[(1, 4, h)]["congestion_index"] for h in range(24)]
        expected_best_hour = min(range(24), key=lambda h: day_idx[h])
        expected_worst_hour = max(range(24), key=lambda h: day_idx[h])
        assert body["whole_day_best_hour"] == expected_best_hour
        assert body["whole_day_worst_hour"] == expected_worst_hour

        best_cell = app_module.GRID[(1, 4, expected_best_hour)]
        worst_cell = app_module.GRID[(1, 4, expected_worst_hour)]
        expected_saving = round(worst_cell["delay_minutes"] - best_cell["delay_minutes"], 1)
        assert body["whole_day_saving_minutes"] == pytest.approx(expected_saving, abs=0.05)

        # The window (7-11 AM) excludes hour 0, so a materially bigger saving
        # exists outside it -- must be flagged, and the whole-day saving must
        # be at least as large as the window-constrained one.
        assert body["window_constrained"] is True
        assert body["whole_day_saving_minutes"] >= body["saving_vs_worst_minutes"]
        assert "Best overall" in body["summary"]

    def test_window_not_constrained_when_it_already_contains_the_best_hour(self, client):
        """A full 0-23 window can never be missing a better hour outside it."""
        r = client.get("/best-time?corridor=1&day=4&earliest=0&latest=23")
        body = r.get_json()
        assert body["window_constrained"] is False
        assert body["recommended_hour"] == body["whole_day_best_hour"]
        assert body["saving_vs_worst_minutes"] == pytest.approx(body["whole_day_saving_minutes"], abs=0.05)

    def test_saving_pct_is_consistent_with_minutes(self, client):
        r = client.get("/best-time?corridor=0&day=1&earliest=8&latest=20")
        body = r.get_json()
        if body["saving_vs_worst_minutes"] >= 0.5:
            assert body["saving_vs_worst_pct"] > 0.0

    def test_period_day_stays_within_day_hours(self, client):
        r = client.get("/best-time?corridor=0&day=1&period=day")
        assert r.status_code == 200
        body = r.get_json()
        assert body["period"] == "day"
        assert body["earliest"] == 6 and body["latest"] == 21
        assert 6 <= body["recommended_hour"] <= 21
        for alt in body["alternatives"]:
            assert 6 <= alt["hour"] <= 21
        assert body["window_constrained"] is False, (
            "an explicit period must never fall back to the whole-day "
            "comparison that made night silently win every time"
        )

    def test_period_night_stays_within_night_hours(self, client):
        r = client.get("/best-time?corridor=0&day=1&period=night")
        assert r.status_code == 200
        body = r.get_json()
        assert body["period"] == "night"
        valid_hours = set(range(22, 24)) | set(range(0, 6))
        assert body["recommended_hour"] in valid_hours
        for alt in body["alternatives"]:
            assert alt["hour"] in valid_hours

    def test_period_any_covers_whole_day(self, client):
        r = client.get("/best-time?corridor=0&day=1&period=any")
        assert r.status_code == 200
        body = r.get_json()
        assert body["period"] == "any"
        assert body["earliest"] == 0 and body["latest"] == 23
        assert body["recommended_hour"] == body["whole_day_best_hour"]

    def test_bad_period_value_400(self, client):
        r = client.get("/best-time?corridor=0&day=1&period=evening")
        assert r.status_code == 400
        assert "error" in r.get_json()

    def test_period_omitted_requires_earliest_and_latest_as_before(self, client):
        """Backwards compatibility: no period param -> legacy behavior,
        earliest/latest still required."""
        r = client.get("/best-time?corridor=0&day=1")
        assert r.status_code == 400
        r2 = client.get("/best-time?corridor=0&day=1&earliest=8&latest=12")
        assert r2.status_code == 200
        assert r2.get_json()["period"] == "custom"

    def test_period_summary_is_period_worded(self, client):
        r = client.get("/best-time?corridor=0&day=1&period=day")
        body = r.get_json()
        if body["saving_vs_worst_minutes"] >= 0.5:
            assert "daytime" in body["summary"].lower()

    def test_low_confidence_best_time_gets_caveat_text(self, client):
        app_module = _fresh_app()
        # Every real corridor/day here is measured (high confidence) per the
        # current bootstrap sweep, so directly exercise the text-building
        # logic rather than requiring a naturally low-confidence cell to
        # exist in today's data.
        assert "Limited data" not in app_module.app.test_client().get(
            "/best-time?corridor=0&day=1&earliest=8&latest=12"
        ).get_json()["summary"]


# ── /now ─────────────────────────────────────────────────────────────────

class TestNow:
    def test_happy_path(self, client):
        r = client.get("/now")
        assert r.status_code == 200
        body = r.get_json()
        assert "now_ist" in body
        assert "+05:30" in body["now_ist"]
        assert 0 <= body["day"] <= 6
        assert 0 <= body["hour"] <= 23
        assert len(body["corridors"]) == N_CORRIDORS
        for c in body["corridors"]:
            for key in ("id", "name", "congestion_index", "label",
                        "delay_minutes", "trend", "verdict", "text"):
                assert key in c
            assert c["trend"] in ("rising", "falling", "flat")
            assert c["verdict"] in ("go_now", "wait", "avoid")
        assert "summary" in body
        for key in ("avg_congestion", "worst_corridor", "clear_count"):
            assert key in body["summary"]
        assert "provenance" in body

    def test_text_frames_label_as_forecast_not_live_sensor(self, client):
        """/now text must describe a forecast, not replay a live sensor."""
        r = client.get("/now")
        body = r.get_json()
        for c in body["corridors"]:
            assert "forecast" in c["text"].lower() or "avoid" in c["text"].lower()
            assert c.get("origin") in ("residual_adjusted", "bootstrap")

    def test_now_text_helper_appends_low_confidence_caveat(self, client):
        app_module = _fresh_app()
        confident_text = app_module._now_text("Moderate", "go_now", 0.9)
        unconfident_text = app_module._now_text("Moderate", "go_now", 0.3)
        assert "Limited data" not in confident_text
        assert "Limited data" in unconfident_text


# ── no-model 503 path ────────────────────────────────────────────────────

class TestNoModel503:
    def test_model_backed_endpoints_503_when_forecast_missing(self):
        assert os.path.exists(FORECAST_MODEL_PATH), "expected forecast model present for this test"
        backup = FORECAST_MODEL_PATH + ".testbak"
        shutil.move(FORECAST_MODEL_PATH, backup)
        try:
            app_module = _fresh_app()
            app_module.app.testing = True
            with app_module.app.test_client() as c:
                for path in (
                    "/predict?corridor=0&day=1&hour=8",
                    "/advice?corridor=0&day=1",
                    "/advice/all?day=1",
                    "/best-time?corridor=0&day=1&earliest=8&latest=12",
                    "/now",
                ):
                    r = c.get(path)
                    assert r.status_code == 503, f"{path} did not 503"
                    assert r.get_json() == {"error": "no forecast model trained yet"}

                # corridors + health are not model-backed and must still work
                r_corridors = c.get("/corridors")
                assert r_corridors.status_code == 200
                assert len(r_corridors.get_json()["corridors"]) == N_CORRIDORS

                r_health = c.get("/health")
                assert r_health.status_code == 200
                assert r_health.get_json()["status"] == "ok"
        finally:
            shutil.move(backup, FORECAST_MODEL_PATH)
            _fresh_app()  # reload with the model restored for any subsequent tests
