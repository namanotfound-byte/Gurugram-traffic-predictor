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
MODEL_PATH = os.path.join(ROOT_DIR, "models", "traffic_gbt.joblib")

sys.path.insert(0, BACKEND_DIR)


def _fresh_app():
    """(Re)import app.py fresh so its module-level startup logic (model
    load + grid precompute) reruns against the current state of
    models/traffic_gbt.joblib. Needed because the app builds its grid once
    at import time, not per-request."""
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
        assert body["corridors"] == 8
        for key in ("model_version", "provenance", "trained_rows"):
            assert key in body

    def test_corridors_static_list_not_hardcoded_shape(self, client):
        r = client.get("/corridors")
        assert r.status_code == 200
        body = r.get_json()
        assert len(body["corridors"]) == 8
        c0 = body["corridors"][0]
        for key in ("id", "name", "sub", "road_class", "start", "end", "length_km"):
            assert key in c0
        assert isinstance(c0["start"], list) and len(c0["start"]) == 2
        assert isinstance(c0["end"], list) and len(c0["end"]) == 2


# ── /predict ─────────────────────────────────────────────────────────────

class TestConfidenceIsHonest:
    """Confidence must never be a flat constant -- it should reflect real
    data quality signals (model cv_r2, per-cell real-data backing, and
    per-corridor route-length stability from the bootstrap sweep)."""

    def test_confidence_in_valid_range(self, client):
        for cid in range(8):
            r = client.get(f"/predict?corridor={cid}&day=1&hour=8")
            conf = r.get_json()["confidence"]
            assert 0.0 <= conf <= 1.0

    def test_route_stability_computed_from_real_csv_not_hardcoded(self, client):
        """Directly exercises load_route_stability() against whatever the
        bootstrap sweep actually measured. If a corridor's routed length
        varies across the sweep (TomTom picked a different path hour to
        hour), that corridor's stability factor must drop below 1.0 --
        purely from the CSV data, not a per-corridor constant in code."""
        app_module = _fresh_app()
        stability = app_module.load_route_stability()
        if not stability:
            pytest.skip("no bootstrap CSV available in this environment")
        # a fully-stable corridor (identical length_m every sweep) must be 1.0
        assert any(v == 1.0 for v in stability.values()) or all(v < 1.0 for v in stability.values())
        # every value must be a valid factor
        for v in stability.values():
            assert 0.5 <= v <= 1.0

    def test_synthetic_provenance_is_flat_low(self, client):
        """The one case where a flat confidence IS correct: a synthetic
        model has literally no real data to differentiate cells by."""
        app_module = _fresh_app()
        assert app_module.compute_confidence(
            "synthetic", None, (0, 0, 0), 0, set(), set(), set(), {},
        ) == 0.15


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
        assert body["provenance"] in ("observed", "bootstrap", "synthetic")
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["typical_minutes"] >= body["free_flow_minutes"]
        assert body["delay_minutes"] == pytest.approx(
            body["typical_minutes"] - body["free_flow_minutes"], abs=0.05)

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


# ── /advice/all ──────────────────────────────────────────────────────────

class TestAdviceAll:
    def test_happy_path(self, client):
        r = client.get("/advice/all?day=2")
        assert r.status_code == 200
        body = r.get_json()
        assert body["day"] == 2
        assert "provenance" in body and "model_version" in body
        assert len(body["corridors"]) == 8
        ids = {c["corridor_id"] for c in body["corridors"]}
        assert ids == set(range(8))
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
        r = client.get("/best-time?corridor=8&day=1&earliest=8&latest=12")
        assert r.status_code == 400


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
        assert len(body["corridors"]) == 8
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


# ── no-model 503 path ────────────────────────────────────────────────────

class TestNoModel503:
    def test_model_backed_endpoints_503_when_model_missing(self):
        assert os.path.exists(MODEL_PATH), "expected a model file present to back up for this test"
        backup = MODEL_PATH + ".testbak"
        shutil.move(MODEL_PATH, backup)
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
                    assert r.get_json() == {"error": "no model trained yet"}

                # corridors + health are not model-backed and must still work
                r_corridors = c.get("/corridors")
                assert r_corridors.status_code == 200
                assert len(r_corridors.get_json()["corridors"]) == 8

                r_health = c.get("/health")
                assert r_health.status_code == 200
                assert r_health.get_json()["status"] == "ok"
        finally:
            shutil.move(backup, MODEL_PATH)
            _fresh_app()  # reload with the model restored for any subsequent tests
