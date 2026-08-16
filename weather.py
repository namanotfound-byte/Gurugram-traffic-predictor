#!/usr/bin/env python3
"""
weather.py — Open-Meteo weather + Indian holiday/calendar features for Gurugram.
==================================================================================
WHY THIS FILE EXISTS
---------------------
The bootstrap grid (data/gurugram_bootstrap.csv) is a pure day-of-week x hour
average with ZERO weather/date signal — verified by querying TomTom's
historical model for six different future Fridays at 18:00, including Diwali
week, and getting byte-identical results every time. It cannot teach a model
anything about rain, festivals, or salary-day traffic.

The forecasting model (model/forecast_model.py) is built on the hypothesis
that the *deviation* from that baseline — the residual — is explained by
conditions the baseline doesn't see: weather and the calendar. This file is
the sole source of those two feature families.

WEATHER SOURCE: Open-Meteo. Free, no API key, verified reachable from this
environment. Two endpoints are used:

    Forecast  https://api.open-meteo.com/v1/forecast   (today + future, AND
              recent past via the `past_days` parameter, up to 92 days back)
    Archive   https://archive-api.open-meteo.com/v1/archive  (any historical
              date range, no upper day limit other than "not today")

ENDPOINT-SELECTION FINDING (discovered empirically, not assumed — see the
commands run during development; both endpoints were hit live against
Gurugram's coordinates before writing this file):

    The archive endpoint does NOT carry a `visibility` field at all — every
    value comes back `null` (confirmed for multiple August 2026 dates). The
    forecast endpoint DOES carry real visibility, including for past dates
    requested via `past_days` (confirmed: values in the 8000-10000 m range
    for the last few days). So for anything within the forecast endpoint's
    ~92-day trailing window, `_fetch_forecast` is preferred over
    `_fetch_archive` even for past dates — it is a strict upgrade (same
    precip/temp data, plus real visibility). True `_fetch_archive` is only
    used as a fallback once a date falls outside that 92-day window, and
    visibility is honestly left as unavailable (None) for those rows — it is
    NOT imputed with a fake value, because a silently-invented "looks fine"
    visibility would quietly corrupt the low_visibility feature.

    In practice, because this project's observed data collection only started
    2026-08-16, every row this project will ever backfill falls inside that
    92-day window, so the true-archive branch is exercised only if the repo
    is revived long after a big gap — it exists for correctness, not because
    it is expected to run often.

DERIVED FEATURES (not just raw values — see get_hourly_weather):
    precipitation_mm, is_raining, rain_intensity (none/light/moderate/heavy),
    rain_last_3h (trailing 3-hour cumulative precip — roads stay slick/slow
    after rain stops, so this is hypothesized to be a STRONGER predictor of
    congestion than the instantaneous precipitation_mm; test this claim
    against feature importances once the model trains — see
    model/forecast_model.py's report), visibility_m, low_visibility,
    temperature_c.

CALENDAR FEATURES: get_event_features() uses the `holidays` package
(`holidays.India(subdiv="HR")` — Haryana, since Gurugram is in Haryana) for
is_holiday / holiday_name / days_to_nearest_holiday, plus two features that
are NOT in that package because they aren't public holidays but are real
Gurugram traffic effects worth testing: is_festival_period (the +/- 2 days
around a holiday, when travel/shopping traffic swells beyond the holiday
date itself) and is_month_end (salary-day traffic: most Indian salaries land
on the 1st or the last 1-3 days of the month).

CACHING: every (date, hour) pair fetched is written to data/weather_cache.json
(git-ignored — it's a rebuildable cache, not a data asset) so the same hour
is never re-requested. Fetches are done in whole-day-or-wider batches, not
per hour, to keep request counts low.
"""

import calendar
import datetime
import json
import os

import requests

try:
    import holidays as holidays_lib
except ImportError:  # pragma: no cover - exercised only if the dep is missing
    holidays_lib = None


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(REPO_ROOT, "data", "weather_cache.json")

# Gurugram city-centre coordinates (same point used for every corridor — this
# is a single-city weather feed, not per-corridor; Gurugram's corridors span
# ~25 km at most, well inside one weather cell).
GURUGRAM_LAT = 28.4595
GURUGRAM_LON = 77.0266
TIMEZONE_PARAM = "Asia/Kolkata"
IST_OFFSET = datetime.timedelta(hours=5, minutes=30)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,precipitation,visibility,weather_code"

# Open-Meteo's documented trailing-history window for the forecast endpoint's
# `past_days` parameter. Beyond this, fall back to the true archive endpoint
# (see module docstring: that path loses the visibility field).
FORECAST_PAST_DAYS_MAX = 92
FORECAST_DAYS_MAX = 16  # Open-Meteo's documented forward cap

REQUEST_TIMEOUT_S = 15

# Rain thresholds. Hourly precipitation in mm, treated as an mm/hr-equivalent
# rate since our resolution is 1 hour. Bands follow common meteorological
# convention (light/moderate/heavy rain cutoffs).
RAIN_THRESHOLD_MM = 0.1       # > this counts as "raining" this hour
RAIN_LIGHT_MAX_MM = 2.5
RAIN_MODERATE_MAX_MM = 7.6

# Visibility below this (metres) is flagged low_visibility. 3000 m is a
# conservative "haze/mist/heavy-rain" cutoff (dense-fog convention is closer
# to 1000 m, but NCR haze regularly sits in the 2-3 km band and is still
# enough to change driving behaviour).
LOW_VISIBILITY_M = 3000

# +/- days around a public holiday still counted as "festival period" —
# pre-festival shopping and post-festival return-travel traffic in Gurugram
# is anecdotally a multi-day effect, not a single-day spike.
FESTIVAL_WINDOW_DAYS = 2


# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────
_cache = None  # lazy-loaded in-process dict mirroring CACHE_FILE


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                _cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_cache, f, sort_keys=True)
    os.replace(tmp, CACHE_FILE)  # atomic — a crash mid-write can't corrupt the cache


def _cache_key(date, hour):
    return f"{date.isoformat()}T{hour:02d}"


def _today_ist():
    """'Today' in Asia/Kolkata, independent of the host machine's local timezone
    (CI runners are UTC) — matters because Open-Meteo's date params are
    interpreted in the `timezone` query param we pass (Asia/Kolkata)."""
    return (datetime.datetime.utcnow() + IST_OFFSET).date()


def _store_hourly_response(payload):
    hourly = payload.get("hourly")
    if not hourly:
        return
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [None] * len(times))
    precs = hourly.get("precipitation", [None] * len(times))
    vis = hourly.get("visibility", [None] * len(times))
    codes = hourly.get("weather_code", [None] * len(times))

    cache = _load_cache()
    for i, t in enumerate(times):
        # t looks like "2026-08-17T14:00" -> bucket key "2026-08-17T14"
        cache[t[:13]] = {
            "temperature_2m": temps[i] if i < len(temps) else None,
            "precipitation": precs[i] if i < len(precs) else None,
            "visibility": vis[i] if i < len(vis) else None,
            "weather_code": codes[i] if i < len(codes) else None,
        }
    _save_cache()


def _fetch_forecast(past_days, forecast_days):
    params = {
        "latitude": GURUGRAM_LAT,
        "longitude": GURUGRAM_LON,
        "hourly": HOURLY_VARS,
        "past_days": max(0, min(FORECAST_PAST_DAYS_MAX, past_days)),
        "forecast_days": max(1, min(FORECAST_DAYS_MAX, forecast_days)),
        "timezone": TIMEZONE_PARAM,
    }
    r = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    _store_hourly_response(r.json())


def _fetch_archive(start_date, end_date):
    """True historical archive. NOTE (see module docstring): visibility comes
    back null on this endpoint — confirmed empirically, not assumed."""
    params = {
        "latitude": GURUGRAM_LAT,
        "longitude": GURUGRAM_LON,
        "hourly": HOURLY_VARS,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": TIMEZONE_PARAM,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    _store_hourly_response(r.json())


def _ensure_covered(date, hour):
    """Fetch whatever bulk range covers (date, hour) if it isn't cached yet."""
    key = _cache_key(date, hour)
    if key in _load_cache():
        return

    today = _today_ist()
    days_since = (today - date).days

    if days_since > FORECAST_PAST_DAYS_MAX:
        # Older than the forecast endpoint's trailing window -> true archive.
        _fetch_archive(date, date)
    else:
        past_days = max(0, days_since)
        forecast_days = max(1, (date - today).days + 1) if date >= today else 1
        _fetch_forecast(past_days=past_days, forecast_days=forecast_days)


# ─────────────────────────────────────────────
# DERIVED WEATHER FEATURES
# ─────────────────────────────────────────────
def _derive(raw, prev_precip):
    """raw: cached dict for the target hour (or None if unavailable).
    prev_precip: list of precipitation values for the 1-2 preceding hours
    (whatever was available), used for the rain_last_3h rollup."""
    precip = raw.get("precipitation") if raw else None
    precip = precip if precip is not None else 0.0
    vis = raw.get("visibility") if raw else None
    temp = raw.get("temperature_2m") if raw else None
    code = raw.get("weather_code") if raw else None

    is_raining = precip > RAIN_THRESHOLD_MM
    if precip <= RAIN_THRESHOLD_MM:
        intensity = "none"
    elif precip <= RAIN_LIGHT_MAX_MM:
        intensity = "light"
    elif precip <= RAIN_MODERATE_MAX_MM:
        intensity = "moderate"
    else:
        intensity = "heavy"

    # Trailing 3-hour window INCLUDING the current hour (current + 2 prior).
    rain_last_3h = round(precip + sum(p for p in prev_precip if p is not None), 2)

    low_visibility = bool(vis is not None and vis < LOW_VISIBILITY_M)

    return {
        "temperature_c": temp,
        "precipitation_mm": round(precip, 2),
        "is_raining": is_raining,
        "rain_intensity": intensity,
        "rain_last_3h": rain_last_3h,
        "visibility_m": vis,          # may be None for true-archive-sourced rows
        "low_visibility": low_visibility,
        "weather_code": code,
    }


def get_hourly_weather(date, hour):
    """Weather + derived features for one Gurugram hour.

    date: datetime.date or 'YYYY-MM-DD' string. hour: int 0-23.
    Returns a dict (see _derive) or None if this hour truly has no data
    available from either endpoint (e.g. a network failure) — callers must
    handle None rather than assume weather is always present.
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    _ensure_covered(date, hour)
    cache = _load_cache()
    raw = cache.get(_cache_key(date, hour))
    if raw is None:
        return None

    # Trailing 2 prior hours for the 3h rollup.
    prev_precip = []
    cursor_date, cursor_hour = date, hour
    for _ in range(2):
        cursor_hour -= 1
        if cursor_hour < 0:
            cursor_hour = 23
            cursor_date = cursor_date - datetime.timedelta(days=1)
        _ensure_covered(cursor_date, cursor_hour)
        prow = _load_cache().get(_cache_key(cursor_date, cursor_hour))
        prev_precip.append(prow.get("precipitation") if prow else None)

    return _derive(raw, prev_precip)


def get_weather_range(start_date, end_date):
    """Bulk variant of get_hourly_weather — prefetches the whole
    [start_date, end_date] span in as few HTTP requests as possible (instead
    of one request per hour), then returns {(date, hour): feature_dict} for
    every hour in the range. Used by collect_live.py's backfill pass and by
    forecast_model.py when building the training table over many rows.
    """
    if isinstance(start_date, str):
        start_date = datetime.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.date.fromisoformat(end_date)

    today = _today_ist()
    # 1-day lookback margin so rain_last_3h at start_date 00:00-02:00 doesn't
    # need a second fetch for the previous day's tail hours.
    lookback_start = start_date - datetime.timedelta(days=1)
    days_since_lookback = (today - lookback_start).days

    if days_since_lookback > FORECAST_PAST_DAYS_MAX:
        archive_end = min(end_date, today - datetime.timedelta(days=1))
        if lookback_start <= archive_end:
            _fetch_archive(lookback_start, archive_end)
        if end_date >= today:
            _fetch_forecast(past_days=0, forecast_days=(end_date - today).days + 1)
    else:
        past_days = max(0, days_since_lookback)
        forecast_days = max(1, (end_date - today).days + 1) if end_date >= today else 1
        _fetch_forecast(past_days=past_days, forecast_days=forecast_days)

    out = {}
    d = start_date
    while d <= end_date:
        for h in range(24):
            out[(d, h)] = get_hourly_weather(d, h)
        d += datetime.timedelta(days=1)
    return out


# ─────────────────────────────────────────────
# CALENDAR / EVENT FEATURES
# ─────────────────────────────────────────────
_holiday_cache = {}  # year -> holidays.HolidayBase


def _holidays_for_year(year):
    if holidays_lib is None:
        return None
    if year not in _holiday_cache:
        # +/-1 year so days_to_nearest_holiday near Jan 1 / Dec 31 still finds
        # its true nearest neighbour instead of falling off the edge of the
        # queried year.
        _holiday_cache[year] = holidays_lib.India(
            years=[year - 1, year, year + 1], subdiv="HR"
        )
    return _holiday_cache[year]


def _is_month_end(date):
    """Salary-day traffic: most Indian salaries land on the 1st or in the
    last 1-3 days of the month, both of which drive discretionary/shopping
    trips. Covers day<=2 (post-credit spending) and the last 3 days of the
    month (pre-credit / month-end errands)."""
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.day <= 2 or date.day >= last_day - 2


def get_event_features(date):
    """Calendar features for one date. Returns is_holiday, holiday_name,
    is_festival_period, is_month_end, days_to_nearest_holiday."""
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    hset = _holidays_for_year(date.year)
    if hset is None:  # `holidays` package not installed
        return {
            "is_holiday": False,
            "holiday_name": None,
            "is_festival_period": False,
            "is_month_end": _is_month_end(date),
            "days_to_nearest_holiday": None,
        }

    is_holiday = date in hset
    holiday_name = hset.get(date)
    nearest = min((abs((date - d).days) for d in hset), default=None)
    is_festival_period = nearest is not None and nearest <= FESTIVAL_WINDOW_DAYS

    return {
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        "is_festival_period": is_festival_period,
        "is_month_end": _is_month_end(date),
        "days_to_nearest_holiday": nearest,
    }


if __name__ == "__main__":
    # Quick manual smoke test: python weather.py
    today = _today_ist()
    print(f"Today (IST): {today}")
    w = get_hourly_weather(today, 18)
    print("Weather @18:00 today:", w)
    e = get_event_features(today)
    print("Event features today:", e)
