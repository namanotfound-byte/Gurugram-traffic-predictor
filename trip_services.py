"""External place-search and road-routing adapters used by the trip planner.

This module deliberately contains no Flask or Vercel-specific code.  The
local Flask API and the production serverless functions expose the same small
JSON contract around these providers:

* Google Places supplies high-coverage search-as-you-type suggestions when configured.
* Photon (OpenStreetMap data) remains the no-key fallback.
* TomTom calculates an actual connected driving route between selected points.

The existing TOMTOM_API_KEY is routing-only, so place search must not depend
on TomTom Search entitlement. GOOGLE_PLACES_API_KEY stays server-side.
PHOTON_BASE_URL is configurable so the fallback can be replaced by a hosted
Photon instance without changing the frontend.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parent
PLACE_CATALOG_PATH = REPO_ROOT / "data" / "place_catalog.json"
PHOTON_BASE_URL = os.environ.get("PHOTON_BASE_URL", "https://photon.komoot.io").rstrip("/")
GURUGRAM_BBOX = (76.65, 28.00, 77.35, 28.85)  # min lon, min lat, max lon, max lat
SEARCH_FOCUS = (28.4595, 77.0266)
USER_AGENT = "gurugram-traffic-predictor/1.0 (place search and trip routing)"

with PLACE_CATALOG_PATH.open(encoding="utf-8") as catalog_file:
    VERIFIED_PLACE_CATALOG = json.load(catalog_file)


class TripServiceError(RuntimeError):
    """Safe provider error carrying an HTTP status suitable for the client."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _load_env_value(variable_name: str) -> Optional[str]:
    key = os.environ.get(variable_name)
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key_name, value = line.split("=", 1)
            if key_name.strip() == variable_name:
                return value.strip().strip('"').strip("'") or None
    return None


def _load_tomtom_key() -> Optional[str]:
    return _load_env_value("TOMTOM_API_KEY")


def _get_json(url: str, params: dict, timeout: int = 15) -> dict:
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        full_url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Never include the requested URL here because it contains the key.
        if exc.code == 429:
            raise TripServiceError("The map provider is busy. Please try again shortly.", 429) from exc
        if exc.code in (401, 403):
            raise TripServiceError("Trip routing is not configured for this deployment.", 503) from exc
        if exc.code == 404:
            raise TripServiceError("No connected driving route was found.", 422) from exc
        raise TripServiceError(f"The map provider returned HTTP {exc.code}.", 502) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TripServiceError("The map provider could not be reached. Please try again.", 502) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise TripServiceError("The map provider returned an unreadable response.", 502) from exc


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 10) -> dict:
    request_headers = {"Accept": "application/json", "Content-Type": "application/json", **headers}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise TripServiceError("Place search is busy. Please try again shortly.", 429) from exc
        if exc.code in (401, 403):
            raise TripServiceError("High-precision place search is not configured for this deployment.", 503) from exc
        raise TripServiceError(f"Place search returned HTTP {exc.code}.", 502) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TripServiceError("Place search could not be reached. Please try again.", 502) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise TripServiceError("Place search returned an unreadable response.", 502) from exc


def _place_context(properties: dict) -> list[str]:
    candidates = [
        properties.get("housenumber"),
        properties.get("street"),
        properties.get("district"),
        properties.get("city"),
        properties.get("county"),
        properties.get("state"),
        properties.get("country"),
    ]
    name = str(properties.get("name") or "").strip().casefold()
    seen = {name} if name else set()
    parts = []
    for value in candidates:
        value = str(value or "").strip()
        folded = value.casefold()
        if value and folded not in seen:
            seen.add(folded)
            parts.append(value)
    return parts


def normalise_photon_response(data: dict, limit: int = 6) -> list[dict]:
    """Convert Photon GeoJSON into the compact contract used by the UI."""
    results = []
    seen = set()
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        props = feature.get("properties") or {}
        if geometry.get("type") != "Point" or len(coords) < 2:
            continue
        try:
            lon, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        name = str(props.get("name") or props.get("street") or "").strip()
        if not name:
            continue
        osm_type = str(props.get("osm_type") or "place")
        osm_id = str(props.get("osm_id") or f"{lat:.6f},{lon:.6f}")
        place_id = f"{osm_type}:{osm_id}"
        dedupe_key = (name.casefold(), round(lat, 5), round(lon, 5))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        context = _place_context(props)
        results.append({
            "id": place_id,
            "name": name,
            "label": ", ".join([name] + context),
            "secondary": ", ".join(context),
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "type": str(props.get("type") or props.get("osm_value") or "place"),
            "source": "openstreetmap",
        })
        if len(results) >= limit:
            break
    return results


def normalise_search_query(query: str) -> str:
    """Make common user spellings such as ``CyberHub`` searchable."""
    query = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(query or ""))
    query = re.sub(r"[^\w\s-]", " ", query, flags=re.UNICODE)
    return " ".join(query.split())


def search_verified_catalog(query: str, limit: int = 6) -> list[dict]:
    normalised = normalise_search_query(query).casefold()
    tokens = normalised.split()
    if not tokens:
        return []
    matches = []
    for place in VERIFIED_PLACE_CATALOG:
        aliases = [place["name"], place["label"], *place.get("aliases", [])]
        folded_aliases = [normalise_search_query(value).casefold() for value in aliases]
        haystack = " ".join(folded_aliases)
        if not all(token in haystack for token in tokens):
            continue
        if normalised in folded_aliases:
            rank = 0
        elif any(alias.startswith(normalised) for alias in folded_aliases):
            rank = 1
        else:
            rank = 2
        result = {key: place[key] for key in (
            "id", "name", "label", "secondary", "lat", "lon", "type"
        )}
        result["source"] = "verified_catalog"
        matches.append((rank, result))
    matches.sort(key=lambda item: (item[0], item[1]["name"]))
    return [item[1] for item in matches[:limit]]


def merge_place_results(primary: list[dict], secondary: list[dict], limit: int = 6) -> list[dict]:
    merged = []
    seen = set()
    for place in [*primary, *secondary]:
        key = (place["name"].casefold(), round(place["lat"], 5), round(place["lon"], 5))
        if key in seen:
            continue
        seen.add(key)
        merged.append(place)
        if len(merged) >= limit:
            break
    return merged


def normalise_google_autocomplete(data: dict, limit: int = 6) -> list[dict]:
    results = []
    for suggestion in data.get("suggestions", []):
        prediction = suggestion.get("placePrediction") or {}
        place_id = str(prediction.get("placeId") or "").strip()
        structured = prediction.get("structuredFormat") or {}
        name = str((structured.get("mainText") or {}).get("text") or "").strip()
        secondary = str((structured.get("secondaryText") or {}).get("text") or "").strip()
        label = str((prediction.get("text") or {}).get("text") or ", ".join(filter(None, [name, secondary]))).strip()
        if not place_id or not name or not label:
            continue
        types = prediction.get("types") or []
        results.append({
            "id": f"google:{place_id}",
            "provider_id": place_id,
            "name": name,
            "label": label,
            "secondary": secondary,
            "type": str(types[0] if types else "place"),
            "source": "google_places",
            "needs_details": True,
        })
        if len(results) >= limit:
            break
    return results


def _valid_session_token(value: str) -> Optional[str]:
    value = str(value or "").strip()
    return value if re.fullmatch(r"[A-Za-z0-9_-]{16,36}", value) else None


def _search_google_places(query: str, session_token: str, limit: int) -> list[dict]:
    key = _load_env_value("GOOGLE_PLACES_API_KEY")
    if not key:
        return []
    min_lon, min_lat, max_lon, max_lat = GURUGRAM_BBOX
    token = _valid_session_token(session_token)
    payload = {
        "input": query,
        "includedRegionCodes": ["in"],
        "languageCode": "en",
        "regionCode": "IN",
        "origin": {"latitude": SEARCH_FOCUS[0], "longitude": SEARCH_FOCUS[1]},
        "locationRestriction": {
            "rectangle": {
                "low": {"latitude": min_lat, "longitude": min_lon},
                "high": {"latitude": max_lat, "longitude": max_lon},
            }
        },
    }
    if token:
        payload["sessionToken"] = token
    data = _post_json(
        "https://places.googleapis.com/v1/places:autocomplete",
        payload,
        {"X-Goog-Api-Key": key},
        timeout=10,
    )
    return normalise_google_autocomplete(data, limit=limit)


def normalise_google_place(data: dict) -> dict:
    location = data.get("location") or {}
    try:
        lat, lon = float(location["latitude"]), float(location["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TripServiceError("The selected place did not include a usable location.", 502) from exc
    parse_coordinate(f"{lat},{lon}", "place")
    name = str((data.get("displayName") or {}).get("text") or "").strip()
    address = str(data.get("formattedAddress") or "").strip()
    if not name:
        raise TripServiceError("The selected place did not include a usable address.", 502)
    address_includes_name = address.casefold().startswith(name.casefold())
    label = address if address_includes_name else ", ".join(filter(None, [name, address]))
    secondary = address[len(name):].lstrip(", ") if address_includes_name else address
    return {
        "id": f"google:{data.get('id')}",
        "name": name,
        "label": label,
        "secondary": secondary,
        "lat": lat,
        "lon": lon,
        "type": str(data.get("primaryType") or "place"),
        "source": "google_places",
    }


def resolve_google_place(place_id: str, session_token: str = "") -> dict:
    key = _load_env_value("GOOGLE_PLACES_API_KEY")
    if not key:
        raise TripServiceError("High-precision place search is not configured for this deployment.", 503)
    place_id = str(place_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,256}", place_id):
        raise TripServiceError("Invalid place identifier.", 400)
    params = {"languageCode": "en", "regionCode": "IN"}
    token = _valid_session_token(session_token)
    if token:
        params["sessionToken"] = token
    url = "https://places.googleapis.com/v1/places/" + urllib.parse.quote(place_id, safe="")
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, headers={
        "Accept": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "id,displayName,formattedAddress,location,primaryType",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise TripServiceError("High-precision place search is not configured for this deployment.", 503) from exc
        if exc.code == 404:
            raise TripServiceError("The selected place is no longer available.", 404) from exc
        raise TripServiceError(f"Place details returned HTTP {exc.code}.", 502) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TripServiceError("Place details could not be reached. Please try again.", 502) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise TripServiceError("Place details returned an unreadable response.", 502) from exc
    return normalise_google_place(data)


def search_places(query: str, limit: int = 6, session_token: str = "") -> list[dict]:
    query = " ".join(str(query or "").split())
    if len(query) < 2:
        raise TripServiceError("Type at least two characters to search.", 400)
    if len(query) > 120:
        raise TripServiceError("Location searches must be 120 characters or fewer.", 400)
    limit = max(1, min(int(limit), 8))
    try:
        google = _search_google_places(normalise_search_query(query), session_token, limit)
    except TripServiceError:
        google = []
    if google:
        return google
    verified = search_verified_catalog(query, limit=limit)
    provider_query = normalise_search_query(query)
    min_lon, min_lat, max_lon, max_lat = GURUGRAM_BBOX
    try:
        data = _get_json(
            f"{PHOTON_BASE_URL}/api",
            {
                "q": provider_query,
                "limit": limit,
                "lang": "en",
                "lat": SEARCH_FOCUS[0],
                "lon": SEARCH_FOCUS[1],
                "zoom": 10,
                "location_bias_scale": 0.15,
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            },
            timeout=10,
        )
    except TripServiceError:
        if verified:
            return verified
        raise
    photon = normalise_photon_response(data, limit=limit)
    return merge_place_results(verified, photon, limit=limit)


def parse_coordinate(raw: str, field_name: str) -> Tuple[float, float]:
    try:
        lat_s, lon_s = str(raw).split(",", 1)
        lat, lon = float(lat_s), float(lon_s)
    except (TypeError, ValueError) as exc:
        raise TripServiceError(f"'{field_name}' must be 'latitude,longitude'.", 400) from exc
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise TripServiceError(f"'{field_name}' contains an invalid coordinate.", 400)
    min_lon, min_lat, max_lon, max_lat = GURUGRAM_BBOX
    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
        raise TripServiceError(f"'{field_name}' is outside the Gurugram/Delhi NCR service area.", 400)
    return lat, lon


def validate_depart_at(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise TripServiceError("'departAt' is required.", 400)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TripServiceError("'departAt' must be an ISO-8601 date and time.", 400) from exc
    if parsed.tzinfo is None:
        raise TripServiceError("'departAt' must include a timezone offset.", 400)
    return value


def normalise_tomtom_response(data: dict) -> dict:
    routes = data.get("routes") or []
    if not routes:
        raise TripServiceError("No connected driving route was found.", 422)
    route = routes[0]
    summary = route.get("summary") or {}
    coordinates = []
    for leg in route.get("legs") or []:
        for point in leg.get("points") or []:
            try:
                coord = [float(point["longitude"]), float(point["latitude"])]
            except (KeyError, TypeError, ValueError):
                continue
            if not coordinates or coordinates[-1] != coord:
                coordinates.append(coord)
    if len(coordinates) < 2:
        raise TripServiceError("The route provider did not return road geometry.", 502)
    travel_seconds = int(summary.get("travelTimeInSeconds") or 0)
    no_traffic_seconds = int(summary.get("noTrafficTravelTimeInSeconds") or 0)
    if no_traffic_seconds <= 0:
        raise TripServiceError(
            "The route provider did not return the no-traffic baseline "
            "required by the prediction model.",
            502,
        )
    historic_seconds = int(summary.get("historicTrafficTravelTimeInSeconds") or travel_seconds)
    traffic_delay = int(summary.get("trafficDelayInSeconds") or max(0, travel_seconds - no_traffic_seconds))
    return {
        "provider": "tomtom",
        "summary": {
            "distance_meters": int(summary.get("lengthInMeters") or 0),
            "travel_time_seconds": travel_seconds,
            "traffic_delay_seconds": max(0, traffic_delay),
            "no_traffic_time_seconds": no_traffic_seconds,
            "historic_traffic_time_seconds": historic_seconds,
            "departure_time": summary.get("departureTime"),
            "arrival_time": summary.get("arrivalTime"),
        },
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def calculate_route(origin: Tuple[float, float], destination: Tuple[float, float], depart_at: str) -> dict:
    key = _load_tomtom_key()
    if not key:
        raise TripServiceError("Trip routing is not configured for this deployment.", 503)
    depart_at = validate_depart_at(depart_at)
    path = f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}"
    data = _get_json(
        f"https://api.tomtom.com/routing/1/calculateRoute/{path}/json",
        {
            "key": key,
            "departAt": depart_at,
            # Keep the connected geometry independent of TomTom's traffic
            # forecast. Our residual model compares the departure hours.
            "traffic": "false",
            "computeTravelTimeFor": "all",
            "routeRepresentation": "polyline",
            "routeType": "shortest",
            "travelMode": "car",
        },
        timeout=20,
    )
    return normalise_tomtom_response(data)
