import pytest

import trip_services


def test_photon_results_are_normalised_and_deduplicated():
    payload = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [77.0881, 28.4950]},
                "properties": {
                    "name": "Cyber Hub",
                    "district": "DLF Phase 2",
                    "city": "Gurugram",
                    "state": "Haryana",
                    "country": "India",
                    "osm_type": "W",
                    "osm_id": 123,
                    "osm_value": "commercial",
                },
            },
            {
                "geometry": {"type": "Point", "coordinates": [77.0881, 28.4950]},
                "properties": {"name": "Cyber Hub", "osm_type": "W", "osm_id": 456},
            },
        ]
    }

    results = trip_services.normalise_photon_response(payload)

    assert len(results) == 1
    assert results[0]["id"] == "W:123"
    assert results[0]["name"] == "Cyber Hub"
    assert "Gurugram" in results[0]["secondary"]
    assert results[0]["lat"] == pytest.approx(28.4950)
    assert results[0]["lon"] == pytest.approx(77.0881)
    assert results[0]["source"] == "openstreetmap"


def test_search_normalisation_splits_camel_case_place_names():
    assert trip_services.normalise_search_query("CyberHub") == "Cyber Hub"


def test_verified_catalog_fills_missing_landmarks_and_aliases():
    elevate = trip_services.search_verified_catalog("Elevate Hines")
    cyber_hub = trip_services.search_verified_catalog("CyberHub")

    assert elevate[0]["id"] == "verified:elevate-hines"
    assert elevate[0]["lat"] == pytest.approx(28.409)
    assert cyber_hub[0]["id"] == "verified:dlf-cyber-hub"
    assert all(item["source"] == "verified_catalog" for item in elevate + cyber_hub)


def test_google_autocomplete_is_normalised_without_pretending_it_has_coordinates():
    payload = {"suggestions": [{"placePrediction": {
        "placeId": "ChIJexample123",
        "text": {"text": "Mahindra Luminare, Sector 59, Gurugram, Haryana, India"},
        "structuredFormat": {
            "mainText": {"text": "Mahindra Luminare"},
            "secondaryText": {"text": "Sector 59, Gurugram, Haryana, India"},
        },
        "types": ["apartment_complex"],
    }}]}

    result = trip_services.normalise_google_autocomplete(payload)[0]

    assert result["provider_id"] == "ChIJexample123"
    assert result["needs_details"] is True
    assert "lat" not in result


def test_google_place_details_are_normalised_and_service_area_checked():
    result = trip_services.normalise_google_place({
        "id": "ChIJexample123",
        "displayName": {"text": "Mahindra Luminare"},
        "formattedAddress": "Mahindra Luminare, Sector 59, Gurugram, Haryana, India",
        "location": {"latitude": 28.413, "longitude": 77.104},
        "primaryType": "apartment_complex",
    })

    assert result["source"] == "google_places"
    assert result["lat"] == pytest.approx(28.413)
    assert result["lon"] == pytest.approx(77.104)

    address_only = trip_services.normalise_google_place({
        "id": "ChIJexample456",
        "displayName": {"text": "Mahindra Luminare"},
        "formattedAddress": "Sector 59, Gurugram, Haryana, India",
        "location": {"latitude": 28.413, "longitude": 77.104},
        "primaryType": "apartment_complex",
    })
    assert address_only["label"] == "Mahindra Luminare, Sector 59, Gurugram, Haryana, India"


def test_coordinate_validation_rejects_points_outside_service_area():
    assert trip_services.parse_coordinate("28.4595,77.0266", "origin") == (28.4595, 77.0266)
    with pytest.raises(trip_services.TripServiceError, match="outside") as exc:
        trip_services.parse_coordinate("19.0760,72.8777", "destination")
    assert exc.value.status == 400


def test_tomtom_route_is_normalised_to_geojson():
    payload = {
        "routes": [{
            "summary": {
                "lengthInMeters": 4100,
                "travelTimeInSeconds": 900,
                "trafficDelayInSeconds": 180,
                "noTrafficTravelTimeInSeconds": 720,
                "historicTrafficTravelTimeInSeconds": 840,
                "departureTime": "2026-09-23T08:00:00+05:30",
                "arrivalTime": "2026-09-23T08:15:00+05:30",
            },
            "legs": [{"points": [
                {"latitude": 28.4595, "longitude": 77.0266},
                {"latitude": 28.4723, "longitude": 77.0724},
            ]}],
        }]
    }

    route = trip_services.normalise_tomtom_response(payload)

    assert route["summary"]["distance_meters"] == 4100
    assert route["summary"]["traffic_delay_seconds"] == 180
    assert route["geometry"] == {
        "type": "LineString",
        "coordinates": [[77.0266, 28.4595], [77.0724, 28.4723]],
    }


def test_no_route_fails_loudly():
    with pytest.raises(trip_services.TripServiceError, match="No connected") as exc:
        trip_services.normalise_tomtom_response({"routes": []})
    assert exc.value.status == 422


def test_route_without_no_traffic_baseline_fails_loudly():
    payload = {
        "routes": [{
            "summary": {"lengthInMeters": 5000, "travelTimeInSeconds": 900},
            "legs": [{"points": [
                {"latitude": 28.45, "longitude": 77.03},
                {"latitude": 28.46, "longitude": 77.04},
            ]}],
        }],
    }
    with pytest.raises(trip_services.TripServiceError, match="no-traffic baseline"):
        trip_services.normalise_tomtom_response(payload)


def test_calculate_route_requests_traffic_neutral_shortest_geometry(monkeypatch):
    captured = {}

    def fake_get_json(url, params, timeout=15):
        captured.update({"url": url, "params": params, "timeout": timeout})
        return {
            "routes": [{
                "summary": {
                    "lengthInMeters": 5000,
                    "travelTimeInSeconds": 900,
                    "noTrafficTravelTimeInSeconds": 720,
                },
                "legs": [{"points": [
                    {"latitude": 28.45, "longitude": 77.03},
                    {"latitude": 28.46, "longitude": 77.04},
                ]}],
            }],
        }

    monkeypatch.setattr(trip_services, "_load_tomtom_key", lambda: "test-key")
    monkeypatch.setattr(trip_services, "_get_json", fake_get_json)
    trip_services.calculate_route(
        (28.45, 77.03),
        (28.46, 77.04),
        "2026-09-24T07:00:00+05:30",
    )

    assert captured["params"]["traffic"] == "false"
    assert captured["params"]["routeType"] == "shortest"
    assert captured["params"]["computeTravelTimeFor"] == "all"
