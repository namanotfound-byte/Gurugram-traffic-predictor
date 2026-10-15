import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ApiError,
  calculateRoute,
  mergePlaceResults,
  normaliseGoogleAutocomplete,
  normaliseGooglePlace,
  normalisePhoton,
  normaliseSearchQuery,
  normaliseTomTom,
  parseCoordinate,
  searchVerifiedCatalog,
  validateDepartAt,
} from '../api/_trip_lib.mjs'

test('normalisePhoton returns stable selected-place data', () => {
  const results = normalisePhoton({ features: [{
    geometry: { type: 'Point', coordinates: [77.0881, 28.495] },
    properties: {
      name: 'Cyber Hub', district: 'DLF Phase 2', city: 'Gurugram',
      state: 'Haryana', country: 'India', osm_type: 'W', osm_id: 123,
    },
  }] })
  assert.equal(results.length, 1)
  assert.equal(results[0].id, 'W:123')
  assert.match(results[0].secondary, /Gurugram/)
  assert.equal(results[0].source, 'openstreetmap')
})

test('camel-case searches and verified Gurugram aliases resolve', () => {
  assert.equal(normaliseSearchQuery('CyberHub'), 'Cyber Hub')
  assert.equal(searchVerifiedCatalog('CyberHub')[0].id, 'verified:dlf-cyber-hub')
  assert.equal(searchVerifiedCatalog('Elevate Hines')[0].id, 'verified:elevate-hines')
})

test('verified results are ranked before provider results without duplicates', () => {
  const verified = searchVerifiedCatalog('CyberHub')
  const merged = mergePlaceResults(verified, [{
    id: 'W:1', name: 'DLF Cyber Hub', label: 'DLF Cyber Hub', secondary: '',
    lat: 28.4951011, lon: 77.0885127, type: 'commercial', source: 'openstreetmap',
  }])
  assert.equal(merged.length, 1)
  assert.equal(merged[0].source, 'verified_catalog')
})

test('Google predictions resolve in a separate details step', () => {
  const prediction = normaliseGoogleAutocomplete({ suggestions: [{ placePrediction: {
    placeId: 'ChIJexample123',
    text: { text: 'Mahindra Luminare, Sector 59, Gurugram, Haryana, India' },
    structuredFormat: {
      mainText: { text: 'Mahindra Luminare' },
      secondaryText: { text: 'Sector 59, Gurugram, Haryana, India' },
    },
    types: ['apartment_complex'],
  } }] })[0]
  assert.equal(prediction.needs_details, true)
  assert.equal(prediction.provider_id, 'ChIJexample123')
  assert.equal(prediction.lat, undefined)

  const selected = normaliseGooglePlace({
    id: 'ChIJexample123',
    displayName: { text: 'Mahindra Luminare' },
    formattedAddress: 'Mahindra Luminare, Sector 59, Gurugram, Haryana, India',
    location: { latitude: 28.413, longitude: 77.104 },
    primaryType: 'apartment_complex',
  })
  assert.equal(selected.source, 'google_places')
  assert.equal(selected.lat, 28.413)

  const addressOnly = normaliseGooglePlace({
    id: 'ChIJexample456',
    displayName: { text: 'Mahindra Luminare' },
    formattedAddress: 'Sector 59, Gurugram, Haryana, India',
    location: { latitude: 28.413, longitude: 77.104 },
    primaryType: 'apartment_complex',
  })
  assert.equal(addressOnly.label, 'Mahindra Luminare, Sector 59, Gurugram, Haryana, India')
})

test('coordinates are limited to the supported NCR area', () => {
  assert.deepEqual(parseCoordinate('28.4595,77.0266', 'origin'), { lat: 28.4595, lon: 77.0266 })
  assert.throws(() => parseCoordinate('19.0760,72.8777', 'destination'), ApiError)
})

test('departAt requires an explicit timezone', () => {
  assert.equal(validateDepartAt('2026-09-23T08:00:00+05:30'), '2026-09-23T08:00:00+05:30')
  assert.throws(() => validateDepartAt('2026-09-23T08:00:00'), ApiError)
})

test('normaliseTomTom returns GeoJSON and traffic timing', () => {
  const route = normaliseTomTom({ routes: [{
    summary: {
      lengthInMeters: 4100, travelTimeInSeconds: 900,
      trafficDelayInSeconds: 180, noTrafficTravelTimeInSeconds: 720,
    },
    legs: [{ points: [
      { latitude: 28.4595, longitude: 77.0266 },
      { latitude: 28.4723, longitude: 77.0724 },
    ] }],
  }] })
  assert.equal(route.summary.distance_meters, 4100)
  assert.equal(route.summary.traffic_delay_seconds, 180)
  assert.deepEqual(route.geometry.coordinates[0], [77.0266, 28.4595])
})

test('normaliseTomTom rejects a route without a no-traffic model baseline', () => {
  assert.throws(() => normaliseTomTom({ routes: [{
    summary: { lengthInMeters: 5000, travelTimeInSeconds: 900 },
    legs: [{ points: [
      { latitude: 28.45, longitude: 77.03 },
      { latitude: 28.46, longitude: 77.04 },
    ] }],
  }] }), /no-traffic baseline/)
})

test('calculateRoute requests one traffic-neutral shortest geometry', async () => {
  const previousKey = process.env.TOMTOM_API_KEY
  const previousFetch = globalThis.fetch
  let requestedUrl
  process.env.TOMTOM_API_KEY = 'test-key'
  globalThis.fetch = async url => {
    requestedUrl = new URL(String(url))
    return {
      ok: true,
      status: 200,
      json: async () => ({ routes: [{
        summary: {
          lengthInMeters: 5000,
          travelTimeInSeconds: 900,
          noTrafficTravelTimeInSeconds: 720,
        },
        legs: [{ points: [
          { latitude: 28.45, longitude: 77.03 },
          { latitude: 28.46, longitude: 77.04 },
        ] }],
      }] }),
    }
  }
  try {
    await calculateRoute(
      { lat: 28.45, lon: 77.03 },
      { lat: 28.46, lon: 77.04 },
      '2026-09-24T07:00:00+05:30',
    )
    assert.equal(requestedUrl.searchParams.get('traffic'), 'false')
    assert.equal(requestedUrl.searchParams.get('routeType'), 'shortest')
    assert.equal(requestedUrl.searchParams.get('computeTravelTimeFor'), 'all')
  } finally {
    globalThis.fetch = previousFetch
    if (previousKey === undefined) delete process.env.TOMTOM_API_KEY
    else process.env.TOMTOM_API_KEY = previousKey
  }
})
