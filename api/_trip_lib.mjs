import verifiedPlaceCatalog from '../data/place_catalog.json' with { type: 'json' }

const PHOTON_BASE_URL = (process.env.PHOTON_BASE_URL || 'https://photon.komoot.io').replace(/\/$/, '')
const BBOX = { minLon: 76.65, minLat: 28.00, maxLon: 77.35, maxLat: 28.85 }

export class ApiError extends Error {
  constructor(message, status = 502) {
    super(message)
    this.status = status
  }
}

export function jsonResponse(body, status = 200, headers = {}) {
  return Response.json(body, {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      ...headers,
    },
  })
}

export function errorResponse(error) {
  const status = Number.isInteger(error?.status) ? error.status : 500
  const message = status >= 500 && !error?.status
    ? 'The trip service encountered an unexpected error.'
    : String(error?.message || 'Unknown error')
  return jsonResponse({ error: message }, status, { 'Cache-Control': 'no-store' })
}

function contextParts(properties) {
  const candidates = [
    properties.housenumber,
    properties.street,
    properties.district,
    properties.city,
    properties.county,
    properties.state,
    properties.country,
  ]
  const name = String(properties.name || '').trim().toLocaleLowerCase()
  const seen = new Set(name ? [name] : [])
  const parts = []
  for (const candidate of candidates) {
    const value = String(candidate || '').trim()
    const folded = value.toLocaleLowerCase()
    if (value && !seen.has(folded)) {
      seen.add(folded)
      parts.push(value)
    }
  }
  return parts
}

export function normalisePhoton(data, limit = 6) {
  const results = []
  const seen = new Set()
  for (const feature of data?.features || []) {
    const geometry = feature?.geometry || {}
    const coordinates = geometry.coordinates || []
    const properties = feature?.properties || {}
    if (geometry.type !== 'Point' || coordinates.length < 2) continue
    const lon = Number(coordinates[0])
    const lat = Number(coordinates[1])
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue
    const name = String(properties.name || properties.street || '').trim()
    if (!name) continue
    const dedupeKey = `${name.toLocaleLowerCase()}|${lat.toFixed(5)}|${lon.toFixed(5)}`
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    const context = contextParts(properties)
    const osmType = String(properties.osm_type || 'place')
    const osmId = String(properties.osm_id || `${lat.toFixed(6)},${lon.toFixed(6)}`)
    results.push({
      id: `${osmType}:${osmId}`,
      name,
      label: [name, ...context].join(', '),
      secondary: context.join(', '),
      lat: Number(lat.toFixed(7)),
      lon: Number(lon.toFixed(7)),
      type: String(properties.type || properties.osm_value || 'place'),
      source: 'openstreetmap',
    })
    if (results.length >= limit) break
  }
  return results
}

export function normaliseSearchQuery(query) {
  return String(query || '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function searchVerifiedCatalog(query, limit = 6) {
  const normalised = normaliseSearchQuery(query).toLocaleLowerCase()
  const tokens = normalised.split(' ').filter(Boolean)
  if (!tokens.length) return []
  return verifiedPlaceCatalog
    .map(place => {
      const aliases = [place.name, place.label, ...(place.aliases || [])]
        .map(value => normaliseSearchQuery(value).toLocaleLowerCase())
      const haystack = aliases.join(' ')
      if (!tokens.every(token => haystack.includes(token))) return null
      const rank = aliases.includes(normalised) ? 0 : aliases.some(alias => alias.startsWith(normalised)) ? 1 : 2
      const { id, name, label, secondary, lat, lon, type } = place
      return { rank, place: { id, name, label, secondary, lat, lon, type, source: 'verified_catalog' } }
    })
    .filter(Boolean)
    .sort((a, b) => a.rank - b.rank || a.place.name.localeCompare(b.place.name))
    .slice(0, limit)
    .map(item => item.place)
}

export function mergePlaceResults(primary, secondary, limit = 6) {
  const merged = []
  const seen = new Set()
  for (const place of [...primary, ...secondary]) {
    const key = `${place.name.toLocaleLowerCase()}|${Number(place.lat).toFixed(5)}|${Number(place.lon).toFixed(5)}`
    if (seen.has(key)) continue
    seen.add(key)
    merged.push(place)
    if (merged.length >= limit) break
  }
  return merged
}

export function normaliseGoogleAutocomplete(data, limit = 6) {
  const results = []
  for (const suggestion of data?.suggestions || []) {
    const prediction = suggestion?.placePrediction
    if (!prediction?.placeId) continue
    const name = String(prediction.structuredFormat?.mainText?.text || '').trim()
    const secondary = String(prediction.structuredFormat?.secondaryText?.text || '').trim()
    const label = String(prediction.text?.text || [name, secondary].filter(Boolean).join(', ')).trim()
    if (!name || !label) continue
    results.push({
      id: `google:${prediction.placeId}`,
      provider_id: prediction.placeId,
      name,
      label,
      secondary,
      type: String(prediction.types?.[0] || 'place'),
      source: 'google_places',
      needs_details: true,
    })
    if (results.length >= limit) break
  }
  return results
}

export function normaliseGooglePlace(data) {
  const lat = Number(data?.location?.latitude)
  const lon = Number(data?.location?.longitude)
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new ApiError('The selected place did not include a usable location.', 502)
  }
  parseCoordinate(`${lat},${lon}`, 'place')
  const name = String(data?.displayName?.text || '').trim()
  const address = String(data?.formattedAddress || '').trim()
  if (!name) throw new ApiError('The selected place did not include a usable address.', 502)
  const addressIncludesName = address.toLocaleLowerCase().startsWith(name.toLocaleLowerCase())
  const label = addressIncludesName ? address : [name, address].filter(Boolean).join(', ')
  const secondary = addressIncludesName
    ? address.slice(name.length).replace(/^,\s*/, '')
    : address
  return {
    id: `google:${data.id}`,
    name,
    label,
    secondary,
    lat,
    lon,
    type: String(data.primaryType || 'place'),
    source: 'google_places',
  }
}

function validSessionToken(value) {
  value = String(value || '').trim()
  return /^[A-Za-z0-9_-]{16,36}$/.test(value) ? value : undefined
}

async function searchGooglePlaces(query, sessionToken, limit) {
  const key = process.env.GOOGLE_PLACES_API_KEY
  if (!key) return []
  let response
  try {
    response = await fetch('https://places.googleapis.com/v1/places:autocomplete', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': key,
      },
      body: JSON.stringify({
        input: query,
        ...(validSessionToken(sessionToken) ? { sessionToken: validSessionToken(sessionToken) } : {}),
        includedRegionCodes: ['in'],
        languageCode: 'en',
        regionCode: 'IN',
        origin: { latitude: 28.4595, longitude: 77.0266 },
        locationRestriction: {
          rectangle: {
            low: { latitude: BBOX.minLat, longitude: BBOX.minLon },
            high: { latitude: BBOX.maxLat, longitude: BBOX.maxLon },
          },
        },
      }),
      signal: AbortSignal.timeout(10000),
    })
  } catch {
    return []
  }
  if (!response.ok) return []
  try { return normaliseGoogleAutocomplete(await response.json(), limit) } catch { return [] }
}

export async function resolveGooglePlace(placeId, sessionToken) {
  const key = process.env.GOOGLE_PLACES_API_KEY
  if (!key) throw new ApiError('High-precision place search is not configured for this deployment.', 503)
  placeId = String(placeId || '').trim()
  if (!/^[A-Za-z0-9_-]{8,256}$/.test(placeId)) throw new ApiError('Invalid place identifier.', 400)
  const url = new URL(`https://places.googleapis.com/v1/places/${placeId}`)
  url.searchParams.set('languageCode', 'en')
  url.searchParams.set('regionCode', 'IN')
  const token = validSessionToken(sessionToken)
  if (token) url.searchParams.set('sessionToken', token)
  let response
  try {
    response = await fetch(url, {
      headers: {
        Accept: 'application/json',
        'X-Goog-Api-Key': key,
        'X-Goog-FieldMask': 'id,displayName,formattedAddress,location,primaryType',
      },
      signal: AbortSignal.timeout(10000),
    })
  } catch {
    throw new ApiError('Place details could not be reached. Please try again.', 502)
  }
  if (response.status === 429) throw new ApiError('Place search is busy. Please try again shortly.', 429)
  if (response.status === 401 || response.status === 403) {
    throw new ApiError('High-precision place search is not configured for this deployment.', 503)
  }
  if (response.status === 404) throw new ApiError('The selected place is no longer available.', 404)
  if (!response.ok) throw new ApiError(`Place details returned HTTP ${response.status}.`, 502)
  let data
  try { data = await response.json() } catch { throw new ApiError('Place details returned an unreadable response.', 502) }
  return normaliseGooglePlace(data)
}

export async function searchPlaces(query, limit = 6, sessionToken) {
  query = String(query || '').trim().replace(/\s+/g, ' ')
  if (query.length < 2) throw new ApiError('Type at least two characters to search.', 400)
  if (query.length > 120) throw new ApiError('Location searches must be 120 characters or fewer.', 400)
  limit = Math.max(1, Math.min(Number(limit) || 6, 8))
  const google = await searchGooglePlaces(normaliseSearchQuery(query), sessionToken, limit)
  if (google.length) return google
  const verified = searchVerifiedCatalog(query, limit)
  const upstream = new URL(`${PHOTON_BASE_URL}/api`)
  upstream.search = new URLSearchParams({
    q: normaliseSearchQuery(query),
    limit: String(limit),
    lang: 'en',
    lat: '28.4595',
    lon: '77.0266',
    zoom: '10',
    location_bias_scale: '0.15',
    bbox: `${BBOX.minLon},${BBOX.minLat},${BBOX.maxLon},${BBOX.maxLat}`,
  }).toString()
  let response
  try {
    response = await fetch(upstream, {
      headers: {
        Accept: 'application/json',
        'User-Agent': 'gurugram-traffic-predictor/1.0 (place search)',
      },
      signal: AbortSignal.timeout(10000),
    })
  } catch {
    if (verified.length) return verified
    throw new ApiError('Place search could not be reached. Please try again.', 502)
  }
  if (response.status === 429) throw new ApiError('Place search is busy. Please try again shortly.', 429)
  if (!response.ok) throw new ApiError(`Place search returned HTTP ${response.status}.`, 502)
  let data
  try { data = await response.json() } catch { throw new ApiError('Place search returned an unreadable response.', 502) }
  return mergePlaceResults(verified, normalisePhoton(data, limit), limit)
}

export function parseCoordinate(raw, fieldName) {
  const pieces = String(raw || '').split(',')
  if (pieces.length !== 2) throw new ApiError(`'${fieldName}' must be 'latitude,longitude'.`, 400)
  const lat = Number(pieces[0])
  const lon = Number(pieces[1])
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    throw new ApiError(`'${fieldName}' contains an invalid coordinate.`, 400)
  }
  if (lat < BBOX.minLat || lat > BBOX.maxLat || lon < BBOX.minLon || lon > BBOX.maxLon) {
    throw new ApiError(`'${fieldName}' is outside the Gurugram/Delhi NCR service area.`, 400)
  }
  return { lat, lon }
}

export function validateDepartAt(value) {
  value = String(value || '').trim()
  if (!value) throw new ApiError("'departAt' is required.", 400)
  if (!/T\d{2}:\d{2}/.test(value) || !/(Z|[+-]\d{2}:\d{2})$/.test(value) || !Number.isFinite(Date.parse(value))) {
    throw new ApiError("'departAt' must be an ISO-8601 date and time with a timezone offset.", 400)
  }
  return value
}

export function normaliseTomTom(data) {
  const route = data?.routes?.[0]
  if (!route) throw new ApiError('No connected driving route was found.', 422)
  const summary = route.summary || {}
  const coordinates = []
  for (const leg of route.legs || []) {
    for (const point of leg.points || []) {
      const coord = [Number(point.longitude), Number(point.latitude)]
      if (!coord.every(Number.isFinite)) continue
      const previous = coordinates[coordinates.length - 1]
      if (!previous || previous[0] !== coord[0] || previous[1] !== coord[1]) coordinates.push(coord)
    }
  }
  if (coordinates.length < 2) throw new ApiError('The route provider did not return road geometry.', 502)
  const travelSeconds = Number(summary.travelTimeInSeconds || 0)
  const noTrafficSeconds = Number(summary.noTrafficTravelTimeInSeconds || 0)
  if (!(noTrafficSeconds > 0)) {
    throw new ApiError('The route provider did not return the no-traffic baseline required by the prediction model.', 502)
  }
  return {
    provider: 'tomtom',
    summary: {
      distance_meters: Number(summary.lengthInMeters || 0),
      travel_time_seconds: travelSeconds,
      traffic_delay_seconds: Math.max(0, Number(summary.trafficDelayInSeconds || (travelSeconds - noTrafficSeconds))),
      no_traffic_time_seconds: noTrafficSeconds,
      historic_traffic_time_seconds: Number(summary.historicTrafficTravelTimeInSeconds || travelSeconds),
      departure_time: summary.departureTime || null,
      arrival_time: summary.arrivalTime || null,
    },
    geometry: { type: 'LineString', coordinates },
  }
}

export async function calculateRoute(origin, destination, departAt) {
  const key = process.env.TOMTOM_API_KEY
  if (!key) throw new ApiError('Trip routing is not configured for this deployment.', 503)
  departAt = validateDepartAt(departAt)
  const path = `${origin.lat},${origin.lon}:${destination.lat},${destination.lon}`
  const upstream = new URL(`https://api.tomtom.com/routing/1/calculateRoute/${path}/json`)
  upstream.search = new URLSearchParams({
    key,
    departAt,
    // Geometry must not change because of TomTom's traffic forecast: the
    // product model, not the routing provider, compares departure hours.
    traffic: 'false',
    computeTravelTimeFor: 'all',
    routeRepresentation: 'polyline',
    routeType: 'shortest',
    travelMode: 'car',
  }).toString()
  let response
  try {
    response = await fetch(upstream, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(20000) })
  } catch {
    throw new ApiError('The route provider could not be reached. Please try again.', 502)
  }
  if (response.status === 429) throw new ApiError('The route provider is busy. Please try again shortly.', 429)
  if (response.status === 401 || response.status === 403) {
    throw new ApiError('Trip routing is not configured for this deployment.', 503)
  }
  if (response.status === 404) throw new ApiError('No connected driving route was found.', 422)
  if (!response.ok) throw new ApiError(`The route provider returned HTTP ${response.status}.`, 502)
  let data
  try { data = await response.json() } catch { throw new ApiError('The route provider returned an unreadable response.', 502) }
  return normaliseTomTom(data)
}
