import {
  calculateRoute,
  errorResponse,
  jsonResponse,
  parseCoordinate,
} from './_trip_lib.mjs'

export default {
  async fetch(request) {
    try {
      if (request.method !== 'GET') return jsonResponse({ error: 'Method not allowed.' }, 405)
      const url = new URL(request.url)
      const origin = parseCoordinate(url.searchParams.get('origin'), 'origin')
      const destination = parseCoordinate(url.searchParams.get('destination'), 'destination')
      const route = await calculateRoute(origin, destination, url.searchParams.get('departAt'))
      return jsonResponse(route, 200, { 'Cache-Control': 'private, no-store' })
    } catch (error) {
      return errorResponse(error)
    }
  },
}

