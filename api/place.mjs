import { errorResponse, jsonResponse, resolveGooglePlace } from './_trip_lib.mjs'

export default {
  async fetch(request) {
    try {
      if (request.method !== 'GET') return jsonResponse({ error: 'Method not allowed.' }, 405)
      const url = new URL(request.url)
      const place = await resolveGooglePlace(
        url.searchParams.get('id'),
        url.searchParams.get('sessionToken'),
      )
      return jsonResponse(
        { place, provider: 'google_places', attribution: 'Google Places' },
        200,
        { 'Cache-Control': 'private, no-store' },
      )
    } catch (error) {
      return errorResponse(error)
    }
  },
}
