import { errorResponse, jsonResponse, searchPlaces } from './_trip_lib.mjs'

export default {
  async fetch(request) {
    try {
      if (request.method !== 'GET') return jsonResponse({ error: 'Method not allowed.' }, 405)
      const url = new URL(request.url)
      const results = await searchPlaces(url.searchParams.get('q'), 6, url.searchParams.get('sessionToken'))
      const usesGoogle = results.some(place => place.source === 'google_places')
      return jsonResponse(
        {
          results,
          provider: usesGoogle ? 'google_places' : 'photon+verified-catalog',
          attribution: usesGoogle
            ? 'Google Places'
            : 'OpenStreetMap contributors and verified official place sources',
        },
        200,
        {
          'Cache-Control': usesGoogle
            ? 'private, no-store'
            : 'public, s-maxage=300, stale-while-revalidate=86400',
        },
      )
    } catch (error) {
      return errorResponse(error)
    }
  },
}
