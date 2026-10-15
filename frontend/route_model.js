(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.GTPRouteModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const MIN_MODELLED_METERS = 1000;
  const MIN_ROUTE_COVERAGE_PCT = 25;

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  /**
   * Apply the project's corridor forecast to the part of a real TomTom route
   * that overlaps monitored roads. TomTom supplies only the connected geometry
   * and the no-traffic baseline; the time-dependent delay comes from our model.
   */
  function estimateRoute(summary, segments) {
    // Never fall back to the provider's traffic-aware ETA here: doing so would
    // silently put TomTom back in charge of the recommendation. A missing
    // no-traffic baseline makes the candidate unusable instead.
    const baseSeconds = Math.max(0, finiteNumber(
      summary && summary.no_traffic_time_seconds,
      0,
    ));
    let modelDelaySeconds = 0;
    let modelledMeters = 0;
    let reverseMeters = 0;
    let confidenceWeighted = 0;

    for (const segment of segments || []) {
      const overlapMeters = Math.max(0, finiteNumber(segment.overlapMeters, 0));
      const corridorMeters = Math.max(1, finiteNumber(segment.corridorLengthKm, 0) * 1000);
      const fullCorridorDelayMinutes = Math.max(0, finiteNumber(segment.delayMinutes, 0));
      if (!overlapMeters || !Number.isFinite(Number(segment.delayMinutes))) continue;
      const shareOfCorridor = Math.min(1, overlapMeters / corridorMeters);
      modelDelaySeconds += fullCorridorDelayMinutes * 60 * shareOfCorridor;
      modelledMeters += overlapMeters;
      const directionFactor = segment.direction === 'reverse' ? 0.65 : 1;
      reverseMeters += segment.direction === 'reverse' ? overlapMeters : 0;
      confidenceWeighted += Math.max(0, Math.min(1, finiteNumber(segment.confidence, 0)))
        * directionFactor * overlapMeters;
    }

    return {
      base_seconds: baseSeconds,
      model_delay_seconds: modelDelaySeconds,
      travel_time_seconds: baseSeconds + modelDelaySeconds,
      modelled_meters: modelledMeters,
      reverse_meters: reverseMeters,
      confidence: modelledMeters ? confidenceWeighted / modelledMeters : 0,
      model_based: modelledMeters > 0 && baseSeconds > 0,
    };
  }

  function rankCandidates(candidates) {
    return (candidates || [])
      .filter(item => item && item.estimate && item.estimate.model_based
        && item.estimate.travel_time_seconds > 0)
      .slice()
      .sort((a, b) => a.estimate.travel_time_seconds - b.estimate.travel_time_seconds);
  }

  function assessCoverage(routeDistanceMeters, modelledMeters) {
    const routeMeters = Math.max(0, finiteNumber(routeDistanceMeters, 0));
    const coveredMeters = Math.max(0, finiteNumber(modelledMeters, 0));
    const coveragePct = routeMeters > 0 ? Math.min(100, coveredMeters / routeMeters * 100) : 0;
    return {
      sufficient: coveredMeters >= MIN_MODELLED_METERS
        && coveragePct >= MIN_ROUTE_COVERAGE_PCT,
      modelled_meters: coveredMeters,
      route_distance_meters: routeMeters,
      coverage_pct: coveragePct,
      minimum_modelled_meters: MIN_MODELLED_METERS,
      minimum_coverage_pct: MIN_ROUTE_COVERAGE_PCT,
    };
  }

  return {
    MIN_MODELLED_METERS,
    MIN_ROUTE_COVERAGE_PCT,
    assessCoverage,
    estimateRoute,
    rankCandidates,
  };
});
