const assert = require('node:assert/strict');
const test = require('node:test');

const { assessCoverage, estimateRoute, rankCandidates } = require('../frontend/route_model.js');

test('route estimate uses model delay instead of provider traffic time', () => {
  const estimate = estimateRoute(
    { no_traffic_time_seconds: 1200, travel_time_seconds: 9999 },
    [{
      overlapMeters: 10000,
      corridorLengthKm: 10,
      delayMinutes: 6,
      confidence: 0.8,
      direction: 'forward',
    }],
  );
  assert.equal(estimate.travel_time_seconds, 1560);
  assert.equal(estimate.model_delay_seconds, 360);
  assert.equal(estimate.confidence, 0.8);
});

test('partial corridor overlap scales the model delay', () => {
  const estimate = estimateRoute(
    { no_traffic_time_seconds: 900 },
    [{
      overlapMeters: 2000,
      corridorLengthKm: 10,
      delayMinutes: 5,
      confidence: 0.9,
      direction: 'reverse',
    }],
  );
  assert.equal(estimate.model_delay_seconds, 60);
  assert.ok(Math.abs(estimate.confidence - 0.585) < 1e-12);
  assert.equal(estimate.reverse_meters, 2000);
});

test('candidate ranking follows model estimates and excludes uncovered routes', () => {
  const ranked = rankCandidates([
    { hour: 10, estimate: { model_based: true, travel_time_seconds: 1500 } },
    { hour: 7, estimate: { model_based: true, travel_time_seconds: 1200 } },
    { hour: 8, estimate: { model_based: false, travel_time_seconds: 900 } },
  ]);
  assert.deepEqual(ranked.map(item => item.hour), [7, 10]);
});

test('missing no-traffic baseline is never replaced with provider traffic time', () => {
  const estimate = estimateRoute(
    { travel_time_seconds: 9999 },
    [{
      overlapMeters: 10000,
      corridorLengthKm: 10,
      delayMinutes: 6,
      confidence: 0.8,
      direction: 'forward',
    }],
  );
  assert.equal(estimate.base_seconds, 0);
  assert.equal(estimate.model_based, false);
  assert.deepEqual(rankCandidates([{ hour: 7, estimate }]), []);
});

test('coverage requires both one kilometre and one quarter of the route', () => {
  assert.equal(assessCoverage(5000, 800).sufficient, false);
  assert.equal(assessCoverage(10000, 1200).sufficient, false);
  assert.equal(assessCoverage(5000, 1300).sufficient, true);
  assert.equal(assessCoverage(5000, 1300).coverage_pct, 26);
});
