import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { createApp, validateSimulation } from '../src/app.js';
import { ndviScale, percentile, predict } from '../src/cooling.js';

const fixture = JSON.parse(
  readFileSync(new URL('../../pipeline/tests/fixtures/cooling_case.json', import.meta.url), 'utf8'),
);

const close = (a, b) => Math.abs(a - b) <= 1e-9 + 1e-9 * Math.abs(b);

test('predict matches the Python implementation on the shared fixture', () => {
  for (const { input, expected } of fixture.cases) {
    const got = predict(fixture.params, input);
    for (const [k, v] of Object.entries(expected)) {
      if (Array.isArray(v)) v.forEach((x, i) => assert.ok(close(got[k][i], x), `${k}[${i}]: ${got[k][i]} vs ${x}`));
      else assert.ok(close(got[k], v), `${k}: ${got[k]} vs ${v}`);
    }
  }
});

test('percentile interpolates like numpy', () => {
  assert.equal(percentile([1, 2, 3, 4], 50), 2.5);
  assert.equal(percentile([5, 1, 3], 0), 1);
  assert.ok(close(percentile([0, 10], 5), 0.5));
});

test('ndviScale clips to [0, 1]', () => {
  assert.equal(ndviScale(0.6, 0.2, 0.2), 1);
  assert.equal(ndviScale(0.6, 0.7, 0.2), 0);
  assert.equal(ndviScale(0.1, 0.0, 0.2), 0);
});

test('validateSimulation rejects bad input', () => {
  assert.throws(() => validateSimulation({}), /Polygon/);
  assert.throws(() => validateSimulation({ geometry: { type: 'Point', coordinates: [0, 0] } }), /Polygon/);
  const poly = { type: 'Polygon', coordinates: [[[77, 28.6], [77.01, 28.6], [77.01, 28.61], [77, 28.6]]] };
  assert.throws(() => validateSimulation({ geometry: poly, ndviTarget: 2 }), /ndviTarget/);
  assert.equal(validateSimulation({ geometry: poly }).ndviTarget, 0.6);
});

async function call(app, method, path, body) {
  const server = app.listen(0);
  const { port } = server.address();
  try {
    const res = await fetch(`http://127.0.0.1:${port}${path}`, {
      method,
      headers: { 'content-type': 'application/json' },
      body: body && JSON.stringify(body),
    });
    return { status: res.status, body: await res.json() };
  } finally {
    server.close();
  }
}

const square = { type: 'Polygon', coordinates: [[[77.2, 28.6], [77.202, 28.6], [77.202, 28.602], [77.2, 28.602], [77.2, 28.6]]] };

function fakeDb(traits) {
  return {
    params: async () => fixture.params,
    hotspots: async () => ({ type: 'FeatureCollection', features: [] }),
    traits: async () => traits,
    rings: async (_g, step, max) =>
      Array.from({ length: max / step }, (_, i) => ({ outer_m: (i + 1) * step, geometry: square })),
  };
}

test('POST /api/simulate returns a prediction and cooling rings that fade with distance', async () => {
  const app = createApp({ db: fakeDb({ area_m2: 4e4, shape: 1.1, valid: true, ndvi_now: 0.15, built: 0.7 }), year: 2024 });
  const { status, body } = await call(app, 'POST', '/api/simulate', { geometry: square, ndviTarget: 0.7 });
  assert.equal(status, 200);
  assert.ok(body.prediction.dT > 0);
  const c = body.rings.features.map((f) => f.properties.cooling_c);
  assert.ok(c.length >= 1);
  for (let i = 1; i < c.length; i++) assert.ok(c[i] < c[i - 1]);
});

test('POST /api/simulate explains polygons outside the study area', async () => {
  const app = createApp({ db: fakeDb({ area_m2: 4e4, shape: 1.1, valid: true, ndvi_now: null, built: null }), year: 2024 });
  const { status, body } = await call(app, 'POST', '/api/simulate', { geometry: square });
  assert.equal(status, 400);
  assert.match(body.error, /outside/);
});

test('POST /api/simulate rejects tiny polygons', async () => {
  const app = createApp({ db: fakeDb({ area_m2: 100, shape: 1, valid: true, ndvi_now: 0.2, built: 0.5 }), year: 2024 });
  const { status } = await call(app, 'POST', '/api/simulate', { geometry: square });
  assert.equal(status, 400);
});
