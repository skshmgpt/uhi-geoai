import express from 'express';

import { coolingAt, predict } from './cooling.js';

const RING_STEP_M = 50;
const RING_MAX_M = 1000;
const MIN_AREA_M2 = 500;
const MAX_AREA_M2 = 5_000_000;

export class InputError extends Error {}

export function validateSimulation(body) {
  const geometry = body?.geometry;
  if (!geometry || geometry.type !== 'Polygon' || !Array.isArray(geometry.coordinates)) {
    throw new InputError('Send a GeoJSON Polygon in "geometry".');
  }
  const ndviTarget = body.ndviTarget ?? 0.6;
  if (typeof ndviTarget !== 'number' || ndviTarget < 0.25 || ndviTarget > 0.9) {
    throw new InputError('"ndviTarget" must be a number between 0.25 and 0.9.');
  }
  return { geometry, ndviTarget };
}

// `db` is injected so routes can be tested without PostGIS.
// Year of the synthetic test city (pipeline/uhi/synthetic.py); never a real observation year.
export const SYNTHETIC_YEAR = 2099;

export function createApp({ db, year }) {
  const app = express();
  app.use(express.json({ limit: '1mb' }));
  let cachedParams = null;
  const params = async () => (cachedParams ??= await db.params(year));

  app.get('/api/health', (_req, res) => res.json({ ok: true, year }));

  app.get('/api/meta', async (_req, res, next) => {
    try {
      const p = await params();
      res.json({ year, synthetic: year === SYNTHETIC_YEAR, calibrated: Boolean(p), nSites: p?.n_sites ?? 0 });
    } catch (e) {
      next(e);
    }
  });

  app.get('/api/hotspots', async (req, res, next) => {
    try {
      const limit = Math.min(Number(req.query.limit ?? 50), 500);
      res.json(await db.hotspots(year, { limit }));
    } catch (e) {
      next(e);
    }
  });

  app.post('/api/simulate', async (req, res, next) => {
    try {
      const t0 = performance.now();
      const { geometry, ndviTarget } = validateSimulation(req.body);
      const p = await params();
      if (!p) return res.status(503).json({ error: `No cooling model is loaded for ${year}. Run scripts/load_db.sh ${year}.` });

      const t = await db.traits(geometry);
      if (!t?.valid) throw new InputError('The polygon is not valid (it may cross itself).');
      if (t.area_m2 < MIN_AREA_M2 || t.area_m2 > MAX_AREA_M2) {
        throw new InputError(`Draw an area between ${MIN_AREA_M2} m² and ${MAX_AREA_M2 / 1e6} km².`);
      }
      if (t.ndvi_now == null || t.built == null) {
        throw new InputError('The polygon is outside the analysed area of Delhi.');
      }

      const input = { area_m2: t.area_m2, ndvi_target: ndviTarget, ndvi_now: t.ndvi_now, shape: t.shape, built: t.built };
      const pred = predict(p, input);
      const reach = Math.min(Math.max(pred.extent_q[2], RING_STEP_M), RING_MAX_M);
      const maxM = Math.ceil(reach / RING_STEP_M) * RING_STEP_M;
      const rings = await db.rings(geometry, RING_STEP_M, maxM);
      res.json({
        input,
        prediction: pred,
        rings: {
          type: 'FeatureCollection',
          features: rings.map((r) => {
            const mid = r.outer_m - RING_STEP_M / 2;
            return {
              type: 'Feature',
              geometry: r.geometry,
              properties: { outer_m: r.outer_m, cooling_c: coolingAt(pred.dT, pred.ell, mid) },
            };
          }),
        },
        took_ms: Math.round(performance.now() - t0),
      });
    } catch (e) {
      next(e);
    }
  });

  app.use((err, _req, res, _next) => {
    if (err instanceof InputError) return res.status(400).json({ error: err.message });
    console.error(err);
    res.status(500).json({ error: 'The server hit an unexpected error. Check the API logs.' });
  });

  return app;
}
