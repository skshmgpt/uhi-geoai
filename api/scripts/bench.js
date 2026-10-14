// E9: round-trip latency of /api/simulate for random polygons of 0.1-50 ha inside the study area.
// Usage: node scripts/bench.js [baseUrl] [n]   (writes ../pipeline/results/e9_latency.json)
import { writeFileSync } from 'node:fs';

const BASE = process.argv[2] ?? 'http://localhost:8080';
const N = Number(process.argv[3] ?? 1000);
const WARMUP = 20;
// Study box 76.84-77.35 E, 28.40-28.89 N, shrunk by ~2 km so every polygon is inside it.
const BOX = { w: 76.86, e: 77.33, s: 28.42, n: 28.87 };
const M_PER_DEG_LAT = 111_320;

// Seeded PRNG (mulberry32) so the polygon set is reproducible.
let seed = 20261015;
const rand = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Star-shaped polygon with 6-10 vertices, scaled to a log-uniform target area in [0.1, 50] ha.
function randomPolygon() {
  const areaM2 = 10 ** (Math.log10(1_000) + rand() * (Math.log10(500_000) - Math.log10(1_000)));
  const k = 6 + Math.floor(rand() * 5);
  const angles = Array.from({ length: k }, (_, i) => (2 * Math.PI * (i + rand() * 0.6)) / k);
  const radii = angles.map(() => 0.7 + rand() * 0.6);
  // Area of the unit star polygon, then scale radii to hit the target area.
  let unit = 0;
  for (let i = 0; i < k; i++) {
    const j = (i + 1) % k;
    unit += 0.5 * radii[i] * radii[j] * Math.sin(angles[j] - angles[i] + (j === 0 ? 2 * Math.PI : 0));
  }
  const scale = Math.sqrt(areaM2 / unit);
  const lat0 = BOX.s + rand() * (BOX.n - BOX.s);
  const lon0 = BOX.w + rand() * (BOX.e - BOX.w);
  const mPerDegLon = M_PER_DEG_LAT * Math.cos((lat0 * Math.PI) / 180);
  const ring = angles.map((a, i) => [
    lon0 + (Math.cos(a) * radii[i] * scale) / mPerDegLon,
    lat0 + (Math.sin(a) * radii[i] * scale) / M_PER_DEG_LAT,
  ]);
  ring.push(ring[0]);
  return { areaM2, geometry: { type: 'Polygon', coordinates: [ring] } };
}

async function simulate(geometry) {
  const t0 = performance.now();
  const r = await fetch(`${BASE}/api/simulate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ geometry, ndviTarget: 0.6 }),
  });
  const body = await r.json();
  return { ms: performance.now() - t0, status: r.status, serverMs: body.took_ms };
}

const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};
const r1 = (x) => Math.round(x * 10) / 10;

for (let i = 0; i < WARMUP; i++) await simulate(randomPolygon().geometry);

const ok = [];
const server = [];
const statuses = {};
for (let i = 0; i < N; i++) {
  const { geometry } = randomPolygon();
  const res = await simulate(geometry);
  statuses[res.status] = (statuses[res.status] ?? 0) + 1;
  if (res.status === 200) {
    ok.push(res.ms);
    server.push(res.serverMs);
  }
}

const hot = [];
for (let i = 0; i < 50; i++) {
  const t0 = performance.now();
  await (await fetch(`${BASE}/api/hotspots?limit=50`)).json();
  hot.push(performance.now() - t0);
}

const out = {
  n_requests: N,
  warmup: WARMUP,
  area_ha: [0.1, 50],
  statuses,
  simulate_round_trip_ms: { p50: r1(pct(ok, 50)), p95: r1(pct(ok, 95)), max: r1(Math.max(...ok)), n: ok.length },
  simulate_server_ms: { p50: r1(pct(server, 50)), p95: r1(pct(server, 95)) },
  hotspots_top50_ms: { p50: r1(pct(hot, 50)), p95: r1(pct(hot, 95)), n: hot.length },
  setup: 'sequential requests, client and server on the same laptop, PostGIS in Docker',
};
writeFileSync(new URL('../../pipeline/results/e9_latency.json', import.meta.url), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out, null, 2));
