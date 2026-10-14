import L from 'leaflet';
import { useEffect, useRef, useState } from 'react';

const DELHI = [28.62, 77.1];
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '–' : v.toFixed(d));

async function getJson(url, init) {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`);
  return body;
}

const CLOSE_PX = 12; // a click this close to the first vertex closes the polygon

function coolingColor(c, max) {
  const t = Math.max(0, Math.min(1, c / Math.max(max, 0.01)));
  return `rgba(28, 130, 160, ${0.12 + 0.6 * t})`;
}

export default function App() {
  const mapEl = useRef(null);
  const map = useRef(null);
  const layers = useRef({});
  const lastPolygon = useRef(null);
  const [meta, setMeta] = useState(null);
  const [hotspots, setHotspots] = useState([]);
  const [ndviTarget, setNdviTarget] = useState(0.6);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [showLst, setShowLst] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [nVerts, setNVerts] = useState(0);
  const draft = useRef(null); // { pts: LatLng[], layer, guide }
  const ndviTargetRef = useRef(ndviTarget);
  ndviTargetRef.current = ndviTarget;

  async function simulate(geometry, target) {
    setBusy(true);
    setError('');
    try {
      const r = await getJson('/api/simulate', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ geometry, ndviTarget: target }),
      });
      setResult(r);
      const max = r.prediction.dT;
      layers.current.rings?.remove();
      layers.current.rings = L.geoJSON(r.rings, {
        style: (f) => ({ weight: 0, fillColor: coolingColor(f.properties.cooling_c, max), fillOpacity: 1 }),
        onEachFeature: (f, l) =>
          l.bindTooltip(`${f.properties.outer_m - 50}–${f.properties.outer_m} m: ${fmt(f.properties.cooling_c, 2)} °C cooler`),
      }).addTo(map.current);
      layers.current.drawn.bringToFront();
    } catch (e) {
      setResult(null);
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function startDrawing() {
    cancelDrawing();
    const g = layers.current.draftGroup;
    const style = { color: '#1c82a0', weight: 2, dashArray: '4 4', fillOpacity: 0.1, interactive: false };
    draft.current = { pts: [], layer: L.polygon([], style).addTo(g), guide: L.polyline([], { ...style, fill: false }).addTo(g) };
    map.current.doubleClickZoom.disable();
    mapEl.current.classList.add('drawing');
    setNVerts(0);
    setDrawing(true);
  }

  function addVertex(latlng) {
    const d = draft.current;
    if (!d) return;
    const m = map.current;
    if (d.pts.length >= 3 && m.latLngToContainerPoint(d.pts[0]).distanceTo(m.latLngToContainerPoint(latlng)) < CLOSE_PX) {
      finishDrawing();
      return;
    }
    d.pts.push(latlng);
    d.layer.setLatLngs(d.pts);
    L.circleMarker(latlng, { radius: 4, color: '#1c82a0', fillColor: '#fff', fillOpacity: 1, weight: 2, interactive: false }).addTo(
      layers.current.draftGroup,
    );
    setNVerts(d.pts.length);
  }

  function cancelDrawing() {
    draft.current = null;
    layers.current.draftGroup?.clearLayers();
    map.current?.doubleClickZoom.enable();
    mapEl.current?.classList.remove('drawing');
    setDrawing(false);
  }

  function finishDrawing() {
    const pts = draft.current?.pts ?? [];
    cancelDrawing();
    if (pts.length < 3) return;
    const poly = L.polygon(pts, { color: '#1c82a0', weight: 2, fillOpacity: 0.15, interactive: false });
    layers.current.drawn.clearLayers().addLayer(poly);
    lastPolygon.current = poly.toGeoJSON().geometry;
    simulate(lastPolygon.current, ndviTargetRef.current);
  }

  function clearPlan() {
    cancelDrawing();
    layers.current.drawn.clearLayers();
    layers.current.rings?.remove();
    lastPolygon.current = null;
    setResult(null);
  }

  useEffect(() => {
    const onKey = (e) => {
      if (!draft.current) return;
      if (e.key === 'Escape') cancelDrawing();
      if (e.key === 'Enter') finishDrawing();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    const m = L.map(mapEl.current, { zoomControl: true }).setView(DELHI, 11);
    map.current = m;
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(m);
    layers.current.drawn = L.featureGroup().addTo(m);
    layers.current.draftGroup = L.featureGroup().addTo(m);
    m.on('click', (e) => addVertex(e.latlng));
    m.on('mousemove', (e) => {
      const d = draft.current;
      if (d?.pts.length) d.guide.setLatLngs([d.pts[d.pts.length - 1], e.latlng]);
    });

    getJson('/api/meta')
      .then(async (mt) => {
        setMeta(mt);
        const overlay = await fetch(`/layers/lst_${mt.year}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
        if (overlay) {
          layers.current.lst = L.imageOverlay(`/layers/lst_${mt.year}.png`, overlay.bounds, { opacity: 0.6 }).addTo(m);
          layers.current.lstMeta = overlay;
          setMeta({ ...mt, overlay });
        }
        const hs = await getJson('/api/hotspots?limit=20');
        setHotspots(hs.features);
        layers.current.hotspots = L.geoJSON(hs, {
          style: { color: '#c2410c', weight: 1.5, fillOpacity: 0.08 },
          onEachFeature: (f, l) => l.bindTooltip(`Hotspot #${f.properties.rank}: +${fmt(f.properties.mean_intensity)} °C`),
        }).addTo(m);
      })
      .catch((e) => setError(`Cannot reach the API: ${e.message}`));
    return () => m.remove();
  }, []);

  useEffect(() => {
    const l = layers.current.lst;
    if (!l || !map.current) return;
    if (showLst) l.addTo(map.current);
    else l.remove();
  }, [showLst, meta]);

  const p = result?.prediction;
  return (
    <div className="app">
      <aside className="panel">
        <header>
          <h1>Delhi Heat Planner</h1>
          <p className="muted">
            Surface heat in Delhi{meta ? `, April–June ${meta.year}` : ''}. Draw a proposed green space to estimate how much it cools
            the ground around it.
          </p>
          {meta?.synthetic && (
            <p className="banner" role="alert">
              Demo mode: everything on this map comes from a synthetic test city. It is not real Delhi data.
            </p>
          )}
        </header>

        <section>
          <h2>Map layers</h2>
          <label className="row" htmlFor="show-lst">
            <input id="show-lst" type="checkbox" checked={showLst} onChange={(e) => setShowLst(e.target.checked)} disabled={!meta?.overlay} />
            Land surface temperature
          </label>
          {meta?.overlay ? (
            <div className="ramp" aria-hidden="true">
              <span>{fmt(meta.overlay.min_c, 0)} °C</span>
              <span className="ramp-bar" />
              <span>{fmt(meta.overlay.max_c, 0)} °C</span>
            </div>
          ) : (
            <p className="muted small">No temperature layer yet. Run the overlay step of the pipeline.</p>
          )}
        </section>

        <section>
          <h2>Hottest areas</h2>
          {hotspots.length === 0 ? (
            <p className="muted small">No hotspots loaded.</p>
          ) : (
            <ol className="hotspots">
              {hotspots.slice(0, 8).map((f) => (
                <li key={f.properties.rank}>
                  <button type="button" onClick={() => map.current.fitBounds(L.geoJSON(f).getBounds(), { maxZoom: 15 })}>
                    <span>#{f.properties.rank}</span>
                    <span>+{fmt(f.properties.mean_intensity)} °C</span>
                    <span>{fmt(f.properties.area_m2 / 1e4)} ha</span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section>
          <h2>Plan a green space</h2>
          <label htmlFor="ndvi-target">
            Planting density (target NDVI): <strong>{ndviTarget.toFixed(2)}</strong>
          </label>
          <input
            id="ndvi-target"
            type="range"
            min="0.3"
            max="0.85"
            step="0.05"
            value={ndviTarget}
            onChange={(e) => setNdviTarget(Number(e.target.value))}
            onPointerUp={() => lastPolygon.current && simulate(lastPolygon.current, ndviTarget)}
            onKeyUp={() => lastPolygon.current && simulate(lastPolygon.current, ndviTarget)}
          />
          <p className="muted small">0.4 is scattered trees, 0.6 a park, 0.8 dense canopy.</p>
          <div className="buttons">
            {drawing ? (
              <>
                <button type="button" className="primary" onClick={finishDrawing} disabled={nVerts < 3}>
                  Finish area
                </button>
                <button type="button" onClick={cancelDrawing}>Cancel</button>
              </>
            ) : (
              <>
                <button type="button" className="primary" onClick={startDrawing}>
                  {result ? 'Draw a new area' : 'Draw an area'}
                </button>
                {result && <button type="button" onClick={clearPlan}>Clear</button>}
              </>
            )}
          </div>
          {drawing && (
            <p className="muted small" role="status">
              Click the map to add corners ({nVerts} so far). Click the first corner, press Enter or Finish to close; Esc cancels.
            </p>
          )}
          {busy && <p className="small">Estimating…</p>}
          {error && <p className="error" role="alert">{error}</p>}
          {p && (
            <div className="result">
              <div className="big">
                {fmt(p.dT)} <span>°C cooler at the edge</span>
              </div>
              <p className="small">
                90% range {fmt(p.dT_q[0])}–{fmt(p.dT_q[2])} °C. Cooling stays above 0.5 °C for about{' '}
                <strong>{fmt(p.extent_m, 0)} m</strong> (range {fmt(p.extent_q[0], 0)}–{fmt(p.extent_q[2], 0)} m).
              </p>
              <dl className="small">
                <dt>Area</dt>
                <dd>{fmt(result.input.area_m2 / 1e4, 2)} ha</dd>
                <dt>Greenness today</dt>
                <dd>NDVI {fmt(result.input.ndvi_now, 2)}</dd>
                <dt>Built-up around it</dt>
                <dd>{fmt(result.input.built * 100, 0)}%</dd>
                <dt>Share of mature-park cooling</dt>
                <dd>{fmt(p.scale * 100, 0)}%</dd>
              </dl>
              <p className="muted small">
                Surface temperature, not air temperature. Estimated from {meta?.nSites ?? '–'} existing {meta?.synthetic ? 'synthetic' : 'Delhi'} green spaces; a new planting
                reaches this only once it matures.
              </p>
            </div>
          )}
        </section>
      </aside>
      <div ref={mapEl} className="map" role="application" aria-label="Map of Delhi" />
    </div>
  );
}
