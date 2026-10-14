// Cooling prediction for a drawn polygon. Mirrors pipeline/uhi/cooling.py:predict exactly;
// both are checked against pipeline/tests/fixtures/cooling_case.json.

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

// numpy.percentile with the default linear interpolation.
export function percentile(values, q) {
  const s = [...values].sort((a, b) => a - b);
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function ndviScale(ndviTarget, ndviNow, ndviSoil) {
  const denom = ndviTarget - ndviSoil;
  if (denom <= 0) return 0;
  return clip((ndviTarget - ndviNow) / denom, 0, 1);
}

export function extentM(dT, ell, threshold) {
  return dT > threshold ? ell * Math.log(dT / threshold) : 0;
}

export function predict(params, { area_m2, ndvi_target, ndvi_now, shape, built }, q = [5, 50, 95]) {
  const lnArea = Math.log(area_m2);
  const xt = [1, lnArea, ndvi_target, shape, built];
  const xp = [1, lnArea, ndvi_target];
  const s = ndviScale(ndvi_target, ndvi_now, params.ndvi_soil);
  const thr = params.extent_threshold_c;
  const b = params.boot;
  const dTs = b.theta.map((th, i) => Math.max((dot(th, xt) + b.eps_dT[i]) * s, 0));
  const ells = b.phi.map((ph, i) => Math.exp(dot(ph, xp) + b.eps_lnl[i]));
  const extents = dTs.map((d, i) => extentM(d, ells[i], thr));
  const dT = Math.max(dot(params.theta, xt) * s, 0);
  const ell = Math.exp(dot(params.phi, xp));
  return {
    scale: s,
    dT,
    ell,
    extent_m: extentM(dT, ell, thr),
    dT_q: q.map((p) => percentile(dTs, p)),
    ell_q: q.map((p) => percentile(ells, p)),
    extent_q: q.map((p) => percentile(extents, p)),
  };
}

// Cooling at distance d from the polygon edge, in deg C.
export const coolingAt = (dT, ell, d) => dT * Math.exp(-d / ell);
