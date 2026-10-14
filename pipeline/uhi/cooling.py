"""Green-space cooling model: per-site ring profiles, exponential decay fits, cross-site regression.

All rasters here are on the 30 m thermal grid. A calibration site is a connected patch of
30 m pixels whose canopy fraction (from the 10 m segmentation) is at least 0.5.
"""
import json
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from . import config as C
from .lst import NDVI_SOIL

PIXEL_M = 30.0
THETA_FEATURES = ["const", "ln_area", "ndvi", "shape", "built"]
PHI_FEATURES = ["const", "ln_area", "ndvi"]


def block_fraction(mask_10m, factor=3):
    """Fraction of True 10 m pixels inside each 30 m block."""
    h, w = mask_10m.shape[0] // factor * factor, mask_10m.shape[1] // factor * factor
    m = mask_10m[:h, :w].astype(np.float32)
    return m.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def label_sites(canopy_frac, min_area_m2=C.SITE_MIN_AREA_M2, min_frac=C.SITE_CANOPY_FRACTION):
    labels, n = ndimage.label(canopy_frac >= min_frac, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=n + 1) * PIXEL_M**2
    keep = sizes >= min_area_m2
    keep[0] = False
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[labels], int(keep.sum())


def site_shape_index(site_mask):
    """Fragstats SHAPE = 0.25 P / sqrt(A), on the polygon simplified by half a pixel.

    Simplifying removes most of the raster stair-step so the value is comparable with
    PostGIS on a hand-drawn polygon.
    """
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    polys = [shape(g) for g, v in features.shapes(site_mask.astype(np.uint8), mask=site_mask) if v == 1]
    geom = unary_union(polys).simplify(0.5, preserve_topology=True)  # pixel units
    return 0.25 * geom.length * PIXEL_M / math.sqrt(geom.area * PIXEL_M**2)


def site_profiles(lst, sites, n_sites, ndvi, built_frac, exclude, ring_m=C.RING_WIDTH_M, max_m=C.RING_MAX_M):
    """Mean LST per distance ring around each site, plus site traits.

    `exclude` marks pixels left out of every ring mean (water, other canopy).
    Returns a list of dicts; `profile` is indexed by ring (ring k covers ((k-1)*w, k*w]).
    """
    n_rings = n_rings_for(ring_m, max_m)
    pad = int(math.ceil(max_m / PIXEL_M)) + 1
    objects = ndimage.find_objects(sites)
    out = []
    for sid in range(1, n_sites + 1):
        sl = objects[sid - 1]
        r0, r1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, sites.shape[0])
        c0, c1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, sites.shape[1])
        win = (slice(r0, r1), slice(c0, c1))
        site = sites[win] == sid
        # Distance from the site edge: centre-to-centre distance minus half a pixel.
        dist = ndimage.distance_transform_edt(~site) * PIXEL_M - PIXEL_M / 2
        valid = ~site & ~exclude[win] & np.isfinite(lst[win]) & (dist > 0) & (dist <= max_m)
        ring = np.ceil(dist[valid] / ring_m).astype(int)
        sums = np.bincount(ring, weights=lst[win][valid], minlength=n_rings + 1)[1:]
        counts = np.bincount(ring, minlength=n_rings + 1)[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.where(counts > 0, sums / counts, np.nan)
        near = ~site & (dist <= C.BUILT_BUFFER_M) & np.isfinite(built_frac[win])
        area = float(site.sum() * PIXEL_M**2)
        cy, cx = ndimage.center_of_mass(site)
        out.append({
            "site_id": sid,
            "row": r0 + cy,
            "col": c0 + cx,
            "area_m2": area,
            "ndvi": float(np.nanmean(ndvi[win][site])),
            "shape": site_shape_index(site),
            "built": float(np.mean(built_frac[win][near])) if near.any() else float("nan"),
            "profile": profile,
            "counts": counts,
        })
    return out


def n_rings_for(ring_m=C.RING_WIDTH_M, max_m=C.RING_MAX_M):
    return int(math.ceil(max_m / ring_m))


def ring_distances(ring_m=C.RING_WIDTH_M, max_m=C.RING_MAX_M):
    """Mid-point distance of each ring in metres (the last ring may be partial)."""
    return (np.arange(1, n_rings_for(ring_m, max_m) + 1) - 0.5) * ring_m


def decay(d, lst_inf, dT, ell):
    return lst_inf - dT * np.exp(-d / ell)


def fit_decay(d, profile, min_rings=10):
    """Fit LST(d) = LST_inf - dT exp(-d/ell). Returns (lst_inf, dT, ell) or None."""
    ok = np.isfinite(profile)
    if ok.sum() < min_rings:
        return None
    d, y = d[ok], profile[ok]
    p0 = (y[-5:].mean(), max(y[-5:].mean() - y[0], 0.1), 100.0)
    try:
        popt, _ = curve_fit(decay, d, y, p0=p0, bounds=([-50, 0, 10], [90, 20, 3_000]), maxfev=5_000)
    except (RuntimeError, ValueError):
        return None
    return tuple(float(v) for v in popt)


def observed_cooling(d, profile, far=C.FAR_FIELD_M):
    """Cooling relative to the far-field mean, c(d) = mean LST(far) - LST(d)."""
    far_mask = (d >= far[0]) & (d <= far[1]) & np.isfinite(profile)
    if not far_mask.any():
        return np.full_like(profile, np.nan)
    return profile[far_mask].mean() - profile


def extent_m(dT, ell, threshold=C.EXTENT_THRESHOLD_C):
    """Distance where predicted cooling drops below `threshold` (0 if it starts below)."""
    dT, ell = np.asarray(dT, dtype=float), np.asarray(ell, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dT > threshold, ell * np.log(dT / threshold), 0.0)


def theta_design(ln_area, ndvi, shape, built):
    return np.column_stack([np.ones_like(ln_area), ln_area, ndvi, shape, built])


def phi_design(ln_area, ndvi):
    return np.column_stack([np.ones_like(ln_area), ln_area, ndvi])


def _ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def fit_params(sites, n_boot=C.N_BOOTSTRAP, seed=0):
    """Fit the cross-site regressions and a pairs bootstrap with one residual draw per sample.

    `sites` needs keys ln_area, ndvi, shape, built, dT, ell (arrays). Adding a residual
    from the same bootstrap fit turns the coefficient spread into a prediction interval.
    """
    Xt = theta_design(sites["ln_area"], sites["ndvi"], sites["shape"], sites["built"])
    Xp = phi_design(sites["ln_area"], sites["ndvi"])
    yt, yp = sites["dT"], np.log(sites["ell"])
    theta, _ = _ols(Xt, yt)
    phi, _ = _ols(Xp, yp)

    rng = np.random.default_rng(seed)
    n = len(yt)
    boot = {"theta": [], "phi": [], "eps_dT": [], "eps_lnl": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bt, rt = _ols(Xt[idx], yt[idx])
        bp, rp = _ols(Xp[idx], yp[idx])
        boot["theta"].append(bt.tolist())
        boot["phi"].append(bp.tolist())
        boot["eps_dT"].append(float(rng.choice(rt)))
        boot["eps_lnl"].append(float(rng.choice(rp)))
    return {
        "theta_features": THETA_FEATURES,
        "phi_features": PHI_FEATURES,
        "theta": theta.tolist(),
        "phi": phi.tolist(),
        "boot": boot,
        "ndvi_soil": NDVI_SOIL,
        "extent_threshold_c": C.EXTENT_THRESHOLD_C,
        "n_sites": int(n),
    }


def ndvi_scale(ndvi_target, ndvi_now, ndvi_soil=NDVI_SOIL):
    """Share of a mature site's cooling a planting can deliver, from its NDVI gain."""
    denom = ndvi_target - ndvi_soil
    if denom <= 0:
        return 0.0
    return float(np.clip((ndvi_target - ndvi_now) / denom, 0.0, 1.0))


def predict(params, area_m2, ndvi_target, ndvi_now, shape, built, q=(5, 50, 95)):
    """Predicted boundary cooling, decay length and extent for one polygon.

    The point estimate uses the full-data fit; the quantiles come from the bootstrap.
    The JS service (api/src/cooling.js) mirrors this function exactly.
    """
    ln_area = math.log(area_m2)
    xt = np.array([1.0, ln_area, ndvi_target, shape, built])
    xp = np.array([1.0, ln_area, ndvi_target])
    s = ndvi_scale(ndvi_target, ndvi_now, params["ndvi_soil"])
    b = params["boot"]
    dT_s = np.maximum((np.array(b["theta"]) @ xt + np.array(b["eps_dT"])) * s, 0.0)
    ell_s = np.exp(np.array(b["phi"]) @ xp + np.array(b["eps_lnl"]))
    dT = max(float(np.dot(params["theta"], xt)) * s, 0.0)
    ell = float(math.exp(np.dot(params["phi"], xp)))
    thr = params["extent_threshold_c"]
    return {
        "scale": s,
        "dT": dT,
        "ell": ell,
        "extent_m": float(extent_m(dT, ell, thr)),
        "dT_q": [float(v) for v in np.percentile(dT_s, q)],
        "ell_q": [float(v) for v in np.percentile(ell_s, q)],
        "extent_q": [float(v) for v in np.percentile(extent_m(dT_s, ell_s, thr), q)],
    }


def save_params(params, path):
    with open(path, "w") as f:
        json.dump(params, f)
