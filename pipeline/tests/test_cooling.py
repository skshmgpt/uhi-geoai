import json
import math
import os

import numpy as np
import pytest

from uhi import config as C
from uhi import cooling

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "cooling_case.json")


def test_fit_decay_recovers_known_curve():
    d = cooling.ring_distances()
    y = cooling.decay(d, 42.0, 2.5, 150.0) + np.random.default_rng(0).normal(0, 0.05, d.size)
    lst_inf, dT, ell = cooling.fit_decay(d, y)
    assert abs(lst_inf - 42) < 0.1 and abs(dT - 2.5) < 0.15 and abs(ell - 150) < 20


def test_fit_decay_rejects_sparse_profiles():
    d = cooling.ring_distances()
    y = np.full(d.size, np.nan)
    y[:5] = 40
    assert cooling.fit_decay(d, y) is None


def test_extent_formula():
    assert np.isclose(cooling.extent_m(2.0, 100.0), 100 * math.log(4))
    assert cooling.extent_m(0.3, 100.0) == 0


def test_block_fraction():
    m = np.zeros((6, 6), bool)
    m[:3, :3] = True
    m[3, 3] = True
    np.testing.assert_allclose(cooling.block_fraction(m), [[1, 0], [0, 1 / 9]])


def test_square_site_has_shape_index_one():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert np.isclose(cooling.site_shape_index(m), 1.0, atol=0.02)


def test_label_sites_drops_small_patches():
    frac = np.zeros((30, 30))
    frac[2:12, 2:12] = 1.0   # 100 px = 9 ha
    frac[20:22, 20:22] = 1.0  # 4 px = 0.36 ha, below 0.5 ha
    sites, n = cooling.label_sites(frac)
    assert n == 1 and (sites > 0).sum() == 100


def test_profiles_recover_synthetic_park_cooling():
    """A square park in a 45 C field that cools its surroundings by 3 exp(-d/120)."""
    n = 120
    sites = np.zeros((n, n), np.int32)
    sites[50:70, 50:70] = 1
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(sites == 0) * cooling.PIXEL_M - cooling.PIXEL_M / 2  # edge distance
    lst_field = 45.0 - 3.0 * np.exp(-dist / 120.0)
    ndvi = np.where(sites > 0, 0.7, 0.15)
    built = np.full((n, n), 0.6)
    profs = cooling.site_profiles(lst_field, sites, 1, ndvi, built, exclude=np.zeros((n, n), bool))
    _, dT, ell = cooling.fit_decay(cooling.ring_distances(), profs[0]["profile"])
    assert abs(dT - 3.0) < 0.3 and abs(ell - 120) < 15
    assert profs[0]["area_m2"] == 400 * 900 and np.isclose(profs[0]["ndvi"], 0.7)


def synthetic_sites(n=200, seed=1):
    rng = np.random.default_rng(seed)
    s = {
        "ln_area": np.log(rng.uniform(5e3, 2e5, n)),
        "ndvi": rng.uniform(0.35, 0.8, n),
        "shape": rng.uniform(1.0, 2.5, n),
        "built": rng.uniform(0.2, 0.9, n),
    }
    s["dT"] = 0.4 * s["ln_area"] + 2.0 * s["ndvi"] - 0.3 * s["shape"] + 0.5 * s["built"] - 3.0 + rng.normal(0, 0.2, n)
    s["ell"] = np.exp(2.0 + 0.25 * s["ln_area"] + 0.5 * s["ndvi"] + rng.normal(0, 0.1, n))
    return s


def test_fit_params_recovers_coefficients():
    p = cooling.fit_params(synthetic_sites(n=3000), n_boot=50)
    np.testing.assert_allclose(p["theta"], [-3.0, 0.4, 2.0, -0.3, 0.5], atol=0.25)
    np.testing.assert_allclose(p["phi"], [2.0, 0.25, 0.5], atol=0.15)
    assert len(p["boot"]["theta"]) == 50


def test_ndvi_scale():
    assert cooling.ndvi_scale(0.6, 0.2) == 1.0
    assert np.isclose(cooling.ndvi_scale(0.6, 0.4), 0.5)
    assert cooling.ndvi_scale(0.6, 0.7) == 0.0
    assert cooling.ndvi_scale(0.15, 0.1) == 0.0


def test_predict_quantiles_are_ordered_and_scaled():
    p = cooling.fit_params(synthetic_sites(), n_boot=200)
    full = cooling.predict(p, 2e4, 0.7, 0.15, 1.2, 0.6)
    half = cooling.predict(p, 2e4, 0.7, 0.45, 1.2, 0.6)
    assert full["dT_q"][0] <= full["dT_q"][1] <= full["dT_q"][2]
    assert np.isclose(half["dT"], full["dT"] * half["scale"] / full["scale"])


def test_fixture_matches_python_predict():
    """The same fixture is checked by the Node service (api/test/cooling.test.js)."""
    with open(FIXTURE) as f:
        case = json.load(f)
    for c in case["cases"]:
        got = cooling.predict(case["params"], **c["input"])
        for k, v in c["expected"].items():
            np.testing.assert_allclose(got[k], v, rtol=1e-9, atol=1e-9)


def test_extent_capped_at_ring_window():
    assert cooling.extent_m(5.0, 3_000.0) == C.RING_MAX_M
    assert cooling.extent_m(5.0, 3_000.0, max_m=np.inf) > 4_000


def test_ell_at_search_bounds_is_unidentified():
    lo, hi = C.ELL_BOUNDS_M
    assert not cooling.ell_identified(hi - 0.01)
    assert not cooling.ell_identified(lo + 0.0001)
    assert cooling.ell_identified(150.0)
