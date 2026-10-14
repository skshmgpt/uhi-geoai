"""A synthetic city with a known cooling law, for tests and for smoke-testing the full stack.

    python -m uhi.synthetic --out data_demo     # writes rasters for year 2099 (never a real year)

Nothing produced here is a result. The grid sits inside Delhi's UTM zone so the database
and map work unchanged.
"""
import argparse

import numpy as np
from affine import Affine
from scipy import ndimage

from . import config as C
from . import rio

STUDY = C.Grid(x0=700_000.0, y0=3_160_000.0, width_m=9_000.0, height_m=9_000.0)
THERMAL = C.Grid(x0=698_500.0, y0=3_161_500.0, width_m=12_000.0, height_m=12_000.0)
YEAR = 2099
OFFSET = 50  # study window inside the thermal grid, in 30 m pixels (1.5 km)

# The planted law: dT = 0.5 ln A + 3 NDVI - 5;  ln ell = 3 + 0.2 ln A.
TRUE_THETA = {"ln_area": 0.5, "ndvi": 3.0, "const": -5.0}
TRUE_PHI = {"const": 3.0, "ln_area": 0.2}


def make_city(out_dir, seed=7, n_tries=60):
    rng = np.random.default_rng(seed)
    rows10, cols10 = STUDY.shape(10)
    rows30, cols30 = THERMAL.shape(30)
    classes = np.full((rows10, cols10), C.IMPERVIOUS, np.uint8)
    classes[rng.random((rows10, cols10)) < 0.4] = C.BUILT
    cooling_field = np.zeros((rows30, cols30))
    ndvi30 = np.full((rows30, cols30), 0.12)

    placed = []
    for _ in range(n_tries):
        side = int(rng.integers(3, 12))
        r, c = rng.integers(2, STUDY.shape(30)[0] - side - 2, 2)
        if any(abs(r - pr) < 40 and abs(c - pc) < 40 for pr, pc in placed):
            continue
        placed.append((r, c))
        v = float(rng.uniform(0.45, 0.8))
        classes[r * 3:(r + side) * 3, c * 3:(c + side) * 3] = C.CANOPY
        site = np.zeros((rows30, cols30), bool)
        site[OFFSET + r:OFFSET + r + side, OFFSET + c:OFFSET + c + side] = True
        ln_a = np.log(side * side * 900.0)
        dT = TRUE_THETA["ln_area"] * ln_a + TRUE_THETA["ndvi"] * v + TRUE_THETA["const"]
        ell = np.exp(TRUE_PHI["const"] + TRUE_PHI["ln_area"] * ln_a)
        d = ndimage.distance_transform_edt(~site) * 30 - 15
        cooling_field = np.maximum(cooling_field, np.where(site, dT, dT * np.exp(-np.maximum(d, 0) / ell)))
        ndvi30[site] = v

    # Rural ring ~40 C, city ~41.5 C, a few hot industrial blobs +3 C.
    lst30 = np.full((rows30, cols30), 40.0)
    win = (slice(OFFSET, OFFSET + STUDY.shape(30)[0]), slice(OFFSET, OFFSET + STUDY.shape(30)[1]))
    lst30[win] = 41.5
    yy, xx = np.indices((rows30, cols30))
    for _ in range(6):
        cy, cx = rng.integers(OFFSET + 20, OFFSET + 280, 2)
        lst30 += 3.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0**2))
    lst30 = lst30 - cooling_field + rng.normal(0, 0.15, (rows30, cols30))

    worldcover = np.full((rows30, cols30), C.WC_CROP, np.uint8)
    worldcover[win] = C.WC_BUILT

    base = {"driver": "GTiff", "crs": C.CRS, "count": 1}
    p10 = dict(base, height=rows10, width=cols10, transform=Affine(*STUDY.transform(10)))
    p30 = dict(base, height=rows30, width=cols30, transform=Affine(*THERMAL.transform(30)))
    rio.write(f"{out_dir}/classes_{YEAR}.tif", classes, p10, "uint8")
    rio.write(f"{out_dir}/lst_{YEAR}.tif", lst30.astype(np.float32), p30, "float32")
    rio.write(f"{out_dir}/ndvi30_{YEAR}.tif", ndvi30.astype(np.float32), p30, "float32")
    rio.write(f"{out_dir}/worldcover30.tif", worldcover, p30, "uint8")
    return len(placed)


def use_synthetic_grids(setter):
    """Point config at the synthetic grids. `setter(obj, name, value)` is setattr or monkeypatch.setattr."""
    setter(C, "study_grid", lambda: STUDY)
    setter(C, "thermal_grid", lambda: THERMAL)
    setter(C, "RURAL_RING_M", (300, 1_500))  # the synthetic border is only 1.5 km wide


def main():
    import os

    from . import process

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_demo")
    p.add_argument("--results", default="results_demo")
    p.add_argument("--layers", default="../web/public/layers")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    use_synthetic_grids(setattr)
    n = make_city(a.out)
    print(f"synthetic city: {n} parks (year {YEAR}, not real data)")
    process.run_calibrate(a.out, YEAR, f"classes_{YEAR}", a.results, n_boot=200)
    process.run_hotspots(a.out, YEAR, C.HOTSPOT_MIN_INTENSITY_C, a.results)
    process.run_overlay(a.out, YEAR, a.layers)


if __name__ == "__main__":
    main()
