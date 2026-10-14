"""Local processing steps on the Drive exports (run on your laptop or in Colab).

    python -m uhi.process lst --data data --year 2024
    python -m uhi.process hotspots --data data --year 2024 [--min-intensity 2.0]
    python -m uhi.process calibrate --data data --year 2024 [--classes classes_2024]
    python -m uhi.process overlay --data data --year 2024 --out ../web/public/layers

Outputs go next to the inputs in --data unless stated.
"""
import argparse
import csv
import json
import os

import numpy as np

from . import config as C
from . import cooling, hotspots, lst, rio


def run_lst(data, year, classes_name=None):
    bt, prof = rio.read(data, f"bt_{year}")
    ndvi, _ = rio.read(data, f"ndvi30_{year}")
    water = None
    if classes_name:
        # Water emissivity ablation: only defined inside the study box.
        classes, _ = rio.read(data, classes_name, dtype="uint8")
        water = np.zeros(bt[0].shape, bool)
        water[rio.study_window_30m()] = cooling.block_fraction(classes[0] == C.WATER) >= 0.5
    eps = lst.emissivity(ndvi[0], water)
    out = lst.lst_celsius(bt[0], eps).astype(np.float32)
    name = f"lst_{year}" + ("_water" if water is not None else "")
    rio.write(os.path.join(data, f"{name}.tif"), out, prof, "float32", nodata=np.nan)
    print(f"wrote {name}.tif  median {np.nanmedian(out):.2f} C")


def run_hotspots(data, year, min_intensity, out_dir, urban_min=C.HOTSPOT_URBAN_MIN_FRACTION):
    """Needs builtfrac30_{year} from the calibrate step unless urban_min is 0."""
    t_lst, prof = rio.read(data, f"lst_{year}")
    wc, _ = rio.read(data, "worldcover30", dtype="uint8")
    transform = list(prof["transform"])[:6]
    rural = hotspots.rural_mask(t_lst[0].shape, transform, C.study_grid(), wc[0])
    inten, ref = hotspots.intensity(t_lst[0], rural)
    win = rio.study_window_30m()
    study_inten = np.full_like(inten, np.nan)
    study_inten[win] = inten[win]  # hotspots only inside the study box
    if urban_min > 0:
        urban, _ = rio.read(data, f"builtfrac30_{year}")
        study_inten[win] = hotspots.urban_only(study_inten[win], urban[0], urban_min)
    rows = hotspots.candidates(study_inten, transform, min_intensity)
    sql = hotspots.to_sql(rows, year, min_intensity)
    path = os.path.join(out_dir, f"hotspot_candidates_{year}_{min_intensity}.sql")
    with open(path, "w") as f:
        f.write(sql)
    rio.write(os.path.join(data, f"suhi_{year}.tif"), inten.astype(np.float32), prof, "float32", nodata=np.nan)
    print(f"rural reference {ref:.2f} C from {int(rural.sum())} px; urban mask >= {urban_min}; "
          f"{len(rows)} candidate regions -> {path}")


def calibration_inputs(data, year, classes_name):
    classes, cprof = rio.read(data, classes_name, dtype="uint8")
    win = rio.study_window_30m()
    t_lst, tprof = rio.read(data, f"lst_{year}")
    ndvi, _ = rio.read(data, f"ndvi30_{year}")
    c = classes[0]
    canopy = cooling.block_fraction(c == C.CANOPY)
    water = cooling.block_fraction(c == C.WATER)
    built = cooling.block_fraction((c == C.BUILT) | (c == C.IMPERVIOUS))
    rows, cols = canopy.shape
    crop = lambda a: a[win][:rows, :cols]  # noqa: E731
    return {
        "lst": crop(t_lst[0]),
        "ndvi": crop(ndvi[0]),
        "canopy": canopy,
        "water": water,
        "built": built,
        "profile30": rio.profile_for(C.study_grid(), 30, tprof),
    }


def run_calibrate(data, year, classes_name, out_dir, n_boot=C.N_BOOTSTRAP):
    x = calibration_inputs(data, year, classes_name)
    sites, n = cooling.label_sites(x["canopy"])
    exclude = (x["water"] >= 0.5) | (x["canopy"] >= C.SITE_CANOPY_FRACTION)
    profs = cooling.site_profiles(x["lst"], sites, n, x["ndvi"], x["built"], exclude)
    d = cooling.ring_distances()
    t = x["profile30"]["transform"]
    kept, profiles = [], []
    counts = {"candidates": n, "no_fit": 0, "ell_at_lower_bound": 0, "ell_at_upper_bound": 0}
    for s in profs:
        fit = cooling.fit_decay(d, s["profile"])
        if fit is None or not np.isfinite(s["built"]):
            counts["no_fit"] += 1
            continue
        lst_inf, dT, ell = fit
        if not cooling.ell_identified(ell):
            counts["ell_at_upper_bound" if ell > C.ELL_BOUNDS_M[0] * 10 else "ell_at_lower_bound"] += 1
            continue
        s.update(lst_inf=lst_inf, dT=dT, ell=ell, x=t.c + (s["col"] + 0.5) * t.a, y=t.f + (s["row"] + 0.5) * t.e)
        kept.append(s)
        profiles.append(s["profile"])
    counts["kept"] = len(kept)
    print(f"{n} candidate sites, {len(kept)} usable fits ({counts})")

    os.makedirs(out_dir, exist_ok=True)
    keys = ["site_id", "x", "y", "area_m2", "ndvi", "shape", "built", "lst_inf", "dT", "ell"]
    with open(os.path.join(out_dir, f"sites_{year}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    np.savez_compressed(os.path.join(out_dir, f"profiles_{year}.npz"), d=d, profiles=np.array(profiles))

    arr = {k: np.array([s[k] for s in kept]) for k in ("ndvi", "shape", "built", "dT", "ell")}
    arr["ln_area"] = np.log([s["area_m2"] for s in kept])
    params = cooling.fit_params(arr, n_boot=n_boot)
    params["year"] = year
    params["fit_counts"] = counts
    cooling.save_params(params, os.path.join(out_dir, f"cooling_params_{year}.json"))

    prof = x["profile30"]
    rio.write(os.path.join(data, f"builtfrac30_{year}.tif"), x["built"], prof, "float32", nodata=np.nan)
    rio.write(os.path.join(data, f"ndvi30_study_{year}.tif"), x["ndvi"], prof, "float32", nodata=np.nan)
    print(f"theta {np.round(params['theta'], 3)}  phi {np.round(params['phi'], 3)}")


def run_overlay(data, year, out_dir, vmin=None, vmax=None):
    """Colour LST as a Web Mercator PNG plus lat/lon bounds for a Leaflet imageOverlay."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    from PIL import Image
    from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    t_lst, prof = rio.read(data, f"lst_{year}")
    win = rio.study_window_30m()
    src = t_lst[0][win]
    sprof = rio.profile_for(C.study_grid(), 30, prof)
    dst_crs = "EPSG:3857"
    tr, w, h = calculate_default_transform(C.CRS, dst_crs, sprof["width"], sprof["height"], *_bounds(sprof))
    dst = np.full((h, w), np.nan, np.float32)
    reproject(src, dst, src_transform=sprof["transform"], src_crs=C.CRS, dst_transform=tr, dst_crs=dst_crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    lo = vmin if vmin is not None else float(np.nanpercentile(dst, 2))
    hi = vmax if vmax is not None else float(np.nanpercentile(dst, 98))
    rgba = (colormaps["inferno"](np.clip((dst - lo) / (hi - lo), 0, 1)) * 255).astype(np.uint8)
    rgba[~np.isfinite(dst), 3] = 0
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgba).save(os.path.join(out_dir, f"lst_{year}.png"))
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", tr.c, tr.f + h * tr.e, tr.c + w * tr.a, tr.f)
    meta = {"year": year, "bounds": [[south, west], [north, east]], "min_c": lo, "max_c": hi, "colormap": "inferno"}
    with open(os.path.join(out_dir, f"lst_{year}.json"), "w") as f:
        json.dump(meta, f)
    print(f"overlay {w}x{h}, {lo:.1f}-{hi:.1f} C")


def _bounds(p):
    t = p["transform"]
    return t.c, t.f + p["height"] * t.e, t.c + p["width"] * t.a, t.f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["lst", "hotspots", "calibrate", "overlay"])
    p.add_argument("--data", default="data")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--classes", help="class raster name, e.g. classes_2024 or labels_weak")
    p.add_argument("--min-intensity", type=float, default=C.HOTSPOT_MIN_INTENSITY_C)
    p.add_argument("--urban-min", type=float, default=C.HOTSPOT_URBAN_MIN_FRACTION,
                   help="minimum Built + Impervious share for a hotspot pixel; 0 disables the mask")
    p.add_argument("--out", default="results")
    p.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAP)
    a = p.parse_args()
    if a.step == "lst":
        run_lst(a.data, a.year, a.classes)
    elif a.step == "hotspots":
        os.makedirs(a.out, exist_ok=True)
        run_hotspots(a.data, a.year, a.min_intensity, a.out, a.urban_min)
    elif a.step == "calibrate":
        run_calibrate(a.data, a.year, a.classes or f"classes_{a.year}", a.out, a.n_boot)
    else:
        run_overlay(a.data, a.year, a.out)


if __name__ == "__main__":
    main()
