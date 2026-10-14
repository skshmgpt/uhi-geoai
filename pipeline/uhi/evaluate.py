"""Experiments E4 (LST accuracy) and E6 (cooling model, spatially blocked CV).

    python -m uhi.evaluate lst --data data --year 2024 --classes classes_2024
    python -m uhi.evaluate cooling --results results --year 2024

Each writes a JSON file to results/ that the paper's tables are filled from.
"""
import argparse
import csv
import json
import os

import numpy as np

from . import config as C
from . import cooling, rio


def error_stats(pred, ref):
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok], ref[ok]
    e = p - r
    ss_res, ss_tot = np.sum(e**2), np.sum((r - r.mean()) ** 2)
    return {
        "bias": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(ok.sum()),
    }


def eval_lst(data, year, classes_name, out_dir):
    ours, _ = rio.read(data, f"lst_{year}")
    ref, _ = rio.read(data, f"st_{year}")
    win = rio.study_window_30m()
    o, r = ours[0][win], ref[0][win]
    classes, _ = rio.read(data, classes_name, dtype="uint8")
    c = classes[0]
    rows, cols = c.shape[0] // 3, c.shape[1] // 3
    o, r = o[:rows, :cols], r[:rows, :cols]
    built = cooling.block_fraction((c == C.BUILT) | (c == C.IMPERVIOUS)) >= 0.5
    canopy = cooling.block_fraction(c == C.CANOPY) >= 0.5
    all_ = error_stats(o, r)
    res = {
        "year": year,
        "all": all_,
        "built_impervious": error_stats(np.where(built, o, np.nan), r),
        "canopy": error_stats(np.where(canopy, o, np.nan), r),
        "debiased": error_stats(o - all_["bias"], r),
    }
    _dump(res, out_dir, f"e4_lst_{year}.json")


def spatial_blocks(x, y, block_m=2_000):
    bx, by = np.floor(np.asarray(x) / block_m).astype(int), np.floor(np.asarray(y) / block_m).astype(int)
    return bx * 100_000 + by


def _load_sites(results, year):
    with open(os.path.join(results, f"sites_{year}.csv")) as f:
        rows = list(csv.DictReader(f))
    s = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    s["ln_area"] = np.log(s["area_m2"])
    z = np.load(os.path.join(results, f"profiles_{year}.npz"))
    return s, z["d"], z["profiles"]


def _subset(s, idx):
    return {k: v[idx] for k, v in s.items()}


def eval_cooling(results, year, n_splits=10, n_boot=200, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    s, d, profiles = _load_sites(results, year)
    obs = np.array([cooling.observed_cooling(d, p) for p in profiles])
    groups = spatial_blocks(s["x"], s["y"])
    n_splits = min(n_splits, len(np.unique(groups)))
    models = ["distance_only", "linear_synopsis", "decay_linear", "decay_gbt"]
    prof_err = {m: [] for m in models}
    dT_err = {m: [] for m in models[2:]}
    ext_err = {m: [] for m in models[2:]}
    covered = []
    obs_extent = cooling.extent_m(s["dT"], s["ell"])

    for train, test in GroupKFold(n_splits=n_splits).split(s["x"], groups=groups):
        tr, te = _subset(s, train), _subset(s, test)
        o_tr, o_te = obs[train], obs[test]

        # Baseline 1: the mean observed profile of training sites, ignoring site traits.
        mean_prof = np.nanmean(o_tr, axis=0)
        prof_err["distance_only"].append(np.abs(mean_prof[None] - o_te))

        # Baseline 2: the synopsis form, c = t0 + t1 ln A + t2 NDVI + t3 d, pooled over rings.
        def lin_x(ss):
            n = len(ss["ln_area"])
            return np.column_stack([
                np.ones(n * len(d)), np.repeat(ss["ln_area"], len(d)), np.repeat(ss["ndvi"], len(d)), np.tile(d, n),
            ])
        y_tr = o_tr.ravel()
        ok = np.isfinite(y_tr)
        coef, *_ = np.linalg.lstsq(lin_x(tr)[ok], y_tr[ok], rcond=None)
        prof_err["linear_synopsis"].append(np.abs((lin_x(te) @ coef).reshape(o_te.shape) - o_te))

        # Decay models: predict (dT, ell) from traits, then the profile dT exp(-d/ell).
        Xt_tr = cooling.theta_design(tr["ln_area"], tr["ndvi"], tr["shape"], tr["built"])
        Xt_te = cooling.theta_design(te["ln_area"], te["ndvi"], te["shape"], te["built"])
        Xp_tr, Xp_te = cooling.phi_design(tr["ln_area"], tr["ndvi"]), cooling.phi_design(te["ln_area"], te["ndvi"])

        params = cooling.fit_params(tr, n_boot=n_boot, seed=seed)
        lin_dT = np.maximum(Xt_te @ np.array(params["theta"]), 0)
        lin_ell = np.exp(Xp_te @ np.array(params["phi"]))
        boot_dT = np.maximum(Xt_te @ np.array(params["boot"]["theta"]).T + np.array(params["boot"]["eps_dT"]), 0)
        lo, hi = np.percentile(boot_dT, [5, 95], axis=1)
        covered.append((te["dT"] >= lo) & (te["dT"] <= hi))

        gb_dT = GradientBoostingRegressor(random_state=seed).fit(Xt_tr[:, 1:], tr["dT"]).predict(Xt_te[:, 1:])
        gb_ell = np.exp(GradientBoostingRegressor(random_state=seed).fit(Xp_tr[:, 1:], np.log(tr["ell"])).predict(Xp_te[:, 1:]))
        gb_dT = np.maximum(gb_dT, 0)

        for name, pdT, pell in (("decay_linear", lin_dT, lin_ell), ("decay_gbt", gb_dT, gb_ell)):
            curve = pdT[:, None] * np.exp(-d[None] / pell[:, None])
            prof_err[name].append(np.abs(curve - o_te))
            dT_err[name].append(np.abs(pdT - te["dT"]))
            ext_err[name].append(np.abs(cooling.extent_m(pdT, pell) - obs_extent[test]))

    out = {"year": year, "n_sites": int(len(s["dT"])), "n_folds": n_splits, "block_m": 2_000, "models": {}}
    for m in models:
        row = {"profile_mae_c": float(np.nanmean(np.concatenate([e.ravel() for e in prof_err[m]])))}
        if m in dT_err:
            row["dT_mae_c"] = float(np.mean(np.concatenate(dT_err[m])))
            row["extent_mae_m"] = float(np.mean(np.concatenate(ext_err[m])))
        out["models"][m] = row
    out["models"]["decay_linear"]["pi90_coverage"] = float(np.mean(np.concatenate(covered)))
    out["observed"] = {
        "mean_dT_c": float(s["dT"].mean()),
        "dT_ci95": [float(v) for v in _boot_ci(s["dT"], seed)],
        "median_dT_c": float(np.median(s["dT"])),
        "mean_extent_m": float(obs_extent.mean()),
        "extent_ci95": [float(v) for v in _boot_ci(obs_extent, seed)],
        "median_extent_m": float(np.median(obs_extent)),
        "extent_censored_share": float(np.mean(obs_extent >= C.RING_MAX_M)),
        "extent_cap_m": C.RING_MAX_M,
    }
    _dump(out, results, f"e6_cooling_{year}.json")


def _boot_ci(x, seed, n=10_000):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _dump(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(json.dumps(obj, indent=2))
    print(f"-> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["lst", "cooling"])
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--classes", default=None)
    a = p.parse_args()
    if a.experiment == "lst":
        eval_lst(a.data, a.year, a.classes or f"classes_{a.year}", a.results)
    else:
        eval_cooling(a.results, a.year)


if __name__ == "__main__":
    main()
