"""GeoTIFF read/write helpers. Earth Engine may shard large exports into several files."""
import glob
import os

import numpy as np
import rasterio
from rasterio.merge import merge

from . import config as C


def find(data_dir, name):
    paths = sorted(glob.glob(os.path.join(data_dir, f"{name}.tif")) + glob.glob(os.path.join(data_dir, f"{name}-*.tif")))
    if not paths:
        raise FileNotFoundError(f"no export named {name} in {data_dir}")
    return paths


def read(data_dir, name, dtype="float32"):
    """Returns (array [bands, rows, cols], profile). Nodata becomes NaN for float output."""
    paths = find(data_dir, name)
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            arr, profile = src.read(masked=True), src.profile
    else:
        srcs = [rasterio.open(p) for p in paths]
        arr, transform = merge(srcs, masked=True)
        profile = srcs[0].profile
        profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
        for s in srcs:
            s.close()
    if np.dtype(dtype).kind == "f":
        return arr.astype(dtype).filled(np.nan), profile
    return arr.filled(0).astype(dtype), profile


def write(path, arr, profile, dtype=None, nodata=None):
    arr = arr[None] if arr.ndim == 2 else arr
    p = dict(profile)
    p.update(count=arr.shape[0], dtype=dtype or arr.dtype.name, compress="deflate", driver="GTiff")
    if nodata is not None:
        p["nodata"] = nodata
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(p["dtype"]))


def study_window_30m():
    """Row/col slice of the study grid inside the (larger) 30 m thermal grid."""
    s, t = C.study_grid(), C.thermal_grid()
    r0, c0 = int(round((t.y0 - s.y0) / 30)), int(round((s.x0 - t.x0) / 30))
    rows, cols = s.shape(30)
    return slice(r0, r0 + rows), slice(c0, c0 + cols)


def profile_for(grid, res, profile):
    from affine import Affine

    rows, cols = grid.shape(res)
    p = dict(profile)
    p.update(height=rows, width=cols, transform=Affine(*grid.transform(res)), crs=C.CRS)
    return p
