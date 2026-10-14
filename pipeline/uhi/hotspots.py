"""SUHI intensity and hotspot candidate polygons.

Python finds connected pixels above the intensity threshold; PostGIS then merges nearby
fragments with ST_ClusterDBSCAN, drops small clusters and ranks them (db/hotspots.sql).
"""
import numpy as np
from scipy import ndimage

from . import config as C


def rural_mask(shape, transform, study_grid, worldcover_30m, ring_m=None):
    """Rural reference: shrub/grass/crop pixels in a ring (default 5-15 km) outside the study box."""
    ring_m = ring_m or C.RURAL_RING_M
    rows, cols = np.indices(shape)
    x = transform[2] + (cols + 0.5) * transform[0]
    y = transform[5] + (rows + 0.5) * transform[4]
    x0, y1 = study_grid.x0, study_grid.y0
    x1, y0 = x0 + study_grid.width_m, y1 - study_grid.height_m
    dx = np.maximum.reduce([x0 - x, x - x1, np.zeros_like(x)])
    dy = np.maximum.reduce([y0 - y, y - y1, np.zeros_like(y)])
    dist = np.hypot(dx, dy)
    in_ring = (dist >= ring_m[0]) & (dist <= ring_m[1])
    return in_ring & np.isin(worldcover_30m, C.RURAL_WC_CLASSES)


def intensity(lst, rural):
    sample = lst[rural & np.isfinite(lst)]
    if sample.size == 0:
        raise ValueError("No rural reference pixels: check the WorldCover raster and the rural ring distances.")
    ref = float(np.median(sample))
    return lst - ref, ref


def candidates(intensity_c, transform, min_intensity=C.HOTSPOT_MIN_INTENSITY_C):
    """Connected regions with I >= threshold as (WKT polygon, mean intensity, area m2)."""
    from rasterio import features
    from affine import Affine
    from shapely.geometry import shape

    hot = np.isfinite(intensity_c) & (intensity_c >= min_intensity)
    labels, n = ndimage.label(hot, structure=np.ones((3, 3)))
    if n == 0:
        return []
    means = ndimage.mean(intensity_c, labels, index=np.arange(1, n + 1))
    aff = Affine(*transform[:6])
    out = []
    for geom, lab in features.shapes(labels.astype(np.int32), mask=hot, transform=aff):
        poly = shape(geom)
        out.append((poly.wkt, float(means[int(lab) - 1]), float(poly.area)))
    return out


def to_sql(rows, year, min_intensity, table="hotspot_candidates"):
    """INSERT statements for psql; geometries are EPSG:32643."""
    lines = [f"DELETE FROM {table} WHERE year = {year} AND min_intensity = {min_intensity};"]
    for wkt, mean_i, area in rows:
        lines.append(
            f"INSERT INTO {table} (year, min_intensity, mean_intensity, area_m2, geom) VALUES "
            f"({year}, {min_intensity}, {mean_i:.4f}, {area:.1f}, ST_GeomFromText('{wkt}', 32643));"
        )
    return "\n".join(lines) + "\n"
