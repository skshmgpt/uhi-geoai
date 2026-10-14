"""Study-area, grid and threshold settings shared by every pipeline stage."""
import os
from dataclasses import dataclass

EE_PROJECT = os.environ.get("EE_PROJECT", "ecotrace-450710")
DRIVE_FOLDER = os.environ.get("UHI_DRIVE_FOLDER", "uhi-geoai")

# Delhi NCT, approximate bounding box (lon_min, lat_min, lon_max, lat_max).
DELHI_BBOX = (76.84, 28.40, 77.35, 28.89)
CRS = "EPSG:32643"  # UTM 43N, metres
RURAL_RING_M = (5_000, 15_000)  # rural reference ring outside the study box

# Pre-monsoon analysis window.
SEASON = ("04-01", "07-01")  # end exclusive: 1 April to 30 June
MAX_SCENE_CLOUD_PCT = 15
CLOUD_SCORE_MIN = 0.60

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# Segmentation classes. 255 = ignore (grass, crop, bare soil, no data).
CLASSES = ["impervious", "built", "canopy", "water"]
IMPERVIOUS, BUILT, CANOPY, WATER = range(4)
IGNORE = 255

# ESA WorldCover v200 class codes.
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP, WC_BUILT, WC_BARE, WC_WATER = 10, 20, 30, 40, 50, 60, 80
RURAL_WC_CLASSES = (WC_SHRUB, WC_GRASS, WC_CROP)

# Hotspot thresholds from the synopsis; varied in experiment E5.
HOTSPOT_MIN_INTENSITY_C = 2.0
HOTSPOT_MIN_AREA_M2 = 5_000
HOTSPOT_DBSCAN_EPS_M = 60
# Hotspots count only urban pixels: Built + Impervious share of the 30 m cell, from the segmentation.
# Without it, dry bare fields on the fringe outrank the city in the pre-monsoon morning.
HOTSPOT_URBAN_MIN_FRACTION = 0.5

# Cooling model.
SITE_MIN_AREA_M2 = 5_000  # 0.5 ha
SITE_CANOPY_FRACTION = 0.5
RING_WIDTH_M = 30
RING_MAX_M = 1_000
FAR_FIELD_M = (800, 1_000)
BUILT_BUFFER_M = 300
EXTENT_THRESHOLD_C = 0.5
ELL_BOUNDS_M = (10, 3_000)  # decay-length search range for the per-site fit
ELL_BOUND_MARGIN = 0.02  # fits within 2% of either bound are treated as unidentified
N_BOOTSTRAP = 1_000


@dataclass(frozen=True)
class Grid:
    """A north-up UTM grid; the 10 m and 30 m grids share one origin so 3x3 blocks align."""

    x0: float
    y0: float
    width_m: float
    height_m: float

    def transform(self, res):
        return [res, 0, self.x0, 0, -res, self.y0]

    def shape(self, res):
        return int(self.height_m // res), int(self.width_m // res)


def _utm_bounds(bbox, buffer_m=0.0):
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = zip(*(t.transform(x, y) for x in bbox[::2] for y in bbox[1::2]))
    return min(xs) - buffer_m, min(ys) - buffer_m, max(xs) + buffer_m, max(ys) + buffer_m


def grid(buffer_m=0.0, snap=30.0):
    """Grid covering the Delhi box plus `buffer_m`, snapped outward to `snap` metres."""
    import math

    xmin, ymin, xmax, ymax = _utm_bounds(DELHI_BBOX, buffer_m)
    x0, y1 = math.floor(xmin / snap) * snap, math.ceil(ymax / snap) * snap
    x1, y0 = math.ceil(xmax / snap) * snap, math.floor(ymin / snap) * snap
    return Grid(x0=x0, y0=y1, width_m=x1 - x0, height_m=y1 - y0)


def study_grid():
    return grid(0.0)


def thermal_grid():
    """Thermal products extend over the rural ring so the reference can be computed."""
    return grid(RURAL_RING_M[1])
