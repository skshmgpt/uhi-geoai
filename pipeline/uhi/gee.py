"""Earth Engine composites and Drive exports.

    python -m uhi.gee export --year 2024          # seasonal products for one year
    python -m uhi.gee export-static               # WorldCover and weak labels (year-independent)
    python -m uhi.gee status

All exports land in the Drive folder `config.DRIVE_FOLDER`, on grids from config.py so the
10 m and 30 m rasters align exactly (3x3 blocks).
"""
import argparse

import ee

from . import config as C

QA_CLEAR_BITS = 0b11111  # QA_PIXEL bits 0-4: fill, dilated cloud, cirrus, cloud, cloud shadow
ST_SCALE, ST_OFFSET = 0.00341802, 149.0  # Collection 2 Level-2 ST_B10 to kelvin


def init():
    ee.Initialize(project=C.EE_PROJECT)


def region(grid):
    x0, y1 = grid.x0, grid.y0
    return ee.Geometry.Rectangle([x0, y1 - grid.height_m, x0 + grid.width_m, y1], C.CRS, False)


def season(year):
    return f"{year}-{C.SEASON[0]}", f"{year}-{C.SEASON[1]}"


def s2_composite(year, roi):
    """Cloud Score+ masked Sentinel-2 SR median, reflectance x 10000 as int16."""
    start, end = season(year)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.MAX_SCENE_CLOUD_PCT))
        .linkCollection(cs, ["cs_cdf"])
        .map(lambda im: im.updateMask(im.select("cs_cdf").gte(C.CLOUD_SCORE_MIN)).select(C.S2_BANDS))
    )
    return col.median().clamp(0, 10_000).toInt16(), col.size()


def _landsat(collection_ids, year, roi):
    start, end = season(year)
    col = ee.ImageCollection(collection_ids[0])
    for cid in collection_ids[1:]:
        col = col.merge(ee.ImageCollection(cid))
    return col.filterBounds(roi).filterDate(start, end).filter(ee.Filter.lt("CLOUD_COVER", C.MAX_SCENE_CLOUD_PCT))


def _clear(im):
    return im.select("QA_PIXEL").bitwiseAnd(QA_CLEAR_BITS).eq(0)


def _brightness_temperature(im):
    """Per-scene DN -> radiance -> BT (K) with that scene's own calibration constants."""
    rad = im.select("B10").multiply(ee.Number(im.get("RADIANCE_MULT_BAND_10"))).add(
        ee.Number(im.get("RADIANCE_ADD_BAND_10"))
    )
    k1 = ee.Image.constant(ee.Number(im.get("K1_CONSTANT_BAND_10")))
    k2 = ee.Image.constant(ee.Number(im.get("K2_CONSTANT_BAND_10")))
    bt = k2.divide(k1.divide(rad).add(1).log())
    return bt.rename("bt").updateMask(_clear(im)).toFloat()


def landsat_bt(year, roi):
    col = _landsat(["LANDSAT/LC08/C02/T1", "LANDSAT/LC09/C02/T1"], year, roi)
    return col.map(_brightness_temperature).median(), col.size()


def landsat_st(year, roi):
    """USGS Level-2 surface temperature in degrees C: the validation reference (E4)."""
    col = _landsat(["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"], year, roi)
    st = col.map(
        lambda im: im.select("ST_B10").multiply(ST_SCALE).add(ST_OFFSET - 273.15).rename("st").updateMask(_clear(im))
    )
    return st.median().toFloat(), col.size()


def worldcover():
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")


def weak_labels(roi):
    """10 m training labels: buildings -> built, other WorldCover built-up -> impervious,
    tree cover -> canopy, water -> water, everything else ignored."""
    wc = worldcover()
    buildings = (
        ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
        .filterBounds(roi)
        .filter(ee.Filter.gte("confidence", 0.75))
    )
    bld = ee.Image(0).byte().paint(buildings, 1)
    return (
        ee.Image.constant(C.IGNORE).byte()
        .where(wc.eq(C.WC_BUILT), C.IMPERVIOUS)
        .where(wc.eq(C.WC_TREE), C.CANOPY)
        .where(wc.eq(C.WC_WATER), C.WATER)
        .where(bld.eq(1), C.BUILT)
        .rename("label")
    )


def to_30m(img, reducer, grid):
    """Aggregate to the 30 m grid. Composites carry no native projection, so pin them to
    the aligned 10 m grid first; otherwise reduceResolution has nothing to reduce from."""
    return (
        img.reproject(C.CRS, grid.transform(10))
        .reduceResolution(reducer, maxPixels=64)
        .reproject(C.CRS, grid.transform(30))
    )


def export(img, name, grid, res):
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=name,
        folder=C.DRIVE_FOLDER,
        fileNamePrefix=name,
        region=region(grid),
        crs=C.CRS,
        crsTransform=grid.transform(res),
        maxPixels=1e10,
        fileDimensions=[16384, 16384],
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"started {name}: {task.id}")
    return task


def export_year(year):
    study, thermal = C.study_grid(), C.thermal_grid()
    s2, n_s2 = s2_composite(year, region(thermal))
    bt, n_bt = landsat_bt(year, region(thermal))
    st, n_st = landsat_st(year, region(thermal))
    print(f"{year}: {n_s2.getInfo()} Sentinel-2 scenes, {n_bt.getInfo()} Landsat L1, {n_st.getInfo()} Landsat L2")
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    export(s2, f"s2_{year}", study, 10)
    export(to_30m(ndvi, ee.Reducer.mean(), thermal), f"ndvi30_{year}", thermal, 30)
    export(bt, f"bt_{year}", thermal, 30)
    export(st, f"st_{year}", thermal, 30)


def export_static():
    study, thermal = C.study_grid(), C.thermal_grid()
    export(weak_labels(region(study)), "labels_weak", study, 10)
    export(to_30m(worldcover(), ee.Reducer.mode(), thermal).toByte(), "worldcover30", thermal, 30)


def status():
    for t in ee.batch.Task.list()[:20]:
        s = t.status()
        print(f"{s['description']:<20} {s['state']:<10} {s.get('error_message', '')}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--year", type=int, required=True)
    sub.add_parser("export-static")
    sub.add_parser("status")
    a = p.parse_args()
    init()
    if a.cmd == "export":
        export_year(a.year)
    elif a.cmd == "export-static":
        export_static()
    else:
        status()


if __name__ == "__main__":
    main()
