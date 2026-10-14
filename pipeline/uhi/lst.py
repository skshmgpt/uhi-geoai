"""Single-channel LST retrieval from Landsat 8/9 TIRS Band 10 (Avdan & Jovanovska 2016).

DN -> TOA radiance -> brightness temperature runs per scene in Earth Engine (see gee.py),
because the calibration constants differ between Landsat 8 and 9. This module holds the
same maths for local use and the emissivity/LST steps applied to the seasonal BT median.
"""
import numpy as np

RHO_UM_K = 14388.0  # h*c/k_B in micrometre-kelvin
BAND10_WAVELENGTH_UM = 10.895  # centre of the 10.60-11.19 um Band 10 range
NDVI_SOIL, NDVI_VEG = 0.2, 0.5
WATER_EMISSIVITY = 0.991


def dn_to_radiance(dn, mult, add):
    return mult * np.asarray(dn, dtype=np.float64) + add


def radiance_to_bt(radiance, k1, k2):
    """At-sensor brightness temperature in kelvin."""
    return k2 / np.log(k1 / np.asarray(radiance, dtype=np.float64) + 1.0)


def vegetation_proportion(ndvi, ndvi_soil=NDVI_SOIL, ndvi_veg=NDVI_VEG):
    scaled = (np.asarray(ndvi, dtype=np.float64) - ndvi_soil) / (ndvi_veg - ndvi_soil)
    return np.clip(scaled, 0.0, 1.0) ** 2


def emissivity(ndvi, water_mask=None):
    eps = 0.004 * vegetation_proportion(ndvi) + 0.986
    if water_mask is not None:
        eps = np.where(water_mask, WATER_EMISSIVITY, eps)
    return eps


def lst_celsius(bt_kelvin, eps, wavelength_um=BAND10_WAVELENGTH_UM):
    bt = np.asarray(bt_kelvin, dtype=np.float64)
    return bt / (1.0 + (wavelength_um * bt / RHO_UM_K) * np.log(eps)) - 273.15
