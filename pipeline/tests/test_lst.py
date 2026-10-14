import numpy as np

from uhi import lst

# Landsat 8 Band 10 constants as they appear in scene metadata; only used as plausible inputs.
K1, K2 = 774.8853, 1321.0789


def planck_radiance(t_kelvin):
    return K1 / (np.exp(K2 / t_kelvin) - 1)


def test_bt_inverts_planck():
    t = np.array([280.0, 300.0, 330.0])
    np.testing.assert_allclose(lst.radiance_to_bt(planck_radiance(t), K1, K2), t, rtol=1e-10)


def test_dn_to_radiance_is_linear():
    assert lst.dn_to_radiance(1000, 3.342e-4, 0.1) == 3.342e-4 * 1000 + 0.1


def test_vegetation_proportion_clips_before_squaring():
    pv = lst.vegetation_proportion(np.array([-0.5, 0.1, 0.35, 0.5, 0.9]))
    np.testing.assert_allclose(pv, [0, 0, 0.25, 1, 1])


def test_emissivity_range_and_water_override():
    eps = lst.emissivity(np.array([0.0, 0.8]), water_mask=np.array([True, False]))
    np.testing.assert_allclose(eps, [lst.WATER_EMISSIVITY, 0.990])


def test_unit_emissivity_returns_brightness_temperature():
    assert np.isclose(lst.lst_celsius(300.0, 1.0), 300.0 - 273.15)


def test_lower_emissivity_means_hotter_surface():
    assert lst.lst_celsius(310.0, 0.97) > lst.lst_celsius(310.0, 0.99)
