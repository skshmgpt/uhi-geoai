import numpy as np

from uhi import hotspots


def test_candidates_find_hot_region_with_area_and_mean():
    inten = np.zeros((20, 20))
    inten[5:9, 5:10] = 3.0  # 4 x 5 px at 30 m = 18,000 m2
    transform = [30.0, 0, 700_000.0, 0, -30.0, 3_150_000.0]
    rows = hotspots.candidates(inten, transform, 2.0)
    assert len(rows) == 1
    wkt, mean_i, area = rows[0]
    assert np.isclose(mean_i, 3.0) and np.isclose(area, 18_000)
    assert wkt.startswith("POLYGON")


def test_intensity_uses_rural_median():
    lst = np.array([[40.0, 42.0], [30.0, 32.0]])
    rural = np.array([[False, False], [True, True]])
    inten, ref = hotspots.intensity(lst, rural)
    assert ref == 31.0 and inten[0, 1] == 11.0
