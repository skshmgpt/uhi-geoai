"""Calibration + E6 evaluation on a small synthetic city whose true cooling law is known."""
import json

import pytest

from uhi import evaluate, process, synthetic


@pytest.fixture
def city(tmp_path, monkeypatch):
    synthetic.use_synthetic_grids(monkeypatch.setattr)
    return tmp_path, synthetic.make_city(str(tmp_path))


def test_calibrate_and_evaluate(city):
    data, n_parks = city
    results = data / "results"
    year = synthetic.YEAR
    process.run_calibrate(str(data), year, f"classes_{year}", str(results), n_boot=100)

    params = json.loads((results / f"cooling_params_{year}.json").read_text())
    assert params["n_sites"] >= 0.7 * n_parks
    assert abs(params["theta"][1] - synthetic.TRUE_THETA["ln_area"]) < 0.15
    assert abs(params["theta"][2] - synthetic.TRUE_THETA["ndvi"]) < 1.0
    assert abs(params["phi"][1] - synthetic.TRUE_PHI["ln_area"]) < 0.1

    evaluate.eval_cooling(str(results), year, n_splits=5, n_boot=50)
    m = json.loads((results / f"e6_cooling_{year}.json").read_text())["models"]
    assert m["decay_linear"]["profile_mae_c"] < m["linear_synopsis"]["profile_mae_c"]
    assert m["decay_linear"]["profile_mae_c"] < m["distance_only"]["profile_mae_c"]
    assert 0.5 <= m["decay_linear"]["pi90_coverage"] <= 1.0


def test_hotspots_find_the_planted_hot_blobs(city):
    data, _ = city
    process.run_hotspots(str(data), synthetic.YEAR, 2.0, str(data))
    sql = (data / f"hotspot_candidates_{synthetic.YEAR}_2.0.sql").read_text()
    areas = [float(line.split("VALUES (")[1].split(", ")[3]) for line in sql.splitlines() if line.startswith("INSERT")]
    # Six blobs were planted (some may merge); isolated noisy pixels stay below the 5000 m2 cut.
    big = [a for a in areas if a >= 5_000]
    assert 3 <= len(big) <= 6
