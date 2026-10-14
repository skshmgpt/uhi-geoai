"""Regenerate tests/fixtures/cooling_case.json, shared by the Python and Node tests."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_cooling import synthetic_sites  # noqa: E402

from uhi import cooling  # noqa: E402

params = cooling.fit_params(synthetic_sites(), n_boot=100, seed=3)
inputs = [
    {"area_m2": 2e4, "ndvi_target": 0.7, "ndvi_now": 0.15, "shape": 1.2, "built": 0.6},
    {"area_m2": 1.5e5, "ndvi_target": 0.6, "ndvi_now": 0.35, "shape": 2.0, "built": 0.8},
    {"area_m2": 6e3, "ndvi_target": 0.5, "ndvi_now": 0.55, "shape": 1.0, "built": 0.4},
]
cases = [{"input": i, "expected": cooling.predict(params, **i)} for i in inputs]
out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "cooling_case.json")
with open(out, "w") as f:
    json.dump({"params": params, "cases": cases}, f)
print(f"wrote {out}")
