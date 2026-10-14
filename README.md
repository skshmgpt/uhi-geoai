# Delhi Urban Heat Island & Green Cover Analyzer

Maps surface heat across Delhi from Landsat, classifies land cover from Sentinel-2 with a U-Net++ model, ranks heat hotspots, and estimates how much a proposed green space would cool its surroundings. A small web app lets a planner draw a site and see the predicted cooling with a 90% range.

```
pipeline/   Python: Earth Engine exports, LST, segmentation, hotspots, cooling model, experiments
notebooks/  uhi_colab.ipynb: the GPU part (training) and everything after it, run on Google Colab
db/         PostGIS schema and the hotspot clustering / ranking SQL
api/        Node + Express: /api/hotspots, /api/simulate
web/        Vite + React + Leaflet planner
scripts/    load_db.sh: loads one year's outputs into PostGIS
```

## Run order

### 1. Local setup (once)

```bash
cd pipeline && uv venv --python 3.11 && uv pip install -e ".[dev]"
```

```bash
cd pipeline && .venv/bin/pytest -q
```

### 2. Earth Engine exports (laptop, about an hour of Earth Engine time)

Authenticate yourself; the project ID is already the default in `pipeline/uhi/config.py`:

```bash
pipeline/.venv/bin/earthengine authenticate
```

```bash
cd pipeline && .venv/bin/python -m uhi.gee export-static
```

```bash
cd pipeline && for y in 2023 2024 2025; do .venv/bin/python -m uhi.gee export --year $y; done
```

```bash
cd pipeline && .venv/bin/python -m uhi.gee status
```

Files land in `MyDrive/uhi-geoai/`: `labels_weak`, `worldcover30`, and per year `s2_*`, `ndvi30_*`, `bt_*`, `st_*`. Large rasters arrive as several shards; the pipeline merges them.

### 3. Colab (GPU)

Open `notebooks/uhi_colab.ipynb` in Colab with a T4 runtime. Either push this repo to GitHub and set `REPO_URL` in the first cell, or upload `pipeline/` to `MyDrive/uhi-geoai-code/pipeline/`. Run all cells. Training is 4 configurations × 3 seeds and resumes where it stopped if Colab disconnects. Outputs go to `MyDrive/uhi-geoai-out/`.

Test-set accuracy is only meaningful with hand-corrected test tiles in `uhi-geoai-out/tiles/test_corrected/`. Until they exist, `metrics.json` marks the scores as `partly weak labels`.

### 4. Database, API and web app (laptop)

Copy `uhi-geoai-out/results/` to `pipeline/results/` and `uhi-geoai-out/data/` to `pipeline/data/`, then:

```bash
docker compose up -d db
```

```bash
scripts/load_db.sh 2024
```

```bash
cd api && pnpm install && UHI_YEAR=2024 node src/server.js
```

```bash
cd web && pnpm install && pnpm dev
```

Also copy `uhi-geoai-out/results/layers/` to `web/public/layers/` for the temperature overlay. The database listens on port 5434 by default (`UHI_DB_PORT` changes it).

## Demo mode (no Earth Engine needed)

A synthetic city with a known cooling law exercises the whole stack. Everything it produces is labelled year 2099, and the web app shows a banner saying the data is synthetic. It is never a result.

```bash
cd pipeline && .venv/bin/python -m uhi.synthetic
```

```bash
docker compose up -d db && UHI_DATA=./pipeline/data_demo RESULTS=pipeline/results_demo scripts/load_db.sh 2099
```

```bash
cd api && UHI_YEAR=2099 node src/server.js
```

## Experiments

| ID | Question | Where |
|----|----------|-------|
| E1/E2 | Segmentation accuracy and temporal transfer (train 2023–24, test 2025) | `uhi.seg.train`, notebook §3 |
| E4 | LST vs Landsat L2 surface temperature | `python -m uhi.evaluate lst` |
| E5 | Hotspot stability across years and thresholds | `hotspot_jaccard()` in `db/02_hotspots.sql` |
| E6 | Cooling model, spatial cross-validation | `python -m uhi.evaluate cooling` |
| E7 | Canopy-gain check | not built yet |
| E9 | API latency | not built yet |

Every number in the paper must come from these logs. Nothing from the planning documents is a result.
