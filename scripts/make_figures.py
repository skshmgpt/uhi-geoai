"""Paper figures from the logged outputs. Run with pipeline/.venv/bin/python scripts/make_figures.py.

Inputs: pipeline/results/layers/lst_2024.{png,json}, docs/figures/top20_2024.geojson (exported from PostGIS),
pipeline/results/profiles_*.npz. Writes PNGs to docs/figures/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "pipeline" / "results"
OUT = ROOT / "docs" / "figures"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "savefig.dpi": 200})


def lst_map():
    meta = json.loads((RES / "layers" / "lst_2024.json").read_text())
    (s, w), (n, e) = meta["bounds"]
    img = plt.imread(RES / "layers" / "lst_2024.png")
    hs = json.loads((OUT / "top20_2024.geojson").read_text())["features"]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.imshow(img, extent=(w, e, s, n), interpolation="nearest")
    for f in hs:
        polys = f["geometry"]["coordinates"]
        for poly in polys:
            xy = np.array(poly[0])
            ax.fill(xy[:, 0], xy[:, 1], facecolor="none", edgecolor="cyan", linewidth=0.7)
        pts = np.vstack([np.array(p[0]) for p in polys])
        ax.annotate(str(f["properties"]["rank"]), pts.mean(axis=0), color="white", fontsize=7, weight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="cyan", lw=0.5, alpha=0.8))
    ax.add_patch(plt.Rectangle((76.84, 28.40), 0.51, 0.49, fill=False, ec="white", ls="--", lw=0.8))
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    sm = plt.cm.ScalarMappable(cmap=meta["colormap"], norm=plt.Normalize(meta["min_c"], meta["max_c"]))
    fig.colorbar(sm, ax=ax, shrink=0.75, label="LST, our inversion (°C)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lst_hotspots_2024.png")
    plt.close(fig)


def cooling_profiles():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for year, color in [(2023, "#1b9e77"), (2024, "#d95f02"), (2025, "#7570b3")]:
        z = np.load(RES / f"profiles_{year}.npz")
        d, p = z["d"], z["profiles"]
        rel = p - p[:, -1:]  # LST relative to the outermost (1 km) ring of the same site
        med = np.nanmedian(rel, axis=0)
        q1, q3 = np.nanpercentile(rel, [25, 75], axis=0)
        ax.plot(d, med, color=color, label=f"{year} (n = {len(p)})")
        ax.fill_between(d, q1, q3, color=color, alpha=0.15, lw=0)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(100, color="grey", lw=0.6, ls=":")
    ax.text(105, ax.get_ylim()[0] * 0.92, "native TIRS pixel (100 m)", fontsize=7, color="grey")
    ax.set_xlabel("Distance from green-space edge (m)")
    ax.set_ylabel("LST − LST at 1 km ring (°C)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cooling_profiles.png")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#333", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))

    ax.text(1, 47, "Offline pipeline (once per city and season)", fontsize=8.5, weight="bold")
    ax.text(1, 17, "Online service (per planner request)", fontsize=8.5, weight="bold")
    off = "#e8f0f8"
    box(1, 33, 14, 9, "Google Earth Engine\ncomposites\n(S2, Landsat 8/9)", off)
    box(20, 38, 17, 7, "LST inversion\n(Band 10, NDVI ε)", off)
    box(20, 27, 17, 7, "U-Net++ / EffNet-B4\n(Colab GPU → ONNX)", off)
    box(41, 38, 17, 7, "SUHI intensity +\nurban mask", off)
    box(41, 27, 17, 7, "Ring profiles +\ndecay fits", off)
    box(62, 38, 17, 7, "DBSCAN clusters,\nheat-load rank", off)
    box(62, 27, 17, 7, "Cross-site regression\n+ bootstrap", off)
    box(83, 30, 15, 12, "PostGIS\nhotspots,\nparameters,\nNDVI / built\nrasters", "#f6ecd9")
    for y in (41.5, 30.5):
        arrow(15.5, 37.5, 19.5, y)
    arrow(37.5, 41.5, 40.5, 41.5)
    arrow(37.5, 30.5, 40.5, 30.5)
    arrow(28.5, 34, 45, 37.6)  # classes -> urban mask
    arrow(58.5, 41.5, 61.5, 41.5)
    arrow(58.5, 30.5, 61.5, 30.5)
    arrow(79.5, 41.5, 82.5, 38)
    arrow(79.5, 30.5, 82.5, 34)

    on = "#eaf5ea"
    box(2, 3, 20, 9, "React + Leaflet web map\n(draw polygon)", on)
    box(33, 3, 22, 9, "Node / Express API\n/api/simulate, /api/hotspots", on)
    box(64, 3, 22, 9, "Traits in PostGIS\n(area, shape, NDVI, built)", on)
    arrow(22.5, 8.5, 32.5, 8.5)
    arrow(32.5, 6, 22.5, 6)
    ax.text(27, 9.6, "GeoJSON", ha="center", fontsize=6.5)
    ax.text(27, 3.6, "ΔT, extent, 90% PI", ha="center", fontsize=6.5)
    arrow(55.5, 7.5, 63.5, 7.5)
    arrow(90, 29.5, 78, 12.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    lst_map()
    cooling_profiles()
    print("figures written to", OUT)
