"""Cut the 10 m Sentinel-2 composite and weak labels into 256 x 256 tiles with spatial splits.

    python -m uhi.seg.tiles --data data --years 2023 2024 --out tiles

Tiles are grouped into square blocks of `--block-m` metres; whole blocks go to train, val
or test so neighbouring tiles never straddle a split. The test split is the same set of
blocks for every year, so the 2025 composite can be scored on the same places (E3).
"""
import argparse
import json
import os

import numpy as np

from .. import config as C
from .. import rio

TILE = 256
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def ndvi(x):
    b4, b8 = x[2].astype(np.float32), x[3].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num((b8 - b4) / (b8 + b4), nan=0.0)


def block_splits(n_rows, n_cols, tiles_per_block, seed=0):
    """Assign each tile (row, col) to a split by its block."""
    br, bc = -(-n_rows // tiles_per_block), -(-n_cols // tiles_per_block)
    blocks = np.random.default_rng(seed).permutation(br * bc)
    cut1 = int(len(blocks) * SPLITS["train"])
    cut2 = cut1 + int(len(blocks) * SPLITS["val"])
    split_of = {}
    for i, b in enumerate(blocks):
        split_of[b] = "train" if i < cut1 else "val" if i < cut2 else "test"
    return {
        (r, c): split_of[(r // tiles_per_block) * bc + c // tiles_per_block]
        for r in range(n_rows)
        for c in range(n_cols)
    }


def make_tiles(data, years, out, block_m=2_560, min_labelled=0.2, seed=0):
    labels, _ = rio.read(data, "labels_weak", dtype="uint8")
    y = labels[0]
    n_rows, n_cols = y.shape[0] // TILE, y.shape[1] // TILE
    tpb = max(1, block_m // (TILE * 10))
    splits = block_splits(n_rows, n_cols, tpb, seed)
    index = []
    for year in years:
        s2, _ = rio.read(data, f"s2_{year}", dtype="int16")
        for (r, c), split in splits.items():
            win = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            yt = y[win]
            if (yt != C.IGNORE).mean() < min_labelled:
                continue
            xt = s2[(slice(None),) + win]
            if (xt == 0).all(axis=0).mean() > 0.05:
                continue  # composite gap (no clear observation)
            name = f"{year}_{r:03d}_{c:03d}"
            os.makedirs(os.path.join(out, split), exist_ok=True)
            np.savez_compressed(os.path.join(out, split, f"{name}.npz"), x=xt, y=yt)
            index.append({"name": name, "split": split, "year": year, "row": r, "col": c})
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump(index, f)
    counts = {s: sum(1 for i in index if i["split"] == s) for s in SPLITS}
    print(f"{len(index)} tiles {counts}")
    return index


def channel_stats(tile_dir, max_tiles=400):
    """Per-channel mean/std of the 7 model inputs over training tiles."""
    files = sorted(os.listdir(os.path.join(tile_dir, "train")))[:max_tiles]
    xs = [model_input_raw(np.load(os.path.join(tile_dir, "train", f))["x"]) for f in files]
    a = np.stack(xs).astype(np.float64)
    mean, std = a.mean(axis=(0, 2, 3)), a.std(axis=(0, 2, 3)) + 1e-6
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(os.path.join(tile_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
    return stats


def model_input_raw(x):
    """6 reflectance bands (0-1) + NDVI -> 7 channels."""
    refl = x.astype(np.float32) / 10_000.0
    return np.concatenate([refl, ndvi(x)[None]], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    p.add_argument("--out", default="tiles")
    p.add_argument("--block-m", type=int, default=2_560)
    a = p.parse_args()
    make_tiles(a.data, a.years, a.out, a.block_m)
    print(channel_stats(a.out))


if __name__ == "__main__":
    main()
