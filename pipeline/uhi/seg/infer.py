"""ONNX export and full-city inference.

    python -m uhi.seg.infer export --run runs/unetpp_efficientnet-b4_focal_dice_s0 --tiles tiles
    python -m uhi.seg.infer predict --onnx runs/.../model.onnx --stats tiles/stats.json --data data --year 2024

`predict` slides 256-pixel windows with a 32-pixel overlap over s2_<year> and writes
classes_<year>.tif (uint8, 255 where the composite has no data) on the same 10 m grid.
"""
import argparse
import json
import os
import time

import numpy as np

from .. import config as C
from .. import rio
from .tiles import TILE, model_input_raw


def export(run_dir, opset=18):
    import torch

    from .train import build_model

    with open(os.path.join(run_dir, "metrics.json")) as f:
        args = json.load(f)["args"]
    model = build_model(args["arch"], args["encoder"], weights=None)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu"))
    model.eval()
    path = os.path.join(run_dir, "model.onnx")
    torch.onnx.export(
        model, torch.zeros(1, 7, TILE, TILE), path, opset_version=opset,
        input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}},
    )
    print(f"wrote {path}")


def predict(onnx_path, stats_path, data, year, overlap=32, batch=8):
    import onnxruntime as ort

    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], np.float32)[:, None, None]
    std = np.array(stats["std"], np.float32)[:, None, None]
    sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())

    s2, prof = rio.read(data, f"s2_{year}", dtype="int16")
    _, h, w = s2.shape
    step = TILE - 2 * overlap
    logits = np.zeros((len(C.CLASSES), h, w), np.float32)
    starts_r = list(range(0, max(h - TILE, 0) + 1, step)) + [max(h - TILE, 0)]
    starts_c = list(range(0, max(w - TILE, 0) + 1, step)) + [max(w - TILE, 0)]
    windows = sorted({(r, c) for r in starts_r for c in starts_c})
    t0 = time.time()
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        x = np.stack([(model_input_raw(s2[:, r:r + TILE, c:c + TILE]) - mean) / std for r, c in chunk]).astype(np.float32)
        out = sess.run(None, {"x": x})[0]
        for (r, c), lo in zip(chunk, out):
            # Keep only the window centre so tile edges never decide a pixel, except at the image border.
            r0 = 0 if r == 0 else overlap
            c0 = 0 if c == 0 else overlap
            r1 = TILE if r + TILE >= h else TILE - overlap
            c1 = TILE if c + TILE >= w else TILE - overlap
            logits[:, r + r0:r + r1, c + c0:c + c1] = lo[:, r0:r1, c0:c1]
    classes = logits.argmax(0).astype(np.uint8)
    classes[(s2 == 0).all(axis=0)] = C.IGNORE
    path = os.path.join(data, f"classes_{year}.tif")
    rio.write(path, classes, prof, "uint8", nodata=C.IGNORE)
    print(f"wrote {path} ({len(windows)} windows, {time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    q = sub.add_parser("predict")
    q.add_argument("--onnx", required=True)
    q.add_argument("--stats", default="tiles/stats.json")
    q.add_argument("--data", default="data")
    q.add_argument("--year", type=int, required=True)
    a = p.parse_args()
    if a.cmd == "export":
        export(a.run)
    else:
        predict(a.onnx, a.stats, a.data, a.year)


if __name__ == "__main__":
    main()
