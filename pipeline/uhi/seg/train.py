"""Segmentation training and evaluation (E1-E3). Runs on a single Colab GPU.

    python -m uhi.seg.train --tiles tiles --arch unetpp --encoder efficientnet-b4 --loss focal_dice --seed 0
    python -m uhi.seg.train --tiles tiles --arch unet --encoder resnet34 --loss ce --seed 0
    python -m uhi.seg.train --tiles tiles --arch deeplabv3p --encoder resnet50 --loss ce --seed 0

Test scores use hand-corrected labels from tiles/test_corrected/<row>_<col>.npz when present
(key "y"); otherwise weak labels, and the output says so.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .. import config as C
from .tiles import model_input_raw

N_CLASSES = len(C.CLASSES)


class Tiles(Dataset):
    def __init__(self, tile_dir, split, years, stats, augment=False, corrected=False):
        with open(os.path.join(tile_dir, "index.json")) as f:
            index = json.load(f)
        self.items = [i for i in index if i["split"] == split and i["year"] in years]
        self.dir, self.split, self.augment = tile_dir, split, augment
        self.mean = np.array(stats["mean"], np.float32)[:, None, None]
        self.std = np.array(stats["std"], np.float32)[:, None, None]
        self.corrected_dir = os.path.join(tile_dir, "test_corrected") if corrected else None
        self.n_corrected = 0
        if self.corrected_dir and os.path.isdir(self.corrected_dir):
            self.n_corrected = sum(os.path.exists(self._corrected_path(i)) for i in self.items)

    def _corrected_path(self, item):
        return os.path.join(self.corrected_dir, f"{item['row']:03d}_{item['col']:03d}.npz")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        item = self.items[k]
        z = np.load(os.path.join(self.dir, self.split, f"{item['name']}.npz"))
        x = (model_input_raw(z["x"]) - self.mean) / self.std
        y = z["y"]
        if self.corrected_dir and os.path.exists(self._corrected_path(item)):
            y = np.load(self._corrected_path(item))["y"]
        if self.augment:
            x, y = augment(x, y)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y.astype(np.int64)))


def augment(x, y, rng=np.random):
    k = rng.randint(4)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.rand() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    x = x * (1 + rng.uniform(-0.1, 0.1))  # brightness jitter in standardised units
    return x, y


def build_model(arch, encoder, in_channels=7, weights="imagenet"):
    """smp widens the first conv for 7 channels by cycling the pretrained RGB filters
    and rescaling them by 3/7."""
    import segmentation_models_pytorch as smp

    cls = {"unetpp": smp.UnetPlusPlus, "unet": smp.Unet, "deeplabv3p": smp.DeepLabV3Plus}[arch]
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=N_CLASSES)


def class_weights(ds, max_tiles=300):
    counts = np.zeros(N_CLASSES)
    for k in range(min(len(ds), max_tiles)):
        y = np.load(os.path.join(ds.dir, ds.split, f"{ds.items[k]['name']}.npz"))["y"]
        counts += np.bincount(y[y != C.IGNORE].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    w = 1.0 / np.sqrt(np.maximum(counts / counts.sum(), 1e-6))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def focal_loss(logits, y, weights, gamma=2.0):
    ce = F.cross_entropy(logits, y, weight=weights, ignore_index=C.IGNORE, reduction="none")
    pt = torch.exp(-F.cross_entropy(logits, y, ignore_index=C.IGNORE, reduction="none"))
    valid = y != C.IGNORE
    return ((1 - pt) ** gamma * ce)[valid].mean()


def dice_loss(logits, y, eps=1.0):
    valid = (y != C.IGNORE).unsqueeze(1)
    p = logits.softmax(1) * valid
    onehot = F.one_hot(y.clamp(max=N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2) * valid
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def make_loss(name, weights, alpha=0.5):
    if name == "ce":
        return lambda lo, y: F.cross_entropy(lo, y, ignore_index=C.IGNORE)
    if name == "focal_dice":
        return lambda lo, y: alpha * focal_loss(lo, y, weights) + (1 - alpha) * dice_loss(lo, y)
    raise ValueError(name)


def confusion(pred, y, n=N_CLASSES):
    valid = y != C.IGNORE
    return torch.bincount((y[valid] * n + pred[valid]).view(-1), minlength=n * n).reshape(n, n)


def scores(cm):
    cm = cm.double()
    tp = cm.diag()
    iou = tp / (cm.sum(0) + cm.sum(1) - tp).clamp(min=1)
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp(min=1)
    return {
        "miou": float(iou.mean()),
        "iou": {c: float(v) for c, v in zip(C.CLASSES, iou)},
        "f1": {c: float(v) for c, v in zip(C.CLASSES, f1)},
        "oa": float(tp.sum() / cm.sum().clamp(min=1)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cm += confusion(model(x).argmax(1), y)
    return scores(cm.cpu()), cm.cpu().tolist()


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(a.tiles, "stats.json")) as f:
        stats = json.load(f)
    train_ds = Tiles(a.tiles, "train", a.train_years, stats, augment=True)
    val_ds = Tiles(a.tiles, "val", a.train_years, stats)
    kw = dict(batch_size=a.batch, num_workers=a.workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, **kw)

    model = build_model(a.arch, a.encoder).to(device)
    loss_fn = make_loss(a.loss, class_weights(train_ds).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    run_name = f"{a.arch}_{a.encoder}_{a.loss}_s{a.seed}"
    out = os.path.join(a.out, run_name)
    os.makedirs(out, exist_ok=True)
    best, history = -1.0, []
    for epoch in range(a.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val, _ = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "loss": total / max(len(train_dl), 1), "val_miou": val["miou"]})
        print(f"epoch {epoch:3d} loss {history[-1]['loss']:.4f} val mIoU {val['miou']:.4f} ({time.time() - t0:.0f}s)")
        if val["miou"] > best:
            best = val["miou"]
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device))
    result = {"run": run_name, "args": vars(a), "best_val_miou": best, "history": history, "test": {}}
    for year in a.test_years:
        ds = Tiles(a.tiles, "test", [year], stats, corrected=True)
        if len(ds) == 0:
            continue
        s, cm = evaluate(model, DataLoader(ds, **kw), device)
        s.update(confusion=cm, n_tiles=len(ds), n_corrected=ds.n_corrected,
                 labels="hand-corrected" if ds.n_corrected == len(ds) else "partly weak labels")
        result["test"][str(year)] = s
        print(f"test {year}: mIoU {s['miou']:.4f} on {len(ds)} tiles ({s['labels']})")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="tiles")
    p.add_argument("--out", default="runs")
    p.add_argument("--arch", default="unetpp", choices=["unetpp", "unet", "deeplabv3p"])
    p.add_argument("--encoder", default="efficientnet-b4")
    p.add_argument("--loss", default="focal_dice", choices=["ce", "focal_dice"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-years", type=int, nargs="+", default=[2023, 2024])
    p.add_argument("--test-years", type=int, nargs="+", default=[2023, 2024, 2025])
    run(p.parse_args())


if __name__ == "__main__":
    main()
