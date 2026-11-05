"""The identifiability probe: can a small image model read the labels off the renders at all?

A tiny convolutional network, CPU only, trained on one view of each sample to predict the
worn bust ease (regression) and the elasticity class (classification), with whole bodies held
out, against the predict-the-median and predict-the-majority baselines. On a few hundred
samples this says whether the axes are visible in the pictures; it says nothing about real
photos. A run with fewer than about 60 samples is a smoke test of the code, and the result
says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photoai.shards import _records


def _load(run: Path, view: str, size: int):
    from PIL import Image

    recs = [r for r in _records(run) if r["status"] == "ok"
            and view in r.get("files", {}).get("views", {})
            and r["labels"]["ease_worn_cm"].get("bust") is not None]
    xs, ease, cls, bodies = [], [], [], []
    for r in recs:
        im = Image.open(run / r["files"]["views"][view]["rgb"]).convert("RGB")
        im = im.resize((size, int(size * 4 / 3)))
        xs.append(np.asarray(im, dtype=np.float32) / 255.0)
        ease.append(r["labels"]["ease_worn_cm"]["bust"])
        cls.append(r["labels"]["elasticity_class"])
        bodies.append(r["body"]["body_id"])
    return np.stack(xs) if xs else np.zeros((0, 1, 1, 3)), np.array(ease), np.array(cls), bodies


def probe(run: Path, out: Path, epochs: int = 30, view: str = "front", size: int = 96,
          seed: int = 0) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    x, ease, cls, bodies = _load(run, view, size)
    n = len(x)
    res = {"n": int(n), "view": view, "size": size, "epochs": epochs}
    if n < 8:
        res["verdict"] = "too few samples to train anything"
        (out / "probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res
    uniq = sorted(set(bodies))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(uniq, size=max(1, len(uniq) // 4), replace=False).tolist())
    test = np.array([b in held for b in bodies])
    train = ~test
    classes = sorted(set(int(c) for c in cls))
    cidx = {c: i for i, c in enumerate(classes)}
    y_cls = np.array([cidx[int(c)] for c in cls])
    e_mu, e_sd = float(ease[train].mean()), float(ease[train].std() + 1e-6)

    xt = torch.tensor(x).permute(0, 3, 1, 2)
    ye = torch.tensor((ease - e_mu) / e_sd, dtype=torch.float32)
    yc = torch.tensor(y_cls, dtype=torch.long)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                nn.Linear(64 * 16, 64), nn.ReLU())
            self.ease = nn.Linear(64, 1)
            self.cls = nn.Linear(64, len(classes))

        def forward(self, z):
            h = self.f(z)
            return self.ease(h).squeeze(1), self.cls(h)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = np.where(train)[0]
    for _ in range(epochs):
        net.train()
        rng.shuffle(tr)
        for s in range(0, len(tr), 16):
            b = torch.tensor(tr[s:s + 16])
            pe, pc = net(xt[b])
            loss = nn.functional.mse_loss(pe, ye[b]) + nn.functional.cross_entropy(pc, yc[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pe, pc = net(xt)
    pe = pe.numpy() * e_sd + e_mu
    pc = pc.argmax(1).numpy()

    def block(mask):
        if mask.sum() == 0:
            return None
        mae = float(np.abs(pe[mask] - ease[mask]).mean())
        base = float(np.abs(np.median(ease[train]) - ease[mask]).mean())
        acc = float((pc[mask] == y_cls[mask]).mean())
        maj = np.bincount(y_cls[train]).argmax()
        base_acc = float((y_cls[mask] == maj).mean())
        return {"n": int(mask.sum()), "ease_mae_cm": round(mae, 2),
                "ease_mae_predict_median_cm": round(base, 2),
                "elasticity_acc": round(acc, 3), "elasticity_acc_majority": round(base_acc, 3)}

    res.update({"n_train": int(train.sum()), "n_test": int(test.sum()),
                "held_out_bodies": sorted(held), "classes": classes,
                "train": block(train), "test": block(test)})
    res["verdict"] = ("smoke test only: fewer than 60 samples, the numbers mean nothing"
                      if n < 60 else "see test block against its baselines")
    (out / "probe.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
