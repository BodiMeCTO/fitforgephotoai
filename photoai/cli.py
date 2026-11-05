"""``python -m photoai <command>``: plan | run | shard | report | probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="photoai")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="lay out the grid and say how big it is")
    p.add_argument("--config")

    p = sub.add_parser("run", help="build this worker's slice of the grid (resumable)")
    p.add_argument("--config")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="stop after this many samples")

    p = sub.add_parser("shard", help="pack a run into WebDataset tar shards with an index and card")
    p.add_argument("--run", default="outputs/run")
    p.add_argument("--out", default="datasets/photoai-v0")
    p.add_argument("--shard-size", type=int, default=256)

    p = sub.add_parser("report", help="summarise a run: counts, label ranges, QC, timings")
    p.add_argument("--run", default="outputs/run")
    p.add_argument("--md", default=None, help="also write a markdown report here")

    p = sub.add_parser("probe", help="train the small image model that tests whether the labels "
                                     "can be read off the renders")
    p.add_argument("--run", default="outputs/run")
    p.add_argument("--out", default="outputs/probe")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--view", default="front")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        from photoai.sweep import build_grid, load_config

        cfg = load_config(a.config)
        jobs = build_grid(cfg)
        print(json.dumps({"n_jobs": len(jobs), "n_bodies": cfg["n_bodies"], "poses": cfg["poses"],
                          "designs": cfg["designs"], "ease_offsets_cm": cfg["ease_offsets_cm"],
                          "fabrics": cfg["fabrics"], "views": [v["name"] for v in cfg["views"]]},
                         indent=1))
        return 0
    if a.cmd == "run":
        from photoai.sweep import load_config, run

        cfg = load_config(a.config)
        summary = run(cfg, a.out, shard=a.shard, n_shards=a.n_shards, limit=a.limit)
        print(json.dumps(summary, indent=1))
        return 0 if summary["failed"] == 0 else 1
    if a.cmd == "shard":
        from photoai.shards import package

        print(json.dumps(package(Path(a.run), Path(a.out), shard_size=a.shard_size), indent=1))
        return 0
    if a.cmd == "report":
        from photoai.report import report

        rep = report(Path(a.run), md_path=Path(a.md) if a.md else None)
        print(json.dumps(rep, indent=1, default=str))
        return 0
    if a.cmd == "probe":
        from photoai.probe import probe

        res = probe(Path(a.run), Path(a.out), epochs=a.epochs, view=a.view, size=a.size,
                    seed=a.seed)
        print(json.dumps(res, indent=1))
        return 0
    return 2
