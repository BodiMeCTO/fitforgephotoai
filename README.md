# fitforgephotoai

A prototype that builds a **synthetic dataset of garments draped on bodies**, where every sample
carries the exact **ease** (the room the garment has over the body, in cm at bust, waist and hips)
and the **elasticity** (the fabric's stretch) it was made with, plus renders of it from several
views and its settled mesh. Its purpose is to test whether a photo model can learn to read ease
and elasticity off a picture: the first step towards the two uses bodi.me wants, body measurement
from photos and automatic calibration of a garment's ease and elasticity.

It drives three existing pieces: the body model ([annyextension](https://github.com/BodiMeCTO/annyextension)),
the FitForge solver ([fitforgesolver](https://github.com/BodiMeCTO/fitforgesolver): GarmentCode
drafting, the cloth drape, QC, fit labels, the renderer) and, by design but not yet, the API. What
each of those would need to make this a real pipeline is the spec:
[`docs/spec-synthetic-ease-elasticity-dataset.md`](docs/spec-synthetic-ease-elasticity-dataset.md).

**This repo is public.** Nothing it generates is committed: a sample is a GarmentCode draft on an
Anny body, draped and rendered, and the inputs carry licences (GarmentCode MIT, Anny Apache 2.0,
the plus-size body field is CAESAR-derived under the WEAR licence). The prototype does not touch
the licensed DXF pattern library at all.

## What a sample is

```
ANSUR II target (height, bust, waist, hips)      the public US Army survey, via annyext
   -> Anny body fitted to it (rest pose)
   -> GarmentCode draft of a shirt for that body at an ease offset  (-8 .. +12 cm)
   -> draped in one of the solver's fabrics                          (denim .. stretch knit)
   -> the solver's QC and fit labels on the settled mesh
   -> ease read three ways, elasticity recorded, mesh welded and saved
   -> 4 views rendered: shaded picture, panel segmentation, normals, depth
```

Ease comes three ways, because they differ and the difference matters:
- `ease_designed_cm`: the offset the garment was drafted with (the knob that was turned).
- `ease_pattern_cm`: the flat pattern's own girth minus the body's: the true room over the body,
  including the ease the design itself builds in. Only `bust` is trustworthy today (the solver's
  chart reader takes waist and hip at fixed depths below the underarm that do not match a drafted
  shirt; spec item S3).
- `ease_worn_cm`: the settled garment's torso girth on the body minus the body's, by the body
  model's own tailor rule. What the garment does once worn: a tight knit stretches, a loose woven
  hangs.

Elasticity is the fabric as the solver draped it: an ordered class (0 rigid woven, 1 woven, 2 firm
knit, 3 stretch knit), the solver's `strain_limit` (the most the cloth may stretch along the thread)
and `stretch_compliance` (how easily it stretches), and the effective modulus the solver's pressure
label uses.

The record also carries the solver's strain percentiles, air gap, contact pressure (marked
uncalibrated by the solver), fit class per region, QC verdict, convergence, every setting, and the
commits of the solver, the body model and this builder.

## Setup

The builder runs inside the solver's own environment, so the solver, warp, torch, anny and the
pinned annyextension all come from there.

```bash
# 1. the solver, per its README (clones annyextension at the pin, installs the locked environment)
git clone https://github.com/BodiMeCTO/fitforgesolver
cd fitforgesolver && SKIP_GPU_CHECK=1 SKIP_TESTS=1 TORCH_CPU=1 bash scripts/setup_linux.sh
```

```bash
# 2. this package, editable, into that environment (no dependencies of its own)
git clone https://github.com/BodiMeCTO/fitforgephotoai
cd fitforgesolver && uv pip install -e ../fitforgephotoai --no-deps
```

The first body build downloads the Anny weights (about 0.9 GB) to `~/.cache/anny`. No data root
(`FITFORGE_ROOT`) is needed: the drafts are GarmentCode presets vendored in the solver.

## Run

```bash
cd fitforgesolver && .venv/bin/python -m photoai plan --config ../fitforgephotoai/configs/demo_cpu.yaml
```

```bash
cd fitforgephotoai && scripts/run_workers.sh configs/demo_cpu.yaml outputs/demo_cpu 4
```

```bash
cd fitforgesolver && .venv/bin/python -m photoai report --run ../fitforgephotoai/outputs/demo_cpu --md ../fitforgephotoai/outputs/demo_cpu/REPORT.md
```

```bash
cd fitforgesolver && .venv/bin/python -m photoai shard --run ../fitforgephotoai/outputs/demo_cpu --out ../fitforgephotoai/datasets/demo
```

```bash
cd fitforgesolver && .venv/bin/python -m photoai probe --run ../fitforgephotoai/outputs/demo_cpu --out ../fitforgephotoai/outputs/demo_cpu/probe
```

- `run` is resumable and sharded: each worker takes every Nth job of the same deterministic grid
  and skips samples already written. Each worker needs its own warp kernel cache
  (`WARP_CACHE_PATH`); the script sets one per worker. CPU only (`CUDA_VISIBLE_DEVICES=-1`).
- `configs/pilot_cpu.yaml` is the 288-sample run (12 bodies): about 5 hours on 4 cores.
- `probe` trains a very small image model on one view with whole bodies held out and reports
  against predict-the-median / predict-the-majority baselines. It is an identifiability check on
  synthetic data only; it says nothing about real photographs.

Tests: `cd fitforgesolver && .venv/bin/python -m pytest -q ../fitforgephotoai/tests` (fast; no drape).

## Cost, measured here (4-core cloud box, no GPU)

| step | seconds per sample |
|---|---:|
| draft (GarmentCode) | 1.3 |
| drape, the solver's production settings at a 2 cm mesh | 230 |
| drape, the solver's plain defaults (no shoulder hold; the shirt slides off) | 80 |
| QC + labels | 0.1 to 2 |
| 4 renders at 384x512 | 5 |

The production settings (the shoulder band held through dressing, hardened seams, point-triangle
self-collision) are needed: with the plain defaults the shirt slid 20 cm down the body. The
production mesh is 1 cm, four times the triangles: not run here. On the solver's GPU box the same
drape takes seconds, which is the number that decides whether a large dataset is feasible (spec,
section 6).

## Output layout

```
outputs/<run>/config.json
outputs/<run>/bodies/<body_id>/body_rest.ply, body_rest.json
outputs/<run>/samples/<sample_id>/record.json      every label and provenance field
outputs/<run>/samples/<sample_id>/garment.ply      the settled garment, metres, seams welded
outputs/<run>/samples/<sample_id>/<view>.{rgb,seg,normal,depth}.png
outputs/<run>/shards/manifest-<worker>.jsonl       one line per sample, the labels only
datasets/<name>/<name>-000000.tar, index.jsonl, dataset.json, CARD.md
```

## What this clone can't give you

| what | where | how to get it |
|---|---|---|
| the solver and its environment | `../fitforgesolver` (private) | clone it; its README |
| the body model at the solver's pin | `fitforgesolver/vendor/annyextension` | the solver's setup script clones it |
| the Anny weights (0.9 GB) | `~/.cache/anny` | downloaded on the first body build |
| any built sample or dataset | `outputs/`, `datasets/` (ignored) | run the builder; deterministic from config and seed |

## Status

Prototype, 2026-11-05. One smoke sample and a 24-sample demo run were built on CPU in this
session; the 288-sample pilot was not (a session restart killed it after 10 hours, with no
sample finished). `issues/` lists what was found. The handoff is `docs/handoff-2026-11-05.md`.
