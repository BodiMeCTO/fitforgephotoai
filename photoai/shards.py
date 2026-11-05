"""Pack a run into WebDataset-style tar shards: files grouped by sample key in plain tars
(``<id>.<view>.rgb.png``, ``<id>.<view>.seg.png``, ``<id>.json`` ...), ``index.jsonl`` with every
record, a sha256 of the index and a dataset card that says what the data is and is not.
Standard library only; any WebDataset loader reads the result."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path


def _records(run: Path) -> list[dict]:
    recs = []
    for p in sorted((run / "samples").glob("*/record.json")):
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    return recs


def package(run: Path, out: Path, name: str = "photoai", shard_size: int = 256,
            include_failed: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in _records(run) if include_failed or r["status"] == "ok"]
    n_shards = 0
    index = out / "index.jsonl"
    with open(index, "w", encoding="utf-8") as idx:
        for start in range(0, len(recs), shard_size):
            shard_name = f"{name}-{n_shards:06d}.tar"
            with tarfile.open(out / shard_name, "w") as tf:
                for r in recs[start:start + shard_size]:
                    sid = r["sample_id"]
                    for view, files in r.get("files", {}).get("views", {}).items():
                        for kind in ("rgb", "seg", "normal", "depth"):
                            tf.add(run / files[kind], arcname=f"{sid}.{view}.{kind}.png")
                    if r.get("files", {}).get("garment_ply"):
                        tf.add(run / r["files"]["garment_ply"], arcname=f"{sid}.garment.ply")
                    blob = json.dumps(r).encode()
                    info = tarfile.TarInfo(f"{sid}.json")
                    info.size = len(blob)
                    tf.addfile(info, io.BytesIO(blob))
                    idx.write(json.dumps({"shard": shard_name, **r}) + "\n")
            n_shards += 1
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    stats = summarise(recs)
    stats.update({"name": name, "n_shards": n_shards, "index_sha256": digest})
    (out / "dataset.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    (out / "CARD.md").write_text(card(stats), encoding="utf-8")
    return stats


def summarise(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["status"] == "ok"]
    bodies = sorted({r["body"]["body_id"] for r in ok})
    fab = {}
    for r in ok:
        fab[r["job"]["fabric"]] = fab.get(r["job"]["fabric"], 0) + 1
    ease = {}
    for r in ok:
        ease[r["job"]["ease_label"]] = ease.get(r["job"]["ease_label"], 0) + 1
    designs = {}
    for r in ok:
        designs[r["job"]["design"]] = designs.get(r["job"]["design"], 0) + 1
    prov = ok[0]["provenance"] if ok else (recs[0]["provenance"] if recs else {})
    return {
        "n_samples": len(recs), "n_ok": len(ok), "n_failed": len(recs) - len(ok),
        "n_qc_passed": sum(1 for r in ok if r["qc"]["passed"]),
        "n_converged": sum(1 for r in ok if r["drape"]["converged"]),
        "n_bodies": len(bodies),
        "designs": designs, "ease_steps": ease, "fabrics": fab,
        "views": sorted({v for r in ok for v in r.get("files", {}).get("views", {})}),
        "population": prov.get("ansur_source"),
        "solver_commit": prov.get("solver_commit"),
        "annyext_commit": prov.get("annyext_commit"),
        "engine": prov.get("engine"),
        "drape_config": prov.get("drape_config"),
    }


def card(s: dict) -> str:
    return f"""# Dataset card: {s['name']}

Index sha256 `{s['index_sha256']}` · {s['n_shards']} shard(s) · {s['n_ok']} samples
({s['n_qc_passed']} passed the solver's QC, {s['n_converged']} drapes converged, {s['n_failed']} failed and left out).

## What a sample is
A garment drafted by GarmentCode for a body, draped on that body by the FitForge solver, rendered
from {len(s['views'])} views (shaded picture, panel segmentation, normals, depth), with its settled mesh.

## The labels, in words
- **Ease** (cm, full girth at bust, waist and hips): three readings, all in the record.
  `ease_designed_cm` is the offset the garment was drafted with. `ease_pattern_cm` is the flat
  pattern's own girth minus the body's (the garment's true room over the body, including the ease
  the design builds in; trust `bust` only, the solver's chart reader takes waist and hip at fixed
  depths that do not match a drafted shirt). `ease_worn_cm` is the settled garment's torso girth
  on the body minus the body's, by the tailor's rule: what the garment does once worn.
- **Elasticity**: the fabric the drape used, as an ordered class (0 rigid woven .. 3 stretch
  knit) and as the solver's numbers (`strain_limit`, the most the cloth may stretch along the
  thread; `stretch_compliance`, how easily it stretches; the effective modulus in Pa).
- The solver's own fit labels: strain percentiles, air gap, contact pressure (uncalibrated), a
  fit class per region, QC verdict, convergence.

## Counts
bodies {s['n_bodies']} · designs {s['designs']} · ease steps {s['ease_steps']} · fabrics {s['fabrics']}

## Where it comes from
Bodies: Anny sliders fitted to targets drawn from the ANSUR II survey ({s['population']}).
Solver commit `{s['solver_commit']}`, body model commit `{s['annyext_commit']}`,
engine {json.dumps(s['engine'])}. Drape settings: {json.dumps(s['drape_config'])}.

## What it is not
- Not photographs. Shaded renders of a rigid body: no skin, hair, texture, real light or
  camera; the body does not deform under a tight garment.
- Not calibrated physics. The solver's stretch is calibrated for wovens; its bending is not
  calibrated for this load; tension and pressure labels are marked uncalibrated by the solver.
- Not a real-photo benchmark. Nothing here says how a model trained on it does on a photo.
- Not for redistribution without a licence check: drafts come from GarmentCode presets (MIT),
  bodies from Anny (Apache) with CAESAR-derived plus-size displacement (WEAR licence) when
  `extreme_weight` is used.
"""
