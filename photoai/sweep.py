"""The grid (body x pose x design x ease x fabric), one sample's run, and the sharded runner.

A run is deterministic in its config and seed: every worker builds the same grid and takes
its own slice of it (``shard`` of ``n_shards``). A sample that has already been written
(``record.json`` exists) is skipped, so a run can be stopped and resumed. A failure is a
sample too: its record says what failed (``status: failed``) and the renders are absent.

Output layout (``out``):
    bodies/<body_id>/body.ply, body.json      the body, once per body
    samples/<sample_id>/record.json           every label and every provenance field
    samples/<sample_id>/garment.ply           the settled garment, seams welded, metres
    samples/<sample_id>/<view>.rgb.png        the shaded picture
    samples/<sample_id>/<view>.seg.png        0 background, 1 body, 2+k garment panel k
    samples/<sample_id>/<view>.normal.png     world-space normals, 8 bit
    samples/<sample_id>/<view>.depth.png      depth in the silhouette, 16 bit
    shards/manifest-<shard>.jsonl             one line per sample this worker wrote
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from photoai import engine
from photoai.garments import (
    DEFAULT_DESIGNS,
    DEFAULT_EASE_OFFSETS_CM,
    ELASTICITY_LADDER,
    FULL_LENGTH_BOTTOM,
    WAIST_HUNG,
    designed_ease_cm,
    ease_ladder,
    pattern_ease_cm,
)
from photoai.measure import body_girths_cm, garment_girths_cm
from photoai.render import (
    DEFAULT_VIEWS,
    View,
    camera_for,
    encode_depth_png16,
    encode_normal_png,
    look_record,
    sample_look,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Job:
    sample_id: str
    body_index: int
    pose: str
    design: str
    ease_label: str
    ease_offset_cm: float
    fabric: str
    seed: int


def default_config() -> dict:
    return {
        "seed": 0,
        "n_bodies": 4,
        "female_ratio": 0.5,
        "body_tol_cm": 1.5,
        "poses": ["rest"],
        "designs": list(DEFAULT_DESIGNS),
        "ease_offsets_cm": list(DEFAULT_EASE_OFFSETS_CM),
        "fabrics": list(ELASTICITY_LADDER),
        "views": [asdict(v) for v in DEFAULT_VIEWS],
        "image": {"width": 384, "height": 512},
        # on top of the solver's production settings (photoai.engine.drape_config): the CPU
        # pilot's 2 cm mesh; {} = the production 1 cm mesh
        "drape": {"edge_length_cm": 2.0},
        "weld_tol_cm": 1.5,
        "save_body_ply": True,
    }


def load_config(path: str | Path | None) -> dict:
    cfg = default_config()
    if path:
        import yaml

        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in user.items():
            if k == "image" and isinstance(v, dict):
                cfg["image"].update(v)
            else:
                cfg[k] = v
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def build_grid(cfg: dict) -> list[Job]:
    """Every (body, pose, design, ease, fabric) once, in a fixed order. Bodies are not built
    here (that is the worker's job for its own slice); only the indices are laid out."""
    jobs: list[Job] = []
    chash = config_hash(cfg)
    for bi in range(int(cfg["n_bodies"])):
        for pose in cfg["poses"]:
            for design in cfg["designs"]:
                for off in cfg["ease_offsets_cm"]:
                    sign = "+" if off >= 0 else ""
                    elabel = f"ease{sign}{off:g}"
                    for fabric in cfg["fabrics"]:
                        key = f"{chash}|{bi}|{pose}|{design}|{elabel}|{fabric}"
                        sid = hashlib.sha256(key.encode()).hexdigest()[:12]
                        jobs.append(Job(sample_id=sid, body_index=bi, pose=pose, design=design,
                                        ease_label=elabel, ease_offset_cm=float(off),
                                        fabric=fabric, seed=int(cfg["seed"])))
    return jobs


def shard_of(jobs: list[Job], shard: int, n_shards: int) -> list[Job]:
    return [j for i, j in enumerate(jobs) if i % n_shards == shard]


# ---------------------------------------------------------------------------
# bodies for a slice
# ---------------------------------------------------------------------------
class BodyStore:
    """The sampled bodies of one run, built once per worker and written once per run."""

    def __init__(self, cfg: dict, out: Path):
        self.cfg = cfg
        self.out = out
        self._sampled = None
        self._posed: dict[tuple[int, str], object] = {}
        self._sdf: dict[tuple[int, str], object] = {}

    def sampled(self):
        if self._sampled is None:
            self._sampled = engine.sample_bodies(
                int(self.cfg["n_bodies"]), seed=int(self.cfg["seed"]),
                female_ratio=float(self.cfg["female_ratio"]),
                tol_cm=float(self.cfg["body_tol_cm"]))
        return self._sampled

    def body_id(self, index: int) -> str:
        return f"b{int(self.cfg['seed']):03d}_{index:04d}"

    def body(self, index: int, pose: str):
        key = (index, pose)
        if key not in self._posed:
            sb = self.sampled()[index]
            if pose == "rest" and sb.mesh is not None:
                mesh = sb.mesh
            else:
                mesh = engine.build_body(sb.phenotype, pose,
                                         pose_seed=int(self.cfg["seed"]) * 1000 + index)
            self._posed[key] = mesh
            self._write(index, pose, mesh)
        return self._posed[key]

    def contour_girths(self, index: int, pose: str) -> dict:
        body = self.body(index, pose)
        return body_girths_cm(body.vertices, body.triangulated_faces())

    def sdf(self, index: int, pose: str):
        key = (index, pose)
        if key not in self._sdf:
            self._sdf[key] = engine.body_sdf(self.body(index, pose))
        return self._sdf[key]

    def record(self, index: int, pose: str) -> dict:
        sb = self.sampled()[index]
        mesh = self.body(index, pose)
        return {
            "body_id": self.body_id(index),
            "pose": pose,
            "phenotype": engine.phenotype_dict(sb.phenotype),
            "target_cm": {k: round(float(v), 2) for k, v in sb.target.items()
                          if isinstance(v, (int, float))},
            "realised_cm": {k: round(float(v), 2) for k, v in sb.realised.items()
                            if isinstance(v, (int, float))},
            "target_source": str(sb.target.get("source", "")),
            "max_error_cm": round(float(sb.max_error_cm), 2),
            "accepted": bool(sb.accepted),
            "measurements_cm": {k: round(v, 2) for k, v in engine.measure(mesh).items()},
            "population": engine.ansur_source(),
        }

    def _write(self, index: int, pose: str, mesh) -> None:
        d = self.out / "bodies" / self.body_id(index)
        d.mkdir(parents=True, exist_ok=True)
        rec = d / f"body_{pose}.json"
        if not rec.exists():
            rec.write_text(json.dumps(self.record(index, pose), indent=1), encoding="utf-8")
        ply = d / f"body_{pose}.ply"
        if self.cfg.get("save_body_ply", True) and not ply.exists():
            engine.save_ply(ply, mesh.vertices, mesh.triangulated_faces())


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def _png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def run_sample(job: Job, cfg: dict, bodies: BodyStore, out: Path) -> dict:
    """Draft, drape, label and render one sample; write its folder; return its record."""
    t_all = time.time()
    sdir = out / "samples" / job.sample_id
    sdir.mkdir(parents=True, exist_ok=True)
    drape_cfg = engine.drape_config(job.design, cfg.get("drape", {}))
    rec: dict = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": job.sample_id,
        "status": "failed",
        "job": asdict(job),
        "provenance": engine.identity(drape_cfg),
        "timings_s": {},
    }
    try:
        body = bodies.body(job.body_index, job.pose)
        body_rec = bodies.record(job.body_index, job.pose)
        body_meas = body_rec["measurements_cm"]
        rec["body"] = body_rec

        # the garment: drafted for the body shifted by the ease offset
        step = next(s for s in ease_ladder(body_meas, (job.ease_offset_cm,)))
        t = time.time()
        pattern = engine.draft(job.design, step.measurements, name=job.sample_id)
        chart = engine.pattern_chart(pattern)
        rec["timings_s"]["draft"] = round(time.time() - t, 2)
        rec["garment"] = {
            "design": job.design,
            "ease_label": job.ease_label,
            "ease_offset_cm": job.ease_offset_cm,
            "drafted_for_cm": {k: round(float(step.measurements[k]), 2)
                               for k in ("bust", "underbust", "waist", "hips")
                               if k in step.measurements},
            "pattern_chart_cm": chart,
            "n_panels": len(pattern.panels),
            "n_stitches": len(pattern.stitches),
        }
        rec["fabric"] = engine.fabric_record(job.fabric)
        rec["fabric"]["elasticity_class"] = ELASTICITY_LADDER[job.fabric]

        # the drape
        material = engine.material_for(job.fabric)
        attachments = ([engine.AttachmentSpec(top_band_cm=3.0, landmark="waist")]
                       if job.design in WAIST_HUNG else None)
        t = time.time()
        result = engine.drape(pattern, body, material, drape_cfg, attachments=attachments)
        rec["timings_s"]["drape"] = round(time.time() - t, 2)
        rec["drape"] = {"converged": bool(result.converged), "steps_run": int(result.steps_run),
                        "n_vertices": int(len(result.vertices)),
                        "n_faces": int(len(result.faces))}

        # QC and labels on the un-welded mesh
        sdf = bodies.sdf(job.body_index, job.pose)
        t = time.time()
        rec["qc"] = engine.qc(result, body, sdf, job.design in FULL_LENGTH_BOTTOM)
        lab = engine.labels(result, body, sdf, material)
        panel_names = [p.name for p in pattern.panels]
        g_girth = garment_girths_cm(result.vertices, result.faces, engine.face_panels(result),
                                    panel_names, body.vertices)
        b_girth = bodies.contour_girths(job.body_index, job.pose)
        rec["timings_s"]["qc_labels"] = round(time.time() - t, 2)
        rec["labels"] = {
            "ease_designed_cm": designed_ease_cm(step.measurements, body_meas),
            # bust only: the solver's chart reader clamps waist and hip to the hem on a drafted
            # shirt (issue 2026-11-05-chrt); the full chart is in the garment block
            "ease_pattern_cm": ({"bust": pattern_ease_cm(chart, body_meas)["bust"]}
                                if pattern_ease_cm(chart, body_meas)
                                and "bust" in pattern_ease_cm(chart, body_meas) else None),
            "garment_girth_torso_cm": g_girth,
            "body_girth_contour_cm": b_girth,
            "ease_worn_cm": {k: round(g_girth[k] - b_girth[k], 2) for k in g_girth
                             if k in b_girth},
            "elasticity_class": ELASTICITY_LADDER[job.fabric],
            "elasticity_strain_limit": rec["fabric"]["material"]["strain_limit"],
            "elasticity_stretch_compliance": rec["fabric"]["material"]["stretch_compliance"],
            **lab,
        }

        # the saved mesh: welded for a clean surface
        welded = engine.weld(result, tol_cm=float(cfg.get("weld_tol_cm", 1.5)))
        engine.save_ply(sdir / "garment.ply", welded.vertices, welded.faces)
        rec["files"] = {"garment_ply": f"samples/{job.sample_id}/garment.ply",
                        "body_ply": f"bodies/{bodies.body_id(job.body_index)}/body_{job.pose}.ply",
                        "views": {}}

        # the renders
        rng = np.random.default_rng(int(hashlib.sha256(job.sample_id.encode()).hexdigest(), 16)
                                    % (2**32))
        look = sample_look(rng)
        rec["look"] = look_record(look)
        W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
        t = time.time()
        for v in cfg["views"]:
            view = View(**v)
            cam = camera_for(body, view, look, W, H)
            passes = engine.render(body, welded, cam, look["rig"], look["garment_rgb"],
                                   look["body_rgb"], look["bg_value"])
            depth16, drange = encode_depth_png16(passes["depth"], passes["silhouette"])
            _png(sdir / f"{view.name}.rgb.png", passes["rgb"])
            _png(sdir / f"{view.name}.seg.png", passes["seg"])
            _png(sdir / f"{view.name}.normal.png",
                 encode_normal_png(passes["normal"], passes["silhouette"]))
            _png(sdir / f"{view.name}.depth.png", depth16)
            rec["files"]["views"][view.name] = {
                "rgb": f"samples/{job.sample_id}/{view.name}.rgb.png",
                "seg": f"samples/{job.sample_id}/{view.name}.seg.png",
                "normal": f"samples/{job.sample_id}/{view.name}.normal.png",
                "depth": f"samples/{job.sample_id}/{view.name}.depth.png",
                "depth_range_m": drange,
                "azimuth_deg": view.azimuth_deg + look["azimuth_jitter_deg"],
                "framing": view.framing,
                "camera": {"eye": [round(float(x), 4) for x in cam.eye],
                           "target": [round(float(x), 4) for x in cam.target],
                           "fov_deg": cam.fov_deg, "width": W, "height": H},
                "garment_pixels": int((passes["seg"] >= 2).sum()),
            }
        rec["timings_s"]["render"] = round(time.time() - t, 2)
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a failed sample is a record, never a crash
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()[-4000:]
    rec["timings_s"]["total"] = round(time.time() - t_all, 2)
    (sdir / "record.json").write_text(json.dumps(rec, indent=1, default=_json_default),
                                      encoding="utf-8")
    return rec


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
def run(cfg: dict, out: str | Path, shard: int = 0, n_shards: int = 1,
        limit: int | None = None, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    bad = [p for p in cfg["poses"] if p != "rest"]
    if bad:  # the slice rule's arm band assumes the rest pose (review finding 4; spec S8)
        raise ValueError(f"only the rest pose is supported until the worn girth handles a posed "
                         f"body (spec S8); got {bad}")
    jobs = shard_of(build_grid(cfg), shard, n_shards)
    if limit:
        jobs = jobs[:limit]
    bodies = BodyStore(cfg, out)
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    manifest = shards / f"manifest-{shard:03d}.jsonl"
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    with open(manifest, "a", encoding="utf-8") as mf:
        for i, job in enumerate(jobs):
            rec_path = out / "samples" / job.sample_id / "record.json"
            if rec_path.exists():
                n_skip += 1
                continue
            rec = run_sample(job, cfg, bodies, out)
            mf.write(json.dumps(summary_row(rec), default=_json_default) + "\n")
            mf.flush()
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            log(f"[{shard}/{n_shards}] {i + 1}/{len(jobs)} {job.sample_id} {job.design} "
                f"{job.ease_label} {job.fabric} -> {rec['status']} "
                f"{rec['timings_s'].get('total')}s"
                + (f" ({rec.get('error')})" if rec["status"] != "ok" else ""))
    return {"shard": shard, "n_shards": n_shards, "jobs": len(jobs), "ok": n_ok,
            "skipped": n_skip, "failed": n_fail, "seconds": round(time.time() - t0, 1),
            "manifest": str(manifest)}


def summary_row(rec: dict) -> dict:
    """The manifest line: the labels and the facts, no per-region detail."""
    row = {
        "sample_id": rec["sample_id"], "status": rec["status"], **rec["job"],
    }
    if rec["status"] != "ok":
        row["error"] = rec.get("error")
        return row
    row.update({
        "body_id": rec["body"]["body_id"],
        "body_accepted": rec["body"]["accepted"],
        "gender": rec["body"]["phenotype"]["gender"],
        "body_cm": {k: rec["body"]["measurements_cm"].get(k) for k in ("height", "bust", "waist", "hips")},
        "ease_designed_cm": rec["labels"]["ease_designed_cm"],
        "ease_pattern_cm": rec["labels"]["ease_pattern_cm"],
        "ease_worn_cm": rec["labels"]["ease_worn_cm"],
        "ease_measured_cm": rec["labels"]["ease_measured_cm"],
        "elasticity_class": rec["labels"]["elasticity_class"],
        "elasticity_strain_limit": rec["labels"]["elasticity_strain_limit"],
        "strain_pct_p50": rec["labels"]["strain_pct_p50"],
        "strain_pct_p95": rec["labels"]["strain_pct_p95"],
        "fit_class": rec["labels"]["fit_class"],
        "qc_passed": rec["qc"]["passed"],
        "qc_fail_reasons": rec["qc"]["fail_reasons"],
        "converged": rec["drape"]["converged"],
        "steps_run": rec["drape"]["steps_run"],
        "timings_s": rec["timings_s"],
    })
    return row
