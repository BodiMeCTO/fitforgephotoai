"""The one door to the solver and the body model.

Every call into ``fitforge`` (the solver, github.com/BodiMeCTO/fitforgesolver) or ``annyext``
(the body model, github.com/BodiMeCTO/annyextension) lives here, so the rest of this package
reads as what the dataset needs, and so the list of what the solver must provide is this file.

Only the engine's STABLE surface is used: the drape backends, the QC gate, the fit labels, the
GarmentCode draft, the deterministic rasterizer and the mesh writer. The solver's own old
dataset lane (``fitforge.pipeline``, ``fitforge.surrogate``, ``fitforge.grading``,
``fitforge.retex``) is on its deletion list (solver issue 2026-11-03-dset) and is NOT imported;
the few lines it had that this needs (the ease ladder, the shard writer) are rewritten here.

Note on the boundary law (solver ADR-0043): the API is meant to be the solver's only caller.
This prototype calls the solver in-process instead, because no job kind drafts-and-drapes a
sampled body at a ladder of ease, renders it and returns the labels. The spec in ``docs/``
says what the kind would be.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# No GPU here, ever: the solver's rule is one warp process per machine when a GPU is used,
# and this builder runs several CPU workers. Set before warp is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import annyext  # noqa: E402
from annyext.anny_body import AnnyBody, BodyMesh, PhenotypeParams  # noqa: E402
from annyext.ansur import SampledBody, sample_population  # noqa: E402
from annyext.measurements import (  # noqa: E402
    _CALIB as _TAILOR_CALIB,
    _LEVELS as _TAILOR_LEVELS,
    _blob_perimeter,
    _slice_points,
    tailor_measurements,
)
from annyext.posing import PosedAnnyBody, pose_presets, sample_pose  # noqa: E402
from fitforge.body_model import body_model_commit  # noqa: E402
from fitforge.fitlabels.labels import (  # noqa: E402
    compute_girth_ease,
    extract_labels,
    membrane_p90_contact_kpa,
)
from fitforge.patterns.adapter import make_pattern  # noqa: E402
from fitforge.patterns.drafted_chart import chart_from_pattern  # noqa: E402
from fitforge.patterns.spec_source import design_factory, design_names  # noqa: E402
from fitforge.render.condition_pack import render_condition_pack  # noqa: E402
from fitforge.render.export_pack import LightRig, reshade_from_normals  # noqa: E402
from fitforge.render.mesh_io import write_ply  # noqa: E402
from fitforge.render.raster import Camera  # noqa: E402
from fitforge.sim import registry  # noqa: E402
from fitforge.sim.backend import (  # noqa: E402
    AttachmentSpec,
    DrapeConfig,
    DrapeResult,
    Material,
    tri_panel_id,
)
from fitforge.sim.engine_identity import engine_identity  # noqa: E402
from fitforge.sim.presets import default_drape_config  # noqa: E402
from fitforge.sim.materials import FABRICS, youngs_modulus_pa  # noqa: E402
from fitforge.sim.qc import qc_drape  # noqa: E402
from fitforge.sim.warp_xpbd.collision import build_body_sdf  # noqa: E402
from fitforge.sim.warp_xpbd.topology.weld import weld_settled_seams  # noqa: E402

__all__ = [
    "AttachmentSpec", "BodyMesh", "Camera", "DrapeConfig", "DrapeResult", "LightRig",
    "Material", "PhenotypeParams", "SampledBody", "FABRICS",
]

ANSUR_DIR = Path(annyext.__file__).resolve().parents[1] / "data" / "ansur2"


# ---------------------------------------------------------------------------
# bodies
# ---------------------------------------------------------------------------
def sample_bodies(n: int, seed: int, female_ratio: float = 0.5,
                  tol_cm: float = 1.5) -> list[SampledBody]:
    """``n`` bodies whose height, bust, waist and hips follow the ANSUR II survey (the public
    US Army anthropometry, CSVs in annyextension ``data/ansur2``), each fitted on Anny's
    sliders. Deterministic in ``seed``. A body the sliders cannot reach within ``tol_cm`` is
    kept but flagged ``accepted=False``; the record says so."""
    return sample_population(n, seed=seed, tol_cm=tol_cm, data_dir=ANSUR_DIR,
                             with_mesh=True, female_ratio=female_ratio)


def ansur_source() -> str:
    """Which population the sampler draws from: the real CSVs or the summary-statistics
    fallback annyext uses when they are absent."""
    have = all((ANSUR_DIR / f).exists()
               for f in ("ANSUR_II_MALE_Public.csv", "ANSUR_II_FEMALE_Public.csv"))
    return "ansur2_csv" if have else "ansur2_summary_fallback"


def build_body(phenotype: PhenotypeParams, pose: str | None = None,
               pose_seed: int = 0) -> BodyMesh:
    """The body mesh, in the rest pose or one of annyext's pose presets (jittered by seed)."""
    if pose in (None, "rest"):
        return AnnyBody().build(phenotype)
    return PosedAnnyBody().build_posed(sample_pose(pose, seed=pose_seed), phenotype)


def poses_available() -> list[str]:
    return ["rest", *pose_presets()]


def measure(body: BodyMesh) -> dict[str, float]:
    """The tailor's measurements of a body (cm), the solver's own definitions."""
    return {k: float(v) for k, v in tailor_measurements(body).items()}


def tailor_levels() -> dict[str, float]:
    """Where the tailor measures, as a fraction of stature from the floor."""
    return dict(_TAILOR_LEVELS)


def tailor_calibration() -> dict[str, float]:
    """The tailor's per-level factor from convex outline to girth."""
    return dict(_TAILOR_CALIB)


def slice_points(vertices: np.ndarray, faces_tri: np.ndarray, z: float) -> np.ndarray:
    """Where a mesh's edges cross the plane Z=z, in x, y (the tailor's slice)."""
    return _slice_points(np.asarray(vertices), np.asarray(faces_tri), z)


def blob_perimeter_cm(points_xy: np.ndarray, mode: str = "torso") -> float:
    """The tailor's perimeter of a slice: the convex outline of its central band (cm)."""
    return float(_blob_perimeter(points_xy, mode))


def phenotype_dict(ph: PhenotypeParams) -> dict:
    return asdict(ph)


# ---------------------------------------------------------------------------
# garments
# ---------------------------------------------------------------------------
def designs() -> list[str]:
    return list(design_names())


def draft(design: str, measurements: dict[str, float], name: str):
    """A GarmentCode draft of ``design`` for a body with ``measurements`` (cm)."""
    return make_pattern(design_factory(design), measurements, name=name)


def pattern_chart(pattern) -> dict | None:
    """The drafted pattern's own girths (cm, full), read off the flat panels the way a real
    block is measured: bust 2 cm below the underarm, waist 20 cm, hip 42 cm. None when the
    reader does not know this pattern's panels (trousers)."""
    try:
        chart = chart_from_pattern(pattern)
    except Exception:  # noqa: BLE001 - a chart the reader cannot take is "none", not a crash
        return None
    return {k: float(v) for k, v in chart.items() if isinstance(v, (int, float))} or None


def material_for(fabric: str) -> Material:
    return FABRICS[fabric].to_material()


def fabric_record(fabric: str) -> dict:
    """The fabric as the solver drapes it: the human-facing descriptors and the solver's own
    parameters. ``strain_limit`` is the most the cloth may stretch along the thread (fraction);
    ``stretch_compliance`` how easily it stretches (higher = more easily)."""
    f = FABRICS[fabric]
    m = f.to_material()
    return {
        "name": fabric,
        "category": f.category,
        "areal_weight_gsm": f.areal_weight_gsm,
        "bending_rigidity_uN_m": f.bending_rigidity_uN_m,
        "membrane_stiffness": f.membrane_stiffness,
        "youngs_modulus_pa": youngs_modulus_pa(m),
        "material": asdict(m),
    }


# ---------------------------------------------------------------------------
# drape, QC, labels
# ---------------------------------------------------------------------------
def drape_config(design: str, overrides: dict | None = None) -> DrapeConfig:
    """The solver's PRODUCTION drape settings for a design (the virtual try-on chain's: the
    1 cm mesh, the shoulder band held through the dressing ramp so the garment does not slide
    off, hardened seams), with any overrides on top. The CPU pilot overrides the mesh to 2 cm
    (four times fewer triangles); the record carries the settings used."""
    import dataclasses

    cfg = default_drape_config(design=design, device="cpu")
    return dataclasses.replace(cfg, **(overrides or {}))


def face_panels(result: DrapeResult) -> np.ndarray:
    """Each triangle's panel index."""
    return tri_panel_id(result.faces, result.panel_id, getattr(result, "face_panel_id", None))


def drape(pattern, body: BodyMesh, material: Material, config: DrapeConfig,
          attachments: list[AttachmentSpec] | None = None) -> DrapeResult:
    backend = registry.get_backend(config.backend)
    return backend.drape(pattern, body, material, config, attachments=attachments)


def body_sdf(body: BodyMesh, spacing_cm: float = 2.0):
    return build_body_sdf(body.vertices.astype(np.float32),
                          body.triangulated_faces().astype(np.int32), spacing_cm=spacing_cm)


def qc(result: DrapeResult, body: BodyMesh, sdf, full_length_bottom: bool) -> dict:
    rep = qc_drape(result, body, sdf=sdf, expect_full_length_bottom=full_length_bottom)
    return {
        "passed": bool(rep.passed),
        "fail_reasons": list(rep.diagnostics.get("fail_reasons", [])),
        "penetration_ratio": float(rep.penetration_ratio),
        "max_penetration_depth_cm": float(rep.max_penetration_depth_cm),
        "self_intersections": int(rep.self_intersections),
    }


def labels(result: DrapeResult, body: BodyMesh, sdf, material: Material) -> dict:
    """The solver's fit labels on the UN-welded mesh (welding averages seam cuts and would
    inflate the strain). ``ease_measured_cm`` is the garment's outer girth on the settled mesh
    minus the body's at the same height: the solver's own note says this over-reads on loose
    garments (folds, sleeves), so it is kept as a measured label beside the exact one."""
    lab = extract_labels(result, body, sdf=sdf, material=material)
    measured = compute_girth_ease(result, body)  # no target measurements: read off the mesh
    strain = np.asarray(lab.strain_pct)
    return {
        "ease_measured_cm": {k: float(v) for k, v in measured.items()},
        "strain_pct_p50": round(float(np.percentile(strain, 50)), 3),
        "strain_pct_p95": round(float(np.percentile(strain, 95)), 3),
        "air_gap_mm_p50": round(float(np.percentile(lab.air_gap_mm, 50)), 2),
        "penetration_ratio": round(float(lab.penetration_ratio), 4),
        "fit_class": dict(lab.fit_class),
        "per_region": lab.per_region,
        "membrane_pressure_p90_kpa": membrane_p90_contact_kpa(lab.pressure_membrane_kpa),
        "provenance": dict(lab.provenance),
    }


def weld(result: DrapeResult, tol_cm: float = 1.5) -> DrapeResult:
    return weld_settled_seams(result, tol_cm=tol_cm)


# ---------------------------------------------------------------------------
# renders and files
# ---------------------------------------------------------------------------
def render(body: BodyMesh, result: DrapeResult, camera: Camera, rig: LightRig,
           garment_rgb: tuple[int, int, int], body_rgb: tuple[int, int, int],
           bg_value: int) -> dict[str, np.ndarray]:
    """One view of body and garment: the shaded picture plus the geometry passes."""
    out, seg = render_condition_pack(
        body, result.vertices, result.faces, result.panel_id, camera,
        face_panel_id=getattr(result, "face_panel_id", None))
    rgb = reshade_from_normals(out.normal, seg, rig, garment_rgb=garment_rgb,
                               body_rgb=body_rgb, bg_value=bg_value)
    return {"rgb": rgb, "seg": seg, "normal": out.normal, "depth": out.depth,
            "silhouette": out.silhouette}


def save_ply(path: Path, vertices: np.ndarray, faces: np.ndarray,
             colours: np.ndarray | None = None) -> Path:
    return write_ply(path, vertices, faces, colours)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def identity(config: DrapeConfig) -> dict:
    """Which engine, body model and builder made a sample."""
    import fitforge

    return {
        "engine": engine_identity(config),
        "solver_commit": _git_commit(Path(fitforge.__file__).resolve().parents[1]),
        "annyext_commit": body_model_commit(),
        "annyext_path": str(Path(annyext.__file__).resolve().parent),
        "photoai_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "ansur_source": ansur_source(),
        "drape_config": asdict(config),
    }


def now() -> float:
    return time.time()
