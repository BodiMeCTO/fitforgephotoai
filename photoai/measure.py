"""Girths read off the settled garment, torso panels only, by the tailor's own rule.

The solver's own mesh-measured ease takes the garment's outer contour at the body's bust,
waist and hip heights over EVERY garment vertex, so a sleeve hanging past the waist makes the
"waist girth" the sleeve's, and a shirt comes out 20 cm smaller than the body it covers. Here
the section is taken over the torso panels alone (a panel is a torso panel when its name says
so: GarmentCode names them ``*ftorso`` / ``*btorso``), and only where those panels span the
height. The perimeter is the body model's own slice rule (``annyext.measurements``: the convex
outline of the central band of the section) WITHOUT the tailor's per-level calibration factor,
which was fitted to correct the body's hull against anatomical references and has no basis on
cloth (review 2026-11-05, finding 1). The body is measured by the same rule, at the same level,
on the same mesh (``body_girths_cm``), so the two subtract like for like; the tailor's own
measurements stay in the record beside them. On a loose garment the convex outline reads the
outside of the folds, as a tape would. Known limit: the rule's 0.20 m central band clips a
section wider than 40 cm, so very loose garments and very large bodies read low (finding 3);
and the band assumes the rest pose, so posed bodies are refused by the runner (finding 4).
"""
from __future__ import annotations

import numpy as np

from photoai.engine import blob_perimeter_cm, slice_points, tailor_levels

#: the three levels bodi.me's ease is given at
POINTS = ("bust", "waist", "hips")


def torso_panel_ids(panel_names: list[str]) -> set[int]:
    return {i for i, n in enumerate(panel_names) if "torso" in n.lower()}


def garment_girths_cm(garment_vertices: np.ndarray, garment_faces: np.ndarray,
                      face_panel: np.ndarray, panel_names: list[str],
                      body_vertices: np.ndarray) -> dict[str, float]:
    """The torso-only girth of the garment (cm) at the body's bust, waist and hip heights,
    by the slice rule; a level the torso panels do not span is left out. A pattern with no
    panel named as a torso gives no girth at all (never the sleeves instead)."""
    torso = torso_panel_ids(panel_names)
    if not torso:
        return {}
    keep = np.isin(face_panel, list(torso))
    faces = np.asarray(garment_faces)[keep]
    if len(faces) == 0:
        return {}
    used = np.unique(faces)
    bz = body_vertices[:, 2]
    zmin, stature = float(bz.min()), float(bz.max() - bz.min())
    gz = garment_vertices[used, 2]
    levels = tailor_levels()
    out = {}
    for name in POINTS:
        z = zmin + levels[name] * stature
        if not (gz.min() + 0.01 <= z <= gz.max() - 0.01):
            continue
        g = blob_perimeter_cm(slice_points(garment_vertices, faces, z), "torso")
        if g > 0:
            out[name] = round(float(g), 2)
    return out


def body_girths_cm(body_vertices: np.ndarray, body_faces: np.ndarray) -> dict[str, float]:
    """The body's girth at the same levels by the same slice rule, uncalibrated, so the worn
    ease is garment minus body like for like."""
    bz = body_vertices[:, 2]
    zmin, stature = float(bz.min()), float(bz.max() - bz.min())
    levels = tailor_levels()
    out = {}
    for name in POINTS:
        g = blob_perimeter_cm(slice_points(body_vertices, body_faces, zmin + levels[name] * stature),
                              "torso")
        if g > 0:
            out[name] = round(float(g), 2)
    return out
