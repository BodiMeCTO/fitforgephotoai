"""Girths read off the settled garment, torso panels only, by the tailor's own rule.

The solver's own mesh-measured ease takes the garment's outer contour at the body's bust,
waist and hip heights over EVERY garment vertex, so a sleeve hanging past the waist makes the
"waist girth" the sleeve's, and a shirt comes out 20 cm smaller than the body it covers. Here
the section is taken over the torso panels alone (a panel is a torso panel when its name says
so: GarmentCode names them ``*ftorso`` / ``*btorso``), and only where those panels span the
height. The perimeter is then the body model's own tailor rule (``annyext.measurements``: the
convex outline of the central band of the section, times its per-level calibration), so a
garment girth and the body's tailor measurement at the same level subtract like for like.
On a loose garment the convex outline reads the outside of the folds, as a tape would.
"""
from __future__ import annotations

import numpy as np

from photoai.engine import blob_perimeter_cm, slice_points, tailor_calibration, tailor_levels

#: the three levels bodi.me's ease is given at
POINTS = ("bust", "waist", "hips")


def torso_panel_ids(panel_names: list[str]) -> set[int]:
    return {i for i, n in enumerate(panel_names) if "torso" in n.lower()}


def garment_girths_cm(garment_vertices: np.ndarray, garment_faces: np.ndarray,
                      face_panel: np.ndarray, panel_names: list[str],
                      body_vertices: np.ndarray) -> dict[str, float]:
    """The torso-only girth of the garment (cm) at the body's bust, waist and hip heights,
    by the tailor's rule; a level the torso panels do not span is left out."""
    torso = torso_panel_ids(panel_names)
    keep = np.isin(face_panel, list(torso)) if torso else np.ones(len(garment_faces), bool)
    faces = np.asarray(garment_faces)[keep]
    if len(faces) == 0:
        return {}
    used = np.unique(faces)
    bz = body_vertices[:, 2]
    zmin, stature = float(bz.min()), float(bz.max() - bz.min())
    gz = garment_vertices[used, 2]
    levels, calib = tailor_levels(), tailor_calibration()
    out = {}
    for name in POINTS:
        z = zmin + levels[name] * stature
        if not (gz.min() + 0.01 <= z <= gz.max() - 0.01):
            continue
        pts = slice_points(garment_vertices, faces, z)
        g = blob_perimeter_cm(pts, "torso") * calib.get(name, 1.0)
        if g > 0:
            out[name] = round(float(g), 2)
    return out
