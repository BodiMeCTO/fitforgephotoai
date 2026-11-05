"""Cameras, lights and colours for the renders: a few catalogue-like views of each sample with
the lighting, background, garment colour and camera distance varied per sample, so a model
trained on them cannot lean on any of those staying fixed.

The pictures are shaded from the mesh normals (Lambertian, two lights and an ambient floor),
not photoreal. They show silhouette, drape folds and where the cloth pulls, which is the
signal of interest; skin, hair, fabric texture and real light are not in them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photoai.engine import BodyMesh, Camera, LightRig


@dataclass(frozen=True)
class View:
    name: str
    azimuth_deg: float
    framing: str  # full | torso


DEFAULT_VIEWS = (
    View("front", 0.0, "full"),
    View("three_quarter", 40.0, "full"),
    View("side", 90.0, "full"),
    View("front_torso", 0.0, "torso"),
)

GARMENT_PALETTE = [
    (60, 110, 170), (170, 60, 60), (70, 140, 80), (190, 150, 50), (110, 80, 150),
    (40, 40, 45), (230, 230, 225), (150, 95, 60), (200, 120, 150), (80, 150, 150),
]
BODY_TONES = [(224, 190, 165), (200, 160, 130), (170, 125, 95), (130, 90, 65), (95, 65, 45)]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def sample_look(rng: np.random.Generator) -> dict:
    """One sample's lighting, colours and background, drawn once per sample."""
    key = _unit([rng.uniform(-1.2, 1.2), rng.uniform(-1.0, -0.2), rng.uniform(0.1, 0.9)])
    fill = _unit([-key[0], rng.uniform(-0.8, -0.2), rng.uniform(0.0, 0.4)])
    rig = LightRig(key=key, fill=fill,
                   ambient=float(rng.uniform(0.10, 0.30)),
                   k_key=float(rng.uniform(0.7, 1.0)),
                   k_fill=float(rng.uniform(0.1, 0.4)))
    return {
        "rig": rig,
        "garment_rgb": GARMENT_PALETTE[int(rng.integers(len(GARMENT_PALETTE)))],
        "body_rgb": BODY_TONES[int(rng.integers(len(BODY_TONES)))],
        "bg_value": int(rng.integers(200, 256)),
        "distance_scale": float(rng.uniform(0.9, 1.15)),
        "azimuth_jitter_deg": float(rng.uniform(-8, 8)),
        "height_jitter": float(rng.uniform(-0.03, 0.06)),
    }


def look_record(look: dict) -> dict:
    rig = look["rig"]
    return {
        "light_key": [round(float(x), 4) for x in rig.key],
        "light_fill": [round(float(x), 4) for x in rig.fill],
        "ambient": rig.ambient, "k_key": rig.k_key, "k_fill": rig.k_fill,
        "garment_rgb": list(look["garment_rgb"]), "body_rgb": list(look["body_rgb"]),
        "bg_value": look["bg_value"], "distance_scale": look["distance_scale"],
        "azimuth_jitter_deg": look["azimuth_jitter_deg"],
        "height_jitter": look["height_jitter"],
    }


def camera_for(body: BodyMesh, view: View, look: dict, width: int, height: int) -> Camera:
    """A pinhole camera orbiting the body at a stature-scaled distance. Z is up."""
    bv = body.vertices
    zmin, zmax = float(bv[:, 2].min()), float(bv[:, 2].max())
    stature = zmax - zmin
    cx, cy = float(np.median(bv[:, 0])), float(np.median(bv[:, 1]))
    if view.framing == "full":
        dist = stature * 1.9 * look["distance_scale"]
        target = np.array([cx, cy, zmin + 0.52 * stature])
        fov = 38.0
    else:
        dist = stature * 1.1 * look["distance_scale"]
        target = np.array([cx, cy, zmin + 0.68 * stature])
        fov = 34.0
    az = np.radians(view.azimuth_deg + look["azimuth_jitter_deg"])
    eye = target + np.array([np.sin(az) * dist, -np.cos(az) * dist,
                             (0.06 + look["height_jitter"]) * stature])
    return Camera(eye=eye, target=target, up=np.array([0.0, 0.0, 1.0]),
                  fov_deg=fov, width=width, height=height)


def encode_normal_png(normal: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
    img = ((np.clip(normal, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    img[silhouette == 0] = 0
    return img


def encode_depth_png16(depth: np.ndarray, silhouette: np.ndarray) -> tuple[np.ndarray, dict]:
    """Depth inside the silhouette scaled to 16 bits; the record keeps the range (m)."""
    mask = silhouette > 0
    if not mask.any():
        return np.zeros(depth.shape, dtype=np.uint16), {"min_m": 0.0, "max_m": 0.0}
    lo, hi = float(depth[mask].min()), float(depth[mask].max())
    scaled = np.zeros(depth.shape, dtype=np.uint16)
    span = max(hi - lo, 1e-6)
    scaled[mask] = (1 + (depth[mask] - lo) / span * 65000).astype(np.uint16)
    return scaled, {"min_m": round(lo, 5), "max_m": round(hi, 5)}
