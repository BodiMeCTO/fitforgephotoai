"""The garment side of the grid: which designs, the ease ladder, the elasticity ladder.

Ease here is the room the garment has over the body it is drafted for, in cm of full girth at
bust, waist and hips. The ladder drafts the garment for a body ``offset`` cm larger (or
smaller) than the one it is draped on, so the same person wears the garment at every ease
from much too tight to very loose. That is the one axis a photo model must learn to read.

Elasticity is the fabric's stretch. The solver's fabric library gives each fabric a maximum
stretch along the thread (``strain_limit``) and a stretch compliance; the ladder picks four
fabrics that span rigid woven to stretch knit, so the same cut at the same ease is draped in
each, and the picture differs only by what the cloth does under load.
"""
from __future__ import annotations

from dataclasses import dataclass

#: ease offsets (cm, full girth at bust/waist/hips), tight to loose
DEFAULT_EASE_OFFSETS_CM = (-8.0, -4.0, 0.0, 4.0, 8.0, 16.0)

#: the girth keys an ease offset moves
GIRTH_KEYS = ("bust", "underbust", "waist", "hips")

#: the elasticity ladder, in order of increasing stretch: a fabric name from the solver's
#: library and the class the label carries. Numbers (strain_limit, compliance) are read from
#: the library at build time, never typed here (photoai.engine.fabric_record).
ELASTICITY_LADDER: dict[str, int] = {
    "denim": 0,          # rigid woven
    "cotton_poplin": 1,  # woven
    "cotton_jersey": 2,  # firm knit (a tee)
    "jersey_knit": 3,    # stretch knit
}

#: designs that hang from the waist need a waistband hold or they slide off the body
WAIST_HUNG = {"trousers"}
#: full-length bottoms whose hem must reach the feet (the solver's cropped-bottom QC gate);
#: a future "shorts" is waist-hung but NOT full-length
FULL_LENGTH_BOTTOM = {"trousers"}

#: the solver's GarmentCode presets that draft reliably over the body range (its own note):
#: the boxy tee and the body-following fitted shirt together span tight to loose
DEFAULT_DESIGNS = ("shirt", "shirt_fitted")


@dataclass(frozen=True)
class EaseStep:
    label: str
    offset_cm: float
    measurements: dict  # the measurements the garment is drafted for


def ease_ladder(body_measurements: dict[str, float],
                offsets_cm: tuple[float, ...] = DEFAULT_EASE_OFFSETS_CM) -> list[EaseStep]:
    """The body's measurements shifted by each offset on the girths. Widths follow the bust
    so the draft stays self-consistent (the same rule the solver's old ladder used)."""
    out = []
    for off in offsets_cm:
        m = dict(body_measurements)
        for k in GIRTH_KEYS:
            if k in m:
                m[k] = float(m[k]) + off
        if "back_width" in m and "bust" in m:
            m["back_width"] = 0.46 * m["bust"]
            m["shoulder_w"] = 0.76 * m["back_width"]
        sign = "+" if off >= 0 else ""
        out.append(EaseStep(label=f"ease{sign}{off:g}", offset_cm=float(off), measurements=m))
    return out


def designed_ease_cm(garment_measurements: dict, body_measurements: dict) -> dict[str, float]:
    """What the garment was drafted for minus what the body is, per girth (cm)."""
    return {k: round(float(garment_measurements[k]) - float(body_measurements[k]), 2)
            for k in ("bust", "waist", "hips")
            if k in garment_measurements and k in body_measurements}


def pattern_ease_cm(chart: dict | None, body_measurements: dict) -> dict[str, float] | None:
    """The drafted pattern's own girth minus the body's: the garment's TRUE room over the body,
    including the ease GarmentCode builds into the design itself. None without a chart."""
    if not chart:
        return None
    pairs = (("bust", "bust_girth"), ("waist", "waist_girth"), ("hips", "hip_girth"))
    out = {b: round(float(chart[g]) - float(body_measurements[b]), 2)
           for b, g in pairs if g in chart and b in body_measurements}
    return out or None
