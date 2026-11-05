"""Fast checks that need neither the solver nor a body."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photoai.garments import designed_ease_cm, ease_ladder, pattern_ease_cm  # noqa: E402


def test_ease_ladder_moves_every_girth_by_the_offset():
    base = {"bust": 90.0, "underbust": 80.0, "waist": 75.0, "hips": 98.0, "height": 170.0,
            "back_width": 40.0, "shoulder_w": 30.0}
    steps = ease_ladder(base, (-8.0, 0.0, 16.0))
    assert [s.label for s in steps] == ["ease-8", "ease+0", "ease+16"]
    assert steps[0].measurements["bust"] == 82.0 and steps[2].measurements["hips"] == 114.0
    assert steps[1].measurements["height"] == 170.0
    assert abs(steps[2].measurements["back_width"] - 0.46 * 106.0) < 1e-9


def test_designed_ease_is_garment_minus_body():
    assert designed_ease_cm({"bust": 94.0, "waist": 79.0, "hips": 102.0},
                            {"bust": 90.0, "waist": 75.0, "hips": 98.0}) == {
        "bust": 4.0, "waist": 4.0, "hips": 4.0}


def test_pattern_ease_reads_the_chart_keys():
    assert pattern_ease_cm({"bust_girth": 100.0}, {"bust": 90.0, "waist": 1.0}) == {"bust": 10.0}
    assert pattern_ease_cm(None, {"bust": 90.0}) is None


def test_grid_is_deterministic_and_complete():
    from photoai.sweep import build_grid, default_config, shard_of

    cfg = default_config()
    cfg.update({"n_bodies": 3, "designs": ["shirt"], "ease_offsets_cm": [0.0, 4.0],
                "fabrics": ["denim", "jersey_knit"]})
    a, b = build_grid(cfg), build_grid(cfg)
    assert a == b and len(a) == 3 * 1 * 2 * 2
    assert len({j.sample_id for j in a}) == len(a)
    parts = [shard_of(a, i, 4) for i in range(4)]
    assert sorted(j.sample_id for p in parts for j in p) == sorted(j.sample_id for j in a)


def test_cameras_and_look_are_seeded():
    from photoai.render import sample_look

    rng1, rng2 = np.random.default_rng(3), np.random.default_rng(3)
    assert sample_look(rng1)["garment_rgb"] == sample_look(rng2)["garment_rgb"]
