"""Parameter blending vs pose blending.

The paper blends Pi (Sec. 5.2) and re-runs the planner.  The obvious cheaper
alternative -- interpolating the planner's joint-angle *output* at phase-
registered times -- is what these tests pin down as wrong, so that nobody
"optimises" the pipeline back into it later.

Why it is wrong: foot height is FK(q), a nonlinear function of the joint
angles.  Two clips can each hold a foot flat on the ground while the average
of their joint angles does not.  Blending Pi works because ground contact is a
linear property of the control points, so a convex combination inherits it.

The strongest possible statement of the claim is that a blended sample is not
merely close to valid, it *is* a valid clip -- so the first test runs the
blend through the writer and then through the same verifier the nine authored
clips are held to.
"""
import os

import numpy as np
import pytest

from stylized_gait import blend as GB, planner as GP, samples as SM
from stylized_gait import usd_writer as UW, verify as V
import blend_compare as BC

PAIRS = [("forward_slow", "left_turn", 0.5),
         ("forward_fast", "forward_slow", 0.35),
         ("left_strafe", "forward_fast", 0.5),
         ("reverse_slow", "right_turn", 0.5)]
IDS = [f"{a}+{b}@{w}" for a, b, w in PAIRS]


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_a_blended_sample_is_itself_a_valid_clip(pair, tmp_path):
    """the whole point of blending parameters: the result satisfies every
    constraint from scratch, because the planner re-enforced them."""
    a, b, w = pair
    g = SM.build()
    A, B = g[a], g[b]
    vel = (1 - w) * A.vel + w * B.vel
    blend = GB.BlendedSample([A, B], [1 - w, w], vel=vel, name="blend")

    res = GP.solve_sample(blend, fps=60, passes=3)
    p = str(tmp_path / "blend.usda")
    UW.write_usda(res, p, fps=60)

    _, info, issues = V.check(p)
    assert not issues, "; ".join(issues)
    assert info["min_sole_z_cm"] >= -0.01
    assert info["max_stance_slip_mm"] < 0.5
    assert info["loop_quat_delta"] < 1e-6


def test_pose_blending_measurably_breaks_contact():
    """the contrast that makes the test above mean something.  If this ever
    stops failing, the comparison is broken, not fixed."""
    case = BC.run_case("forward_slow", "left_turn", 0.5, "cmp")
    drift = max(abs(mb["clearance"] - mp["clearance"]) for _, mp, mb in case["rows"])
    assert drift > 1e-4, ("pose blending no longer drifts from parameter "
                          f"blending (max {drift * 1000:.4f} mm) -- the "
                          "comparison has stopped being meaningful")


def test_blend_weights_reproduce_the_endpoints():
    """weight 1 on a source must give that source back exactly, or the
    interpolation is not an interpolation."""
    g = SM.build()
    A, B = g["forward_slow"], g["left_turn"]
    for w, ref in (((1.0, 0.0), A), ((0.0, 1.0), B)):
        b = GB.BlendedSample([A, B], list(w), vel=ref.vel, name="edge")
        ts = np.linspace(0, ref.T, 61)
        for side in "LR":
            d = max(abs(b.c_foot_z[side](t) - ref.c_foot_z[side](t)) for t in ts)
            assert d < 1e-12, f"endpoint weight did not reproduce the source ({d:.2e})"
        assert abs(b.T - ref.T) < 1e-12


def test_blend_is_convex_in_the_event_times():
    """contact events have to move with the weights; if they did not, the two
    sources would be registered against each other incorrectly."""
    g = SM.build()
    A, B = g["forward_fast"], g["forward_slow"]
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = GB.BlendedSample([A, B], [1 - w, w], vel=A.vel, name="x")
        assert abs(b.T - ((1 - w) * A.T + w * B.T)) < 1e-12
        assert abs(b.t_ltd - ((1 - w) * A.t_ltd + w * B.t_ltd)) < 1e-12


def test_scalar_and_per_side_psi_both_blend():
    """samples.py carries psi as a scalar; a sample that came back from
    Blender carries it per side.  BlendedSample has to accept a mix of the
    two -- this was a real crash when the round-trip landed."""
    g = SM.build()
    A, B = g["forward_slow"], g["forward_fast"]
    B.psi_toe_lo = {"L": 0.10, "R": 0.12}          # pretend B came from USD
    b = GB.BlendedSample([A, B], [0.5, 0.5], vel=A.vel, name="mixed")
    v = b.psi_toe_lo
    assert isinstance(v, dict) and set(v) == {"L", "R"}
    assert np.isfinite(list(v.values())).all()
    assert abs(v["L"] - 0.5 * (A.psi_toe_lo + 0.10)) < 1e-12
