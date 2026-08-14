"""The Isaac Lab hand-off.

The .npz carries one clip in the robot's own joint order with PhysX sign
conventions already applied.  Two things here are worth guarding.

Sign convention: a UsdPhysics revolute joint measures body1 relative to body0,
and six of QMini's joints have body0/body1 swapped relative to parent->child,
so PhysX reads them with inverted sign.  A joint-limit check cannot catch a
uniform sign flip -- negating every joint still fits inside a symmetric
envelope, which was verified empirically.  What does catch it is half-cycle
antisymmetry: for a symmetric gait, q_R(t + T/2) = -q_L(t).  A wrong
right-leg sign shows up as ~49 deg of residual against ~0 when correct.

Sampler consistency: MotionPlayer.sample is a cyclic Catmull-Rom so that
sample_vel can be its exact analytic derivative.  When sample was linear the
two disagreed by 1.3 rad/s, which is a large torque error in a position
controller.
"""
import os

import numpy as np
import pytest

from stylized_gait import paths

NPZ = os.path.join(paths.DATA, "qmini_step_in_place.npz")
pytestmark = pytest.mark.skipif(not os.path.isfile(NPZ),
                                reason="run scripts/export_isaac_motion.py first")


@pytest.fixture(scope="module")
def d():
    return dict(np.load(NPZ, allow_pickle=True))


def test_shapes_and_joint_order(d):
    q = d["joint_pos"]
    assert q.ndim == 2 and q.shape[1] == 10, f"expected 10 DoF, got {q.shape}"
    names = [str(x) for x in d["joint_names"]]
    assert len(names) == 10 and len(set(names)) == 10
    assert sum("left" in n.lower() for n in names) == 5
    assert set(np.unique(d["joint_sign"])) <= {-1.0, 1.0}
    for k in ("root_pos", "root_quat", "phase", "contact", "times"):
        assert len(d[k]) == len(q), f"{k} has {len(d[k])} rows, joint_pos has {len(q)}"


def test_clip_is_periodic(d):
    """the exported clip includes both endpoints, so the seam must be exact --
    the player loops it and any gap is a step discontinuity in the target."""
    q = d["joint_pos"]
    seam = np.abs(q[0] - q[-1]).max()
    assert seam < 1e-6, f"seam discontinuity {np.rad2deg(seam):.4f} deg"


def test_half_cycle_antisymmetry(d):
    """the check a joint-limit test cannot do: catches a flipped sign on one
    leg.  A pure L/R swap is invisible to this, correctly -- for a symmetric
    gait that is just a half-cycle time shift."""
    q = d["joint_pos"][:-1]                      # drop the duplicated endpoint
    n = len(q)
    assert n % 2 == 0, "odd frame count; cannot shift by exactly half a cycle"
    L, R = q[:, :5], q[:, 5:]
    res = np.rad2deg(np.abs(np.roll(R, -n // 2, axis=0) + L)).max()
    assert res < 1.0, f"half-cycle antisymmetry residual {res:.4f} deg"
    # and the exporter's own recorded value must agree with what we just did
    assert float(d["antisymmetry_deg"]) < 1.0


def test_within_joint_limits(d):
    lo, hi = d["joint_pos_lower"], d["joint_pos_upper"]
    q = d["joint_pos"]
    assert (q >= lo - 1e-6).all(), "clip violates a lower joint limit"
    assert (q <= hi + 1e-6).all(), "clip violates an upper joint limit"


def test_contact_track_is_binary_and_never_airborne(d):
    """step-in-place always has at least one foot down; a frame with neither
    means the planner produced a flight phase, which this robot cannot do."""
    c = d["contact"]
    assert set(np.unique(c)) <= {0.0, 1.0}
    assert (c.sum(axis=1) >= 1).all(), "found an airborne frame"


def test_sampler_matches_its_own_derivative():
    """finite-difference the sampler and compare against the analytic
    velocity.  These disagreed by 1.3 rad/s when sample() was linear."""
    torch = pytest.importorskip("torch")
    import sys
    sys.path.insert(0, paths.ISAAC)
    from motion_player import MotionPlayer

    mp = MotionPlayer.from_npz(NPZ, device="cpu")
    T = float(mp.duration)
    h = 1e-5
    ts = torch.linspace(0.0, T, 97)
    worst = 0.0
    for t in ts:
        a = mp.sample(t - h)
        b = mp.sample(t + h)
        fd = (b - a) / (2 * h)
        an = mp.sample_vel(t)
        worst = max(worst, float((fd - an).abs().max()))
    assert worst < 1e-2, f"sample_vel is not the derivative of sample ({worst:.3e})"
