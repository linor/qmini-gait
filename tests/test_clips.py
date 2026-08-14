"""The invariants a generated clip must satisfy.

These are the checks that caught real bugs during development: the pelvis
reference-point error (clips floated 10 cm forward), the heel/toe sign
inversion (soles through the floor), the IK-branch flip, the cycle-wrap
off-by-one, and the ankle-vs-sole IK target error.  Each assertion below
corresponds to one of those.

Everything runs against the .usda on disk, re-parsed independently of the
generator, so a bug shared between writer and reader cannot hide.
"""
import glob
import os

import numpy as np
import pytest

from stylized_gait import paths, verify as V

CLIPS = sorted(glob.glob(os.path.join(paths.CLIPS, "*.usda")))
IDS = [os.path.basename(p)[:-5] for p in CLIPS]


def test_nine_clips_exist():
    assert len(CLIPS) == 9, f"expected 9 clips, found {len(CLIPS)}"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_no_issues_reported(path):
    """the verifier's own summary must be clean"""
    _, _, issues = V.check(path)
    assert not issues, "; ".join(issues)


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_ground_contact(path):
    """the sole never goes through the floor.  Tolerance is 0.1 mm, the
    planner's contact-lift convergence threshold."""
    _, info, _ = V.check(path)
    assert info["min_sole_z_cm"] >= -0.01, \
        f"ground penetration {info['min_sole_z_cm']:.4f} cm"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_stance_foot_does_not_slide(path):
    """1 mm/frame at 60 fps is 6 cm/s of scuffing, which reads as a skate and
    is what the ankle-vs-sole IK bug produced (1.4 mm/frame)."""
    _, info, _ = V.check(path)
    assert info["max_stance_slip_mm"] < 0.5, \
        f"stance slip {info['max_stance_slip_mm']:.3f} mm/frame"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_joint_headroom(path):
    """stay inside the conservative safe envelope, with margin"""
    _, info, _ = V.check(path)
    assert info["max_joint_use_pct"] <= 95.0, \
        f"joint use {info['max_joint_use_pct']:.0f}% of the safe range"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_cycle_closes(path):
    """q(T) must equal q(0).  A clip that does not close tears when looped and
    corrupts any blend that crosses the loop point."""
    _, info, _ = V.check(path)
    assert info["loop_quat_delta"] < 1e-6, \
        f"loop discontinuity {info['loop_quat_delta']:.2e}"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_root_travels_the_commanded_distance(path):
    """root displacement over one cycle must equal velocity x period, or the
    blend weights do not mean the velocities they claim to."""
    _, info, _ = V.check(path)
    assert info["travel_err_mm"] < 1.0, \
        f"root travel error {info['travel_err_mm']:.3f} mm"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_phase_tagging(path):
    """blending registers two clips by phase, so the tag has to be monotone,
    span exactly [0,1], and line up with the four contact events."""
    d, info, _ = V.check(path)
    phi = np.asarray(d["phase"], float)
    assert np.all(np.diff(phi) > -1e-9), "phase primvar is not monotone"
    assert abs(phi[0]) < 1e-6, f"phase starts at {phi[0]}"
    assert abs(phi[-1] - 1.0) < 1e-6, f"phase ends at {phi[-1]}"
    assert info["event_phase_ok"], "phase landmarks do not match the event times"


@pytest.mark.parametrize("path", CLIPS, ids=IDS)
def test_contact_sequence(path):
    """left support -> double -> right support -> double, in that order"""
    _, info, _ = V.check(path)
    assert info["state_sequence"].startswith("LS->DS1->RS->DS2"), \
        info["state_sequence"]
