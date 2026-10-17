"""Lateral pelvis sway (samples.PELVIS_SWAY -> core.GaitSample.c_pelvis_y).

Added 2026-10-17 for the RL gait library: the authored gaits never shift the
weight sideways (only a +-2 deg pelvis lean), and the trained policy did the
shift by rocking the whole robot on its laterally rounded soles. The sway
translates the pelvis toward the support foot instead. These tests pin that
the default reproduces the authored gaits, and that a sway moves the pelvis
and nothing else.
"""
import numpy as np
import pytest

from stylized_gait import samples as SM
from stylized_gait import planner as GP

SWAY = 0.03


def test_default_has_no_sway(gaits):
    # build() at the module defaults must reproduce the authored gaits exactly
    for name, s in gaits.items():
        assert s.pelvis_sway == 0.0, name
        ts = np.linspace(0.0, s.T, 37)
        assert max(abs(float(s.c_pelvis_y(t))) for t in ts) == 0.0, name


@pytest.fixture(scope="module")
def swayed(monkeypatch_module):
    monkeypatch_module.setattr(SM, "PELVIS_SWAY", SWAY)
    s = SM.build()["step_in_place"]
    return GP.solve_sample(s, fps=60, passes=3)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_sway_moves_the_pelvis_toward_the_support_foot(swayed):
    s = swayed["sample"]
    t = swayed["times"]
    y = swayed["pelvis_p"][:, 1]          # character frame, y = left; zero velocity -> path y = 0
    mid_ls = 0.5 * (s.t_llo + s.t_ltd)    # left swing: the right foot supports -> -y
    mid_rs = 0.5 * (s.t_rlo + s.t_rtd)
    assert y[np.argmin(abs(t - mid_ls))] == pytest.approx(-SWAY, abs=1e-3)
    assert y[np.argmin(abs(t - mid_rs))] == pytest.approx(+SWAY, abs=1e-3)
    # a symmetric gait sways symmetrically: y(t + T/2) = -y(t)
    half = swayed["n"] // 2
    assert np.abs(y + np.roll(y, -half)).max() < 2e-3


def test_sway_leaves_the_feet_alone(swayed, solved):
    base = solved["step_in_place"]
    for side in base["foot_p"]:
        d = np.abs(swayed["foot_p"][side] - base["foot_p"][side]).max()
        # the feet are planned in the path frame and do not know about the
        # pelvis; only the contact-height correction (lowest sole corner on the
        # ground, which changes with the leg tilt) may move them, by < 1 mm
        assert d < 1e-3, (side, d)
        assert np.abs(swayed["err"][side]).max() < 1e-3, side
    # and the legs did have to move to follow the pelvis
    dq = max(np.degrees(np.abs(swayed["q"][sd] - base["q"][sd])).max() for sd in base["q"])
    assert 1.0 < dq < 20.0, dq
