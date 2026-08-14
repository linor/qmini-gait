"""Export to USD, edit it the way a DCC would, read it back, and check the
plan is unchanged.

The point of the round-trip is that an edit made in Blender must survive into
the planner.  Two things are checked separately:

  fidelity   an *unedited* round-trip must reproduce the original plan
  fidelity   an *edited* round-trip must reproduce the edit, exactly

Both run on the ASCII backend, which is what is available without pxr.  When
pxr is present the same file is additionally read through both backends and
the two are required to agree -- see test_backend_agreement.
"""
import os
import re
import shutil

import numpy as np
import pytest

from stylized_gait import blender_io as BIO, library as GL, planner as GP, samples as SM

pytestmark = pytest.mark.usefixtures()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("gait_edit")
    g = SM.build()
    out = {}
    for name, s in g.items():
        p = str(d / f"{name}.usda")
        BIO.export_edit_usd(s, p)
        out[name] = p
    return out


def test_export_writes_every_sample(exported):
    assert len(exported) == 9
    for p in exported.values():
        assert os.path.getsize(p) > 10_000


def test_roundtrip_preserves_the_plan(exported):
    """unedited export -> import -> plan must match the original plan.

    The tolerance is loose (0.5 deg) on purpose: the curves are resampled at
    60 fps on the way out and rebuilt with a Catmull-Rom on the way back, so
    exact equality is not expected.  What would fail here is a frame or unit
    or axis error, all of which are degrees-large.
    """
    g = SM.build()
    lib = BIO.build_library(list(exported.values()), backend="ascii", write=False)
    for name, s in g.items():
        a = GP.solve_sample(s, fps=60, passes=3)
        b = GP.solve_sample(GL.LibrarySample(
            [x for x in lib["samples"] if x["name"] == name][0]), fps=60, passes=3)
        for side in "LR":
            err = np.rad2deg(np.abs(a["q"][side] - b["q"][side])).max()
            assert err < 0.5, f"{name} {side}: round-trip differs by {err:.3f} deg"


def test_an_edit_survives_the_roundtrip(exported, tmp_path):
    """rewrite one channel in the USD the way an animator would, and check the
    imported curve carries the edit rather than the original."""
    src = exported["step_in_place"]
    dst = str(tmp_path / "edited.usda")
    txt = open(src).read()

    # raise the left foot's swing apex by 2 cm at every keyframe
    block = re.search(r'def Xform "LeftFoot"\s*\{.*?xformOp:translate\.timeSamples = \{(.*?)\n\s*\}',
                      txt, re.S)
    assert block, "could not find the LeftFoot translate block"
    body = block.group(1)
    bumped = re.sub(r'\(([-0-9.eE]+), ([-0-9.eE]+), ([-0-9.eE]+)\)',
                    lambda m: f"({m.group(1)}, {m.group(2)}, {float(m.group(3)) + 2.0:.9g})",
                    body)
    open(dst, "w").write(txt.replace(body, bumped))

    lib = BIO.build_library([dst], backend="ascii", write=False)
    s = GL.LibrarySample(lib["samples"][0])
    ref = GL.LibrarySample(BIO.build_library([src], backend="ascii", write=False)["samples"][0])
    ts = np.linspace(0, s.T, 97)
    dz = np.array([s.c_foot_z["L"](t) - ref.c_foot_z["L"](t) for t in ts])
    assert abs(dz.mean() - 0.02) < 1e-6, f"edit landed as {dz.mean() * 100:.3f} cm, expected 2 cm"


def test_derived_scalars_follow_the_curve(exported):
    """psi_toe_lo / psi_heel_td are re-read from the curves rather than carried
    as scalars, precisely so an edited curve cannot disagree with a stale
    number.  Confirm they are consistent after import."""
    lib = BIO.build_library(list(exported.values()), backend="ascii", write=False)
    for d in lib["samples"]:
        s = GL.LibrarySample(d)
        assert abs(s.psi_toe_lo["L"] - s.c_psi_toe["L"](s.t_llo)) < 1e-9
        assert abs(s.psi_heel_td["R"] - s.c_psi_heel["R"](s.t_rtd)) < 1e-9


@pytest.mark.skipif(not BIO.HAVE_PXR, reason="pxr (usd-core) not installed")
def test_backend_agreement(exported):
    """pxr and the ASCII fallback must read the same file identically.

    NOTE: this has never been executed -- pxr could not be installed in the
    environment where these clips were produced.  It is the first thing to run
    on a machine that has usd-core.
    """
    for p in exported.values():
        worst = BIO.selftest(p)
        assert worst < 1e-6, f"{os.path.basename(p)}: backends differ by {worst:.2e}"


@pytest.mark.skipif(not BIO.HAVE_PXR, reason="pxr (usd-core) not installed")
def test_stage_validates(exported):
    """likewise unexecuted: checks up-axis, metersPerUnit, time range, that
    every rig prim's local transform composes, and that /Gait/Reference
    resolves as a real UsdSkelRoot with skinned meshes."""
    for p in exported.values():
        BIO.validate(p)
