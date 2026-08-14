"""
gait_library.py -- load the compiled intermediate produced by blender_io.

``gait_library.json`` is the hand-off point: once it exists, blending, clip
generation and the Isaac export all read it and never touch USD again.  It holds
exactly the Pi the planner needs -- the nine irrecoverable scalars per sample,
plus every Tab. 2 curve as a sampled series.

A LibrarySample quacks like a GaitSample, so gait_planner runs on it unchanged
and gait_overrides.BlendedSample can still blend it.

The three scalars that are *derived* rather than stored:

    psi_toe_lo[side]   the psi_toe curve at the liftoff end of that swing
    psi_heel_td[side]  the psi_heel curve at the touchdown end
    pelvis_z_nom       the mean of the (absolute) pelvis z curve; the curve is
                       re-expressed as an offset from it, which is what the
                       planner expects
"""

import json

import numpy as np

from . import core as GC


class Sampled:
    """a curve as (t, v) samples, evaluated with a cyclic-safe Catmull-Rom.

    The generator's curves are analytic splines; once they have been through a
    DCC they come back as samples, so this is what an edited curve becomes.
    Interpolation matches motion_player.py on the Isaac side.
    """

    def __init__(self, t, v, periodic=False, period=None):
        self.t = np.asarray(t, float)
        self.v = np.asarray(v, float)
        self.periodic = periodic
        self.period = period if period is not None else float(self.t[-1])
        n = len(self.t)
        if periodic and n > 2:
            fwd = np.roll(self.v[:-1], -1)
            bwd = np.roll(self.v[:-1], 1)
            dt = np.diff(self.t).mean()
            m = (fwd - bwd) / (2 * dt)
            self.m = np.append(m, m[0])
        else:
            self.m = np.gradient(self.v, self.t) if n > 2 else np.zeros(n)

    def __call__(self, x):
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, float))
        t = self.t
        if self.periodic:
            x = t[0] + np.mod(x - t[0], self.period)
        xc = np.clip(x, t[0], t[-1])
        i = np.clip(np.searchsorted(t, xc, side="right") - 1, 0, len(t) - 2)
        h = t[i + 1] - t[i]
        s = (xc - t[i]) / h
        s2, s3 = s * s, s * s * s
        out = ((2 * s3 - 3 * s2 + 1) * self.v[i]
               + (s3 - 2 * s2 + s) * h * self.m[i]
               + (-2 * s3 + 3 * s2) * self.v[i + 1]
               + (s3 - s2) * h * self.m[i + 1])
        return float(out[0]) if scalar else out


class LibrarySample:
    """GaitSample-compatible view of one entry in the library"""

    def __init__(self, d):
        self.name = d["name"]
        self.notes = "from gait_library"
        self.vel = np.asarray(d["vel"], float)
        self.T = float(d["T"])
        self.t_llo, self.t_ltd, self.t_rlo, self.t_rtd = [float(x) for x in d["events"]]
        self.phase = GC.PhaseMap(self.t_llo, self.t_ltd, self.t_rlo, self.t_rtd, self.T)
        self.stance_width = float(d["stance_width"])
        self.stance_x_bias = float(d["stance_x_bias"])
        self.stance_splay = float(d["stance_splay"])
        self.dt_psi_lo = float(d["dt_psi_lo"])
        self.dt_psi_td = float(d["dt_psi_td"])

        C = d["curves"]
        mk = lambda k, per=False: Sampled(C[k][0], C[k][1], per, self.T)
        self.c_foot_x = {s: mk(f"foot_x.{s}") for s in "LR"}
        self.c_foot_y = {s: mk(f"foot_y.{s}") for s in "LR"}
        self.c_foot_z = {s: mk(f"foot_z.{s}") for s in "LR"}
        self.c_foot_yaw = {s: mk(f"foot_yaw.{s}") for s in "LR"}
        self.c_foot_gamma = {s: mk(f"foot_gamma.{s}") for s in "LR"}
        self.c_psi_toe = {s: mk(f"psi_toe.{s}") for s in "LR"}
        self.c_psi_heel = {s: mk(f"psi_heel.{s}") for s in "LR"}
        self.c_pelvis_roll = mk("pelvis_roll", True)
        self.c_pelvis_pitch = mk("pelvis_pitch", True)
        self.c_pelvis_yaw = mk("pelvis_yaw", True)

        # pelvis z arrives absolute; split it into nominal height + offset
        pz = Sampled(C["pelvis_z"][0], C["pelvis_z"][1], True, self.T)
        self.pelvis_z_nom = float(np.mean(pz.v[:-1]))
        self.c_pelvis_z = Sampled(pz.t, pz.v - self.pelvis_z_nom, True, self.T)

        # heel/toe support ramps take their target from the curve itself, per
        # side, so an asymmetric edit cannot disagree with a stale scalar
        self.psi_toe_lo = {"L": float(self.c_psi_toe["L"](self.t_llo)),
                           "R": float(self.c_psi_toe["R"](self.t_rlo))}
        self.psi_heel_td = {"L": float(self.c_psi_heel["L"](self.t_ltd)),
                            "R": float(self.c_psi_heel["R"](self.t_rtd))}

    contact_state = GC.GaitSample.contact_state
    cycle_displacement = GC.GaitSample.cycle_displacement


def load(path):
    """-> {name: LibrarySample}"""
    with open(path) as f:
        d = json.load(f)
    if d.get("version") != 1:
        raise ValueError(f"{path}: unsupported library version {d.get('version')}")
    return {s["name"]: LibrarySample(s) for s in d["samples"]}
