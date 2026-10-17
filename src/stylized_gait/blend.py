"""
gait_blend.py -- Sec. 5.2 parameter blending.

A convex combination of gait samples that still quacks like a GaitSample, so
gait_planner runs on it unchanged.  Works on authored samples (samples.py) and
on library samples that came back through Blender (gait_library.py) alike.
"""

import numpy as np

from . import core as GC
from .core import HermiteSpline


def _ZERO(t):
    """a zero curve, for sources without one (e.g. c_pelvis_y from a library
    written before the sway existed)"""
    return 0.0


class BlendedCurve:
    """A function curve of the blended sample.  Evaluating it at a time t of the
    blended cycle means: map t -> phi with the blended phase map, map phi back
    into each source cycle's own time, and combine the source curve values.
    That is Eq. 7 applied through the registration of Sec. 5.1."""

    def __init__(self, srcs, ws, pick, om_blend):
        self.srcs, self.ws, self.pick, self.om = srcs, ws, pick, om_blend

    def __call__(self, t):
        phi = self.om.phase_of(float(t))
        out = 0.0
        for s, w in zip(self.srcs, self.ws):
            out += w * self.pick(s)(s.phase(phi))
        return out


class BlendedSample:
    """A convex combination of gait samples that still quacks like a GaitSample,
    so the unmodified planner can run on it."""

    def __init__(self, srcs, ws, vel=None, name="blend"):
        ws = np.asarray(ws, float)
        ws = ws / ws.sum()
        self.srcs, self.ws = srcs, ws
        self.name = name
        self.notes = "parameter blend"

        def b(f):
            vals = [f(s) for s in srcs]
            if any(isinstance(v, dict) for v in vals):
                # psi targets are per side on library samples
                return {k: float(sum(w * (v[k] if isinstance(v, dict) else v)
                                     for v, w in zip(vals, ws))) for k in "LR"}
            return float(sum(w * v for v, w in zip(vals, ws)))

        # --- Tab. 1 time intervals: alpha-blend min and max independently.
        #     A convex combination cannot reorder them, so t_LLO < t_LTD <
        #     t_RLO < t_RTD survives automatically.
        self.t_llo = 0.0
        self.t_ltd = b(lambda s: s.t_ltd)
        self.t_rlo = b(lambda s: s.t_rlo)
        self.t_rtd = b(lambda s: s.t_rtd)
        self.T = b(lambda s: s.T)
        self.phase = GC.PhaseMap(self.t_llo, self.t_ltd, self.t_rlo, self.t_rtd, self.T)

        self.vel = np.asarray(vel if vel is not None
                              else sum(w * s.vel for s, w in zip(srcs, ws)), float)

        # --- attributes: plain alpha blend (Eq. 7)
        self.stance_width = b(lambda s: s.stance_width)
        self.stance_x_bias = b(lambda s: s.stance_x_bias)
        self.stance_splay = b(lambda s: s.stance_splay)
        self.dt_psi_lo = b(lambda s: s.dt_psi_lo)
        self.dt_psi_td = b(lambda s: s.dt_psi_td)
        self.pelvis_z_nom = b(lambda s: s.pelvis_z_nom)
        self.psi_toe_lo = b(lambda s: s.psi_toe_lo)
        self.psi_heel_td = b(lambda s: s.psi_heel_td)

        # --- function curves: registered blend
        C = lambda pick: BlendedCurve(srcs, ws, pick, self.phase)
        self.c_foot_x = {k: C(lambda s, k=k: s.c_foot_x[k]) for k in "LR"}
        self.c_foot_y = {k: C(lambda s, k=k: s.c_foot_y[k]) for k in "LR"}
        self.c_foot_z = {k: C(lambda s, k=k: s.c_foot_z[k]) for k in "LR"}
        self.c_foot_yaw = {k: C(lambda s, k=k: s.c_foot_yaw[k]) for k in "LR"}
        self.c_foot_gamma = {k: C(lambda s, k=k: s.c_foot_gamma[k]) for k in "LR"}
        self.c_psi_toe = {k: C(lambda s, k=k: s.c_psi_toe[k]) for k in "LR"}
        self.c_psi_heel = {k: C(lambda s, k=k: s.c_psi_heel[k]) for k in "LR"}
        self.c_pelvis_z = C(lambda s: s.c_pelvis_z)
        self.c_pelvis_roll = C(lambda s: s.c_pelvis_roll)
        self.c_pelvis_y = C(lambda s: getattr(s, "c_pelvis_y", _ZERO))
        self.c_pelvis_pitch = C(lambda s: s.c_pelvis_pitch)
        self.c_pelvis_yaw = C(lambda s: s.c_pelvis_yaw)

    contact_state = GC.GaitSample.contact_state
    cycle_displacement = GC.GaitSample.cycle_displacement


