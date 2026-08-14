"""
blend_compare.py -- parameter blending vs pose blending, measured and drawn.

Parameter blending (paper Sec. 5.2): interpolate the gait sample Pi -- time
intervals, attributes, function curves -- into a NEW valid sample, then run the
planner on it.  Constraints are re-enforced from scratch.

Pose blending (what gait_blend.html does): interpolate the planner's OUTPUT,
i.e. the joint angles, at phase-registered times.  Nothing re-checks the result.

Both use the identical phase map registration, identical weights and identical
source samples, so the only difference is what gets interpolated.
"""

import sys, os
import numpy as np


from stylized_gait.core import (
    PHI_LANDMARKS, MonotonicSpline, CHAIN, SAFE_LIMITS_DEG, SOLE_CORNERS,
    rot_zyx, yaw_pitch_of, rot_z, path_pose,
)
from stylized_gait import core as GC
from stylized_gait import planner as GP
from stylized_gait import samples as SM


# ---------------------------------------------------------------------------
# Parameter blending  (paper Sec. 5.2)
# ---------------------------------------------------------------------------

from stylized_gait.blend import BlendedCurve, BlendedSample  # noqa: F401


# ---------------------------------------------------------------------------
# Pose blending  (identical registration, applied to the planner's output)
# ---------------------------------------------------------------------------

def solved(sample, fps=120):
    return GP.solve_sample(sample, fps=fps, passes=3)


def sample_at_phi(res, sample, phi):
    """joint angles + path-relative pelvis rotation of one solved clip at the
    time that corresponds to gait phase phi"""
    t = float(sample.phase(phi)) % sample.T
    x = t * res["fps"]
    i0 = int(np.clip(np.floor(x), 0, res["n"] - 1))
    u = x - i0
    i1 = (i0 + 1) % res["n"]
    q = {}
    for s in "LR":
        q[s] = res["q"][s][i0] * (1 - u) + res["q"][s][i1] * u
    R0, R1 = res["pelvis_R"][i0], res["pelvis_R"][i1]
    yaw0 = sample.vel[2] * (i0 / res["fps"])
    yaw1 = sample.vel[2] * (i1 / res["fps"])
    L0 = rot_z(-yaw0) @ R0
    L1 = rot_z(-yaw1) @ R1
    e0, e1 = _euler(L0), _euler(L1)
    return q, e0 * (1 - u) + e1 * u


def _euler(R):
    yaw, pitch = yaw_pitch_of(R)
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll])


def pose_blend_at_phi(srcs, ress, ws, phi):
    q = {s: np.zeros(5) for s in "LR"}
    e = np.zeros(3)
    for src, res, w in zip(srcs, ress, ws):
        qi, ei = sample_at_phi(res, src, phi)
        for s in "LR":
            q[s] += w * qi[s]
        e += w * ei
    return q, e


# ---------------------------------------------------------------------------
# Shared measurement: everything relative to the support foot's ground plane
# ---------------------------------------------------------------------------

def contact_of(phi):
    if phi < PHI_LANDMARKS[1]: return "LS"
    if phi < PHI_LANDMARKS[2]: return "DS1"
    if phi < PHI_LANDMARKS[3]: return "RS"
    return "DS2"


def measure(q, euler, phi):
    """FK with the root at the origin, then express everything relative to the
    plane the support foot stands on.  Method-agnostic, so the two approaches
    are directly comparable."""
    R0 = rot_zyx(0.0, euler[1], euler[2])
    st = contact_of(phi)
    corners, sole = {}, {}
    for s in "LR":
        leg = GP.LEGS[s]
        p, R = leg.fk(q[s], np.zeros(3), R0)
        corners[s] = np.array([p + R @ (-GC.ANKLE_IN_FOOT + c) for c in SOLE_CORNERS[s]])
        sole[s] = p + R @ (-GC.ANKLE_IN_FOOT)
    if st == "LS":   sup, sw = ["R"], "L"
    elif st == "RS": sup, sw = ["L"], "R"
    else:            sup, sw = ["L", "R"], None
    ground = min(corners[s][:, 2].min() for s in sup)
    out = {"state": st, "ground": ground, "euler": np.asarray(euler, float),
           "sole": {s: sole[s] - np.array([0, 0, ground]) for s in "LR"},
           "corners": {s: corners[s] - np.array([0, 0, ground]) for s in "LR"},
           "R0": R0, "q": q}
    out["clearance"] = (corners[sw][:, 2].min() - ground) if sw else 0.0
    out["swing"] = sw
    # support-polygon span, i.e. how far apart the two feet are
    out["span"] = float(np.linalg.norm(sole["L"][:2] - sole["R"][:2]))
    return out


def joint_use(q):
    u = 0.0
    for s in "LR":
        for j, jn in enumerate(CHAIN):
            lo, hi = SAFE_LIMITS_DEG[jn]
            a = np.rad2deg(q[s][j])
            u = max(u, a / hi if a > 0 else a / lo if a < 0 else 0.0)
    return u


def run_case(nameA, nameB, alpha, label):
    g = SM.build()
    A, B = g[nameA], g[nameB]
    ws = [1 - alpha, alpha]
    vel = (1 - alpha) * A.vel + alpha * B.vel

    resA, resB = solved(A), solved(B)
    blend = BlendedSample([A, B], ws, vel=vel, name=label)
    resP = solved(blend)

    phis = np.linspace(0, 1, 241)
    rows = []
    for phi in phis:
        # parameter blend: read the solved blended sample at this phase
        t = float(blend.phase(phi)) % blend.T
        x = t * resP["fps"]
        i0 = int(np.clip(np.floor(x), 0, resP["n"] - 1)); u = x - i0
        i1 = (i0 + 1) % resP["n"]
        qp = {s: resP["q"][s][i0] * (1 - u) + resP["q"][s][i1] * u for s in "LR"}
        yaw = blend.vel[2] * (i0 / resP["fps"])
        ep = _euler(rot_z(-yaw) @ resP["pelvis_R"][i0])
        mp = measure(qp, ep, phi)
        # pose blend
        qb, eb = pose_blend_at_phi([A, B], [resA, resB], ws, phi)
        mb = measure(qb, eb, phi)
        rows.append((phi, mp, mb))
    return dict(A=A, B=B, alpha=alpha, vel=vel, blend=blend, label=label,
                resA=resA, resB=resB, resP=resP, rows=rows, ws=ws)
