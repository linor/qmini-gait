"""
gait_planner.py -- nominal + stylized motion planning and leg IK.

Implements paper Sec. 4.2 (footstep plan, nominal swing/support foot and pelvis
trajectories) and Sec. 4.3 (stylized component: swing curves, heel-toe rig,
pelvis curves), then solves the 5-DoF leg IK to joint angles.
"""

import numpy as np
from .core import (
    anchors, axes, CHAIN, LIMITS_DEG, SAFE_LIMITS_DEG, rot_axis, rot_y, rot_z, rot_zyx,
    yaw_pitch_of, ANKLE_IN_FOOT, HEEL_IN_FOOT, TOE_IN_FOOT, SOLE_CORNERS,
    NOMINAL_HALF_W, PELVIS_X_OFFSET, STANCE_X_NEUTRAL, path_pose, compose, f_soft,
)

SIDES = ("L", "R")


# ----------------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------------

class Leg:
    def __init__(self, side):
        self.side = side
        self.a = anchors(side)
        self.ax = axes(side)
        # local (parent-relative) rest translations of the skeleton joints
        self.local = {}
        prev = np.zeros(3)
        for j in CHAIN:
            self.local[j] = self.a[j] - prev
            prev = self.a[j]
        self.lim = np.deg2rad(np.array([SAFE_LIMITS_DEG[j] for j in CHAIN]))
        self.lim_raw = np.deg2rad(np.array([LIMITS_DEG[side][j] for j in CHAIN]))

    def fk(self, q, p_pelvis=np.zeros(3), R_pelvis=np.eye(3)):
        """-> (ankle position, foot rotation) in world (character) frame"""
        R = R_pelvis
        p = p_pelvis + R @ self.a["hip_yaw"]
        R = R @ rot_axis(self.ax["hip_yaw"], q[0])
        for i, j in enumerate(CHAIN[1:], start=1):
            p = p + R @ self.local[j]
            R = R @ rot_axis(self.ax[j], q[i])
        return p, R

    def fk_chain(self, q, p_pelvis=np.zeros(3), R_pelvis=np.eye(3)):
        """all joint origins, for diagnostics"""
        out = {}
        R = R_pelvis
        p = p_pelvis + R @ self.a["hip_yaw"]
        out["hip_yaw"] = p.copy()
        R = R @ rot_axis(self.ax["hip_yaw"], q[0])
        for i, j in enumerate(CHAIN[1:], start=1):
            p = p + R @ self.local[j]
            out[j] = p.copy()
            R = R @ rot_axis(self.ax[j], q[i])
        return out, R

    def fk_sole(self, q, p_pelvis, R_pelvis):
        """position of the sole reference point (the foothold origin) and the
        foot rotation.  Driving the IK from the sole rather than the ankle is
        what keeps the contact point pinned: this robot has no ankle roll, so
        the sole tilts under hip roll, and an ankle-based target would drag the
        contact point sideways by ankle_height * sin(roll)."""
        p, R = self.fk(q, p_pelvis, R_pelvis)
        return p + R @ (-ANKLE_IN_FOOT), R

    # ---- IK ---------------------------------------------------------------
    def _residual(self, q, p_tgt, yaw_tgt, pitch_tgt, p_pelvis, R_pelvis):
        p, R = self.fk_sole(q, p_pelvis, R_pelvis)
        yaw, pitch = yaw_pitch_of(R_pelvis.T @ R)   # foot pose in pelvis frame
        e = np.empty(5)
        e[:3] = p_tgt - p
        e[3] = np.arctan2(np.sin(yaw_tgt - yaw), np.cos(yaw_tgt - yaw))
        e[4] = np.arctan2(np.sin(pitch_tgt - pitch), np.cos(pitch_tgt - pitch))
        return e

    def ik(self, p_tgt, yaw_tgt, pitch_tgt, q0, p_pelvis, R_pelvis,
           iters=200, lam=1e-4, tol=1e-9):
        """Damped least-squares IK with hard joint-limit projection.

        Targets: ankle position (world) plus foot yaw / pitch relative to the
        pelvis.  Foot roll is not an independent target -- this robot has no
        ankle roll, so foot roll follows from the hip roll.
        Limits are enforced every iteration, which also keeps the solver on the
        correct (reverse-knee) branch rather than flipping the knee."""
        q = np.clip(np.array(q0, float), self.lim[:, 0], self.lim[:, 1])
        W = np.array([1.0, 1.0, 1.0, 0.12, 0.12])   # position first, then angles
        h = 1e-6
        best_q, best_c = q.copy(), np.inf
        for _ in range(iters):
            e = self._residual(q, p_tgt, yaw_tgt, pitch_tgt, p_pelvis, R_pelvis)
            c = np.linalg.norm(e * W)
            if c < best_c:
                best_c, best_q = c, q.copy()
            if c < tol:
                break
            J = np.empty((5, 5))
            for k in range(5):
                dq = np.zeros(5); dq[k] = h
                J[:, k] = -(self._residual(q + dq, p_tgt, yaw_tgt, pitch_tgt,
                                           p_pelvis, R_pelvis) - e) / h
            Jw = J * W[:, None]
            ew = e * W
            dq = np.linalg.solve(Jw.T @ Jw + lam * np.eye(5), Jw.T @ ew)
            step = np.linalg.norm(dq)
            if step > 0.20:
                dq *= 0.20 / step
            qn = np.clip(q + dq, self.lim[:, 0], self.lim[:, 1])
            if np.max(np.abs(qn - q)) < 1e-10:      # converged or hard against a limit
                q = qn
                break
            q = qn
        e = self._residual(best_q, p_tgt, yaw_tgt, pitch_tgt, p_pelvis, R_pelvis)
        return best_q, e


LEGS = {s: Leg(s) for s in SIDES}


# ----------------------------------------------------------------------------
# Heel / toe rig  (paper Fig. 5)
# ----------------------------------------------------------------------------

def heel_toe_transform(psi_heel, psi_toe):
    """Sequential pitch about the heel, then about the resulting toe, expressed
    in the foothold frame F (paper Fig. 5).  With -pi/2 <= psi_heel <= 0 the
    heel rotation lifts the toe (heel strike); with 0 <= psi_toe <= pi/2 the toe
    rotation lifts the heel (toe off).  In F, rot_y(a) maps x -> (cos a, 0, -sin a),
    so a < 0 raises the toe and a > 0 raises the heel -- the paper's sign
    convention maps directly onto rot_y.  This guarantees the lowest sole point
    stays at or above z = 0."""
    R1 = rot_y(psi_heel)
    t1 = HEEL_IN_FOOT - R1 @ HEEL_IN_FOOT
    toe1 = R1 @ TOE_IN_FOOT + t1
    R2 = rot_y(psi_toe)
    t2 = toe1 - R2 @ toe1
    R = R2 @ R1
    t = R2 @ t1 + t2
    return R, t


# ----------------------------------------------------------------------------
# Full reference trajectory for one gait sample
# ----------------------------------------------------------------------------

def foothold_times(sample, k, side):
    """midpoint of the contact interval for step k (paper Sec. 4.2)"""
    T = sample.T
    if side == "L":
        a = sample.t_ltd + k * T
        b = sample.t_llo + (k + 1) * T
    else:
        a = sample.t_rtd + k * T
        b = sample.t_rlo + (k + 1) * T
    return 0.5 * (a + b)


def stance_pose(sample, side):
    """Constant planar stance pose in the path frame (paper Sec. 4.3, Attributes)."""
    s = +1.0 if side == "L" else -1.0
    return np.array([STANCE_X_NEUTRAL + sample.stance_x_bias,
                     s * 0.5 * sample.stance_width,
                     s * sample.stance_splay])


def foothold(sample, k, side):
    t = foothold_times(sample, k, side)
    return compose(path_pose(sample.vel, t), stance_pose(sample, side))


def swing_alpha(t, a, b):
    """cubic 0->1 with zero endpoint velocity (paper Eqs. 5-6)"""
    return f_soft((t - a) / (b - a))


def foot_pose_at(sample, side, t):
    """world sole-frame pose of one foot: (position, rotation, in_contact)."""
    T = sample.T
    tm = t % T
    if side == "L":
        a, b = sample.t_llo, sample.t_ltd
    else:
        a, b = sample.t_rlo, sample.t_rtd
    swinging = a <= tm < b

    fh_prev = foothold(sample, -1, side)
    fh_next = foothold(sample, 0, side)

    if swinging:
        al = swing_alpha(tm, a, b)
        pl = (1 - al) * fh_prev[:2] + al * fh_next[:2]
        th = fh_prev[2] + al * np.arctan2(np.sin(fh_next[2] - fh_prev[2]),
                                          np.cos(fh_next[2] - fh_prev[2]))
        base = np.array([pl[0], pl[1], 0.0])
        base_yaw = th
        # --- stylized swing offsets, expressed in the nominal swing frame ---
        dx = sample.c_foot_x[side](tm)
        dy = sample.c_foot_y[side](tm)
        dz = sample.c_foot_z[side](tm)
        dyaw = sample.c_foot_yaw[side](tm)
        dgam = sample.c_foot_gamma[side](tm)
        psi_h = sample.c_psi_heel[side](tm)
        psi_t = sample.c_psi_toe[side](tm)
    else:
        # support phase: the foot is latched to the foothold fixed at the last
        # touchdown (paper Sec. 4.2, "Nominal Support/Swing Foot Trajectories")
        t_td_ = sample.t_ltd if side == "L" else sample.t_rtd
        fh = fh_next if tm >= t_td_ else fh_prev
        base = np.array([fh[0], fh[1], 0.0])
        base_yaw = fh[2]
        dx = dy = dz = dyaw = dgam = 0.0
        # automated heel/toe relaxation during support (paper Sec. 4.3)
        t_td = (sample.t_ltd if side == "L" else sample.t_rtd)
        t_lo = (sample.t_llo + T if side == "L" else sample.t_rlo + T)
        te = tm if tm >= t_td else tm + T
        psi_h = _ramp_down(te, t_td, sample.dt_psi_td, _side(sample.psi_heel_td, side))
        psi_t = _ramp_up(te, t_lo, sample.dt_psi_lo, _side(sample.psi_toe_lo, side))

    R_ht, t_ht = heel_toe_transform(psi_h, psi_t)
    R_local = rot_zyx(dyaw, 0.0, dgam) @ R_ht
    p_local = np.array([dx, dy, dz]) + t_ht

    Rb = rot_z(base_yaw)
    R = Rb @ R_local
    p = base + Rb @ p_local
    return p, R, (not swinging)


def _side(v, side):
    """psi targets may be a single scalar (authored samples) or per side (a
    library sample, where each side's value comes from its own curve)"""
    return float(v[side]) if isinstance(v, dict) else float(v)


def _ramp_down(t, t_event, dt, value):
    """heel pitch relaxes linearly to 0 over dt after touchdown"""
    if dt <= 0 or value == 0.0:
        return 0.0
    u = (t - t_event) / dt
    if u <= 0: return value
    if u >= 1: return 0.0
    return value * (1.0 - u)


def _ramp_up(t, t_event, dt, value):
    """toe pitch ramps linearly from 0 to the liftoff value over dt before liftoff"""
    if dt <= 0 or value == 0.0:
        return 0.0
    u = (t - (t_event - dt)) / dt
    if u <= 0: return 0.0
    if u >= 1: return value
    return value * u


def pelvis_pose_at(sample, t):
    """Nominal pelvis pose (z-axis vertical, horizontal axes aligned with the
    path frame) plus the stylized z / rotation curves."""
    p = path_pose(sample.vel, t)
    z = sample.pelvis_z_nom + sample.c_pelvis_z(t % sample.T)
    yaw = p[2] + sample.c_pelvis_yaw(t % sample.T)
    pitch = sample.c_pelvis_pitch(t % sample.T)
    roll = sample.c_pelvis_roll(t % sample.T)
    off = rot_z(p[2]) @ np.array([PELVIS_X_OFFSET, 0.0, 0.0])
    return np.array([p[0] + off[0], p[1] + off[1], z]), rot_zyx(yaw, pitch, roll)


def solve_sample(sample, fps=60, passes=3):
    """Evaluate the reference cycle over [0, T) and solve the leg IK.

    Frame n (t = T) is not solved independently: by construction it is frame 0
    of the next cycle, so its joint angles are copied and only the root
    translation advances by the cycle displacement.  That makes the clip
    exactly periodic in joint space."""
    T = sample.T
    n = int(round(T * fps))
    times = np.arange(n) / fps                      # [0, T)
    q = {s: np.zeros((n, 5)) for s in SIDES}
    err = {s: np.zeros((n, 5)) for s in SIDES}
    pel_p = np.zeros((n, 3))
    pel_R = np.zeros((n, 3, 3))
    foot_p = {s: np.zeros((n, 3)) for s in SIDES}
    foot_R = {s: np.zeros((n, 3, 3)) for s in SIDES}
    contact = {s: np.zeros(n) for s in SIDES}

    q_seed = {s: np.zeros(5) for s in SIDES}
    for p_i in range(passes):
        for i, t in enumerate(times):
            pp, pR = pelvis_pose_at(sample, t)
            pel_p[i], pel_R[i] = pp, pR
            for s in SIDES:
                fp, fR, c = foot_pose_at(sample, s, t)
                yaw_t, pitch_t = yaw_pitch_of(pR.T @ fR)
                # Contact-height correction.  The paper's heel/toe rig keeps the
                # sole above ground in the sagittal plane; this robot also tilts
                # laterally (hip roll, no ankle roll), so the foot rides an edge.
                # Solve, read the achieved roll, lift the target so the lowest
                # sole corner sits exactly on the ground, and re-solve.
                lift, z0 = 0.0, fp[2]
                for _ in range(4):
                    qi, ei = LEGS[s].ik(fp + np.array([0.0, 0.0, lift]),
                                        yaw_t, pitch_t, q_seed[s], pp, pR)
                    _, Rach = LEGS[s].fk_sole(qi, pp, pR)
                    low = min((Rach @ c_)[2] for c_ in SOLE_CORNERS[s])
                    new_lift = max(0.0, -z0 - low)
                    if abs(new_lift - lift) < 1e-7:
                        break
                    lift = new_lift
                fp = fp + np.array([0.0, 0.0, lift])
                foot_p[s][i], foot_R[s][i], contact[s][i] = fp, Rach, c
                q[s][i], err[s][i] = qi, ei
                q_seed[s] = qi
        # next pass starts from the converged end-of-cycle pose

    disp = sample.cycle_displacement()
    return dict(times=times, n=n, fps=fps, q=q, err=err,
                pelvis_p=pel_p, pelvis_R=pel_R,
                foot_p=foot_p, foot_R=foot_R, contact=contact,
                cycle_displacement=disp, sample=sample)
