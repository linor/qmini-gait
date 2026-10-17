"""
gait_core.py -- kinematics, splines, phase map and gait parameterisation for the
QMini / BDX-style biped, following

    Hopkins, Wiedebach, Cesare, Bishop, Knoop, Bacher,
    "Interactive Design of Stylized Walking Gaits for Robotic Characters",
    ACM TOG 2024 (Disney Research / WDI R&D).

Conventions
-----------
ROBOT frame (as authored in qmini_urdf-2legs.usda, cm, Z-up):
    +X = robot LEFT, +Y = robot BACKWARD, +Z = UP
CHARACTER frame (used throughout this module, metres, Z-up):
    +x = FORWARD, +y = LEFT, +z = UP
    char = (-y_robot, x_robot, z_robot)

The robot has 5 revolute DoF per leg:
    hip_yaw   about +z          (both legs)
    hip_roll  about +x (left) / -x (right)     -> positive = abduction
    hip_pitch about -y          (both legs)    -> positive = swing forward
    knee      about -y          (both legs)
    ankle     about -y          (both legs)
There is NO ankle roll, so foot roll is produced solely by hip roll.
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. Robot kinematics (extracted from qmini_urdf-2legs.usda)
# ----------------------------------------------------------------------------

CM = 0.01

def _r2c(p):
    """robot frame (cm) -> character frame (m)"""
    x, y, z = p
    return np.array([-y, x, z]) * CM

# joint anchors in the robot frame (cm), left leg
_ANCHOR_L_ROBOT = {
    "hip_yaw":   ( 5.50, 10.289148,  -2.827945),
    "hip_roll":  (10.90,  8.489148, -10.127945),
    "hip_pitch": (13.80,  8.639149, -10.127945),
    "knee":      (13.77, 16.770876, -18.259672),
    "ankle":     (15.85, 11.469564, -32.824910),
}

CHAIN = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]

# joint limits (deg) straight from the USD PhysicsRevoluteJoints
LIMITS_DEG = {
    "L": {"hip_yaw": (-45.0, 30.0), "hip_roll": (-15.0, 15.0),
          "hip_pitch": (-50.0, 50.0), "knee": (-60.0, 50.0), "ankle": (-60.0, 40.0)},
    "R": {"hip_yaw": (-30.0, 45.0), "hip_roll": (-15.0, 15.0),
          "hip_pitch": (-50.0, 50.0), "knee": (-50.0, 60.0), "ankle": (-40.0, 60.0)},
}

# The USD authors mirror-image limits for the left and right legs, but the
# extracted world rotation axes for hip_pitch / knee / ankle are IDENTICAL on
# both sides (quaternion localRot0 matches to 1e-18).  A mirrored mechanism with
# a shared axis cannot have mirrored limits, so one of the two is a CAD-export
# artifact.  We therefore drive the animations inside the intersection of the
# two, which is safe under either interpretation, and keep the raw per-side
# limits in the USD metadata for reference.
SAFE_LIMITS_DEG = {
    "hip_yaw": (-30.0, 30.0), "hip_roll": (-15.0, 15.0), "hip_pitch": (-50.0, 50.0),
    "knee": (-50.0, 50.0), "ankle": (-40.0, 40.0),
}

# USD physics joint each skeleton joint corresponds to
PHYSICS_JOINT = {
    ("L", "hip_yaw"):   "Revolute_left_yaw",
    ("L", "hip_roll"):  "Revolute_left_roll",
    ("L", "hip_pitch"): "Revolute_left_pitch",
    ("L", "knee"):      "Revolute_left_knee",
    ("L", "ankle"):     "Revolute_left_ankle",
    ("R", "hip_yaw"):   "Revolute_right_yaw",
    ("R", "hip_roll"):  "Revolute_right_roll",
    ("R", "hip_pitch"): "Revolute_right_pitch",
    ("R", "knee"):      "Revolute_right_knee",
    ("R", "ankle"):     "Revolute_right_ankle",
}

EX, EY, EZ = np.eye(3)

def anchors(side):
    """rest-pose joint anchors, character frame, metres, relative to base_link origin"""
    out = {}
    for k, v in _ANCHOR_L_ROBOT.items():
        p = _r2c(v)
        if side == "R":
            p = p * np.array([1.0, -1.0, 1.0])
        out[k] = p
    return out

def axes(side):
    """joint rotation axes in the character frame at rest"""
    s = 1.0 if side == "L" else -1.0
    return {
        "hip_yaw":   EZ.copy(),
        "hip_roll":  s * EX,
        "hip_pitch": -EY,
        "knee":      -EY,
        "ankle":     -EY,
    }

# --- foot geometry (from the collision meshes) ------------------------------
SOLE_Z          = -0.42410      # sole plane, char z, rest pose
ANKLE_Z         = -0.32825
ANKLE_HEIGHT    = ANKLE_Z - SOLE_Z          # +0.09585 : ankle above sole
SOLE_CENTER_X   = -0.10490                  # char x of the contact-patch centre
ANKLE_X         = -0.11470
ANKLE_DX        = ANKLE_X - SOLE_CENTER_X   # -0.00980 : ankle behind sole centre
FOOT_HALF_LEN   = 0.06440                   # heel/toe from the sole centre
NOMINAL_HEIGHT  = -SOLE_Z                   # 0.4241 m : base_link above ground at rest
NOMINAL_HALF_W  = 0.15850                   # |char y| of the ankle at rest
# The base_link origin (our pelvis reference) is NOT above the sole centres in
# the CAD rest pose -- it sits 0.1049 m forward of them.  The path frame is
# anchored at the feet, so the pelvis carries this constant forward offset;
# with it, zero velocity + zero style reproduces the CAD rest pose exactly.
PELVIS_X_OFFSET = -SOLE_CENTER_X            # +0.1049 m
# Neutral fore-aft stance: put the ankle directly under the hip-pitch axis so
# the stance excursion is symmetric about the cheapest (shortest-reach) posture.
HIP_PITCH_X     = -0.08639
STANCE_X_NEUTRAL = PELVIS_X_OFFSET + HIP_PITCH_X - ANKLE_DX   # +0.02831 m

# ankle position expressed in the FOOTHOLD frame F
#   (origin = sole contact centre on the ground, x fwd, y left, z up)
ANKLE_IN_FOOT = np.array([ANKLE_DX, 0.0, ANKLE_HEIGHT])
HEEL_IN_FOOT  = np.array([-FOOT_HALF_LEN, 0.0, 0.0])
TOE_IN_FOOT   = np.array([+FOOT_HALF_LEN, 0.0, 0.0])
# Four corners of the contact patch in the foot frame, relative to the sole
# origin.  The patch is only 18.5 mm wide and sits slightly inboard of the ankle.
_SOLE_Y_IN, _SOLE_Y_OUT = +0.0044, -0.0141
SOLE_CORNERS = {
    "L": [np.array([sx, sy, 0.0]) for sx in (-FOOT_HALF_LEN, FOOT_HALF_LEN)
          for sy in (_SOLE_Y_IN, _SOLE_Y_OUT)],
    "R": [np.array([sx, -sy, 0.0]) for sx in (-FOOT_HALF_LEN, FOOT_HALF_LEN)
          for sy in (_SOLE_Y_IN, _SOLE_Y_OUT)],
}


# ----------------------------------------------------------------------------
# 2. Rotation helpers
# ----------------------------------------------------------------------------

def rot_axis(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)

def rot_z(a): return rot_axis(EZ, a)
def rot_y(a): return rot_axis(EY, a)
def rot_x(a): return rot_axis(EX, a)

def rot_zyx(yaw, pitch, roll):
    """intrinsic Z-Y-X euler -> matrix (paper's pelvis / foot convention)"""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def mat_to_quat(R):
    """-> (w, x, y, z)"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def yaw_pitch_of(R):
    """extract intrinsic Z-Y-X yaw / pitch from a rotation matrix"""
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return yaw, pitch


# ----------------------------------------------------------------------------
# 3. Splines  (paper Sec. 4.3: "parameterised as cubic splines")
# ----------------------------------------------------------------------------

class HermiteSpline:
    """Piecewise cubic Hermite spline through (t, v) knots with explicit or
    auto (Catmull-Rom / periodic) tangents.  Evaluated analytically -- no
    keyframe sampling anywhere in the pipeline."""

    def __init__(self, knots, values, tangents=None, periodic=False):
        self.t = np.asarray(knots, float)
        self.v = np.asarray(values, float)
        self.periodic = periodic
        assert np.all(np.diff(self.t) > 0), "knots must be strictly increasing"
        if tangents is None:
            tangents = self._auto_tangents()
        self.m = np.asarray(tangents, float)

    def _auto_tangents(self):
        t, v, n = self.t, self.v, len(self.t)
        m = np.zeros(n)
        if n == 1:
            return m
        if self.periodic:
            # the last knot duplicates the first (t[-1] == t[0] + T), so the
            # independent nodes are 0..n-2 and the tangent wraps around.
            assert abs(v[-1] - v[0]) < 1e-12, "periodic curve must close in value"
            T = t[-1] - t[0]
            k = n - 1
            for i in range(k):
                ip, im = (i + 1) % k, (i - 1) % k
                dt_next = t[i + 1] - t[i]
                dt_prev = t[i] - t[im] if im < i else t[i] + T - t[im]
                dv_next = v[i + 1] - v[i]
                dv_prev = v[i] - v[im]
                m[i] = 0.5 * (dv_next / dt_next + dv_prev / dt_prev)
            m[-1] = m[0]
        else:
            m[0] = (v[1] - v[0]) / (t[1] - t[0])
            m[-1] = (v[-1] - v[-2]) / (t[-1] - t[-2])
            for i in range(1, n - 1):
                m[i] = 0.5 * ((v[i + 1] - v[i]) / (t[i + 1] - t[i])
                              + (v[i] - v[i - 1]) / (t[i] - t[i - 1]))
        return m

    def __call__(self, x):
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, float))
        t, v, m = self.t, self.v, self.m
        if self.periodic:
            T = t[-1] - t[0]
            x = t[0] + np.mod(x - t[0], T)
        xc = np.clip(x, t[0], t[-1])
        i = np.clip(np.searchsorted(t, xc, side="right") - 1, 0, len(t) - 2)
        h = t[i + 1] - t[i]
        s = (xc - t[i]) / h
        s2, s3 = s * s, s * s * s
        h00 = 2 * s3 - 3 * s2 + 1
        h10 = s3 - 2 * s2 + s
        h01 = -2 * s3 + 3 * s2
        h11 = s3 - s2
        out = h00 * v[i] + h10 * h * m[i] + h01 * v[i + 1] + h11 * h * m[i + 1]
        return float(out[0]) if scalar else out


class MonotonicSpline(HermiteSpline):
    """Fritsch-Carlson monotone cubic -- used for the phase map Omega(phi) so
    that it stays strictly invertible (paper Sec. 5.1)."""

    def __init__(self, knots, values):
        t = np.asarray(knots, float)
        v = np.asarray(values, float)
        n = len(t)
        d = np.diff(v) / np.diff(t)
        m = np.empty(n)
        m[0], m[-1] = d[0], d[-1]
        m[1:-1] = 0.5 * (d[:-1] + d[1:])
        for i in range(n - 1):
            if d[i] == 0:
                m[i] = m[i + 1] = 0.0
            else:
                a, b = m[i] / d[i], m[i + 1] / d[i]
                s = a * a + b * b
                if s > 9.0:
                    tau = 3.0 / np.sqrt(s)
                    m[i] = tau * a * d[i]
                    m[i + 1] = tau * b * d[i]
        super().__init__(t, v, m)

    def inverse(self, y, iters=60):
        """numeric inverse by bisection (monotone => well posed)"""
        scalar = np.isscalar(y)
        y = np.atleast_1d(np.asarray(y, float))
        lo = np.full_like(y, self.t[0])
        hi = np.full_like(y, self.t[-1])
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = self(mid)
            up = f < y
            lo = np.where(up, mid, lo)
            hi = np.where(up, hi, mid)
        r = 0.5 * (lo + hi)
        return float(r[0]) if scalar else r


def f_soft(a):
    """Eq. 8 -- smoothstep with zero first derivative at both limits."""
    a = np.clip(a, 0.0, 1.0)
    return -2 * a ** 3 + 3 * a ** 2


# ----------------------------------------------------------------------------
# 4. Phase map  (paper Sec. 5.1)
# ----------------------------------------------------------------------------

PHI_LLO, PHI_LTD, PHI_RLO, PHI_RTD, PHI_END = 0.0, 0.1, 0.5, 0.6, 1.0
PHI_LANDMARKS = (PHI_LLO, PHI_LTD, PHI_RLO, PHI_RTD, PHI_END)
EVENT_NAMES = ("LLO", "LTD", "RLO", "RTD", "CYCLE_END")


class PhaseMap:
    """Invertible Omega(phi) -> gait time, built as a monotone cubic through the
    five (phi, t) contact-event way-points."""

    def __init__(self, t_llo, t_ltd, t_rlo, t_rtd, T):
        self.events = np.array([t_llo, t_ltd, t_rlo, t_rtd, t_llo + T], float)
        assert np.all(np.diff(self.events) > 0), "contact events out of order"
        self.T = T
        self.spline = MonotonicSpline(PHI_LANDMARKS, self.events)

    def __call__(self, phi):
        return self.spline(phi)

    def phase_of(self, t):
        return self.spline.inverse(t)


# ----------------------------------------------------------------------------
# 5. Gait sample parameterisation  (paper Tabs. 1-3)
# ----------------------------------------------------------------------------

class GaitSample:
    """A complete authored walk cycle: time intervals + attributes + function
    curves, associated with one key velocity from Tab. 3."""

    def __init__(self, name, vel, T, f_ltd, f_rlo, f_rtd,
                 stance_width, stance_x_bias, stance_splay,
                 swing_height, psi_toe_lo, psi_heel_td,
                 dt_psi_lo, dt_psi_td,
                 pelvis_z, pelvis_bounce, pelvis_roll, pelvis_pitch,
                 swing_x_lead=0.0, swing_y_lead=0.0, foot_yaw_amp=0.0,
                 pelvis_sway=0.0, notes=""):
        self.name = name
        self.vel = np.asarray(vel, float)          # (xdot, ydot, thetadot)
        self.T = T

        # --- Tab. 1 time intervals -------------------------------------
        self.t_llo = 0.0
        self.t_ltd = f_ltd * T
        self.t_rlo = f_rlo * T
        self.t_rtd = f_rtd * T
        self.phase = PhaseMap(self.t_llo, self.t_ltd, self.t_rlo, self.t_rtd, T)

        # --- attributes -------------------------------------------------
        self.stance_width = stance_width
        self.stance_x_bias = stance_x_bias
        self.stance_splay = stance_splay
        self.dt_psi_lo = dt_psi_lo
        self.dt_psi_td = dt_psi_td
        self.pelvis_z_nom = pelvis_z
        self.notes = notes

        LS = (self.t_llo, self.t_ltd)
        RS = (self.t_rlo, self.t_rtd)
        self.LS, self.RS = LS, RS

        # --- Tab. 2 function curves -------------------------------------
        # swing curves are clamped to zero at the interval boundaries so the
        # foot starts / ends exactly at the nominal foothold pose.
        def swing_curve(iv, peak, at=0.5, tan0=0.0, tan1=0.0):
            a, b = iv
            return HermiteSpline([a, a + (b - a) * at, b], [0.0, peak, 0.0],
                                 tangents=[tan0, 0.0, tan1])

        def swing_z(iv, h):
            a, b = iv
            d = b - a
            # rise fast, apex at 45%, controlled descent -> soft touchdown
            return HermiteSpline(
                [a, a + 0.28 * d, a + 0.45 * d, a + 0.75 * d, b],
                [0.0, 0.86 * h, h, 0.62 * h, 0.0],
                tangents=[3.4 * h / d, 1.2 * h / d, 0.0, -1.6 * h / d, -1.35 * h / d])

        self.c_foot_z = {"L": swing_z(LS, swing_height), "R": swing_z(RS, swing_height)}
        self.c_foot_x = {"L": swing_curve(LS, swing_x_lead, 0.55),
                         "R": swing_curve(RS, swing_x_lead, 0.55)}
        self.c_foot_y = {"L": swing_curve(LS, +swing_y_lead, 0.5),
                         "R": swing_curve(RS, -swing_y_lead, 0.5)}
        self.c_foot_yaw = {"L": swing_curve(LS, +foot_yaw_amp, 0.5),
                           "R": swing_curve(RS, -foot_yaw_amp, 0.5)}
        self.c_foot_gamma = {"L": swing_curve(LS, 0.0), "R": swing_curve(RS, 0.0)}

        # heel / toe pitch over the swing interval:
        #   psi_toe  starts at the toe-off value and relaxes to 0
        #   psi_heel arrives at the heel-strike value at touchdown
        def toe_curve(iv, v0):
            a, b = iv
            d = b - a
            return HermiteSpline([a, a + 0.35 * d, b], [v0, 0.0, 0.0],
                                 tangents=[0.0, 0.0, 0.0])

        def heel_curve(iv, v1):
            a, b = iv
            d = b - a
            return HermiteSpline([a, a + 0.60 * d, b], [0.0, 0.0, v1],
                                 tangents=[0.0, 0.0, 0.0])

        self.c_psi_toe = {"L": toe_curve(LS, psi_toe_lo), "R": toe_curve(RS, psi_toe_lo)}
        self.c_psi_heel = {"L": heel_curve(LS, psi_heel_td), "R": heel_curve(RS, psi_heel_td)}
        self.psi_toe_lo = psi_toe_lo
        self.psi_heel_td = psi_heel_td

        # --- periodic (CY) curves, C1 across the cycle loop ---------------
        # pelvis rises over each single-support phase -> two bounces per cycle
        mid_ls = 0.5 * (self.t_llo + self.t_ltd)
        mid_rs = 0.5 * (self.t_rlo + self.t_rtd)
        self.c_pelvis_z = HermiteSpline(
            [self.t_llo, mid_ls, self.t_rlo, mid_rs, T],
            [-pelvis_bounce, +pelvis_bounce, -pelvis_bounce, +pelvis_bounce, -pelvis_bounce],
            periodic=True)
        # pelvis leans toward the support foot.  Roll is right-handed about the
        # forward axis, so +roll raises the left side (= lean right).  During the
        # left swing the right foot supports -> lean right -> +roll.
        self.c_pelvis_roll = HermiteSpline(
            [self.t_llo, mid_ls, self.t_rlo, mid_rs, T],
            [0.0, +pelvis_roll, 0.0, -pelvis_roll, 0.0], periodic=True)
        # pelvis SWAYS toward the support foot (not in the paper; added
        # 2026-10-17): a lateral translation in the path frame, y = left, so
        # during the left swing (right foot supports) it moves right (-y).
        # With pelvis_roll alone the reference never shifts the weight; the
        # RL policy did it by rocking the whole robot on its laterally rounded
        # soles (~6-8 deg body roll). Translating the pelvis instead tilts both
        # legs together through the hip roll joints (the soles roll, the body
        # stays level). 0 = the authored gaits unchanged.
        self.pelvis_sway = pelvis_sway
        self.c_pelvis_y = HermiteSpline(
            [self.t_llo, mid_ls, self.t_rlo, mid_rs, T],
            [0.0, -pelvis_sway, 0.0, +pelvis_sway, 0.0], periodic=True)
        self.c_pelvis_pitch = HermiteSpline(
            [self.t_llo, mid_ls, self.t_rlo, mid_rs, T],
            [pelvis_pitch, pelvis_pitch, pelvis_pitch, pelvis_pitch, pelvis_pitch],
            periodic=True)
        self.c_pelvis_yaw = HermiteSpline(
            [self.t_llo, mid_ls, self.t_rlo, mid_rs, T],
            [0.0, 0.0, 0.0, 0.0, 0.0], periodic=True)

    # ---- contact bookkeeping -------------------------------------------
    def contact_state(self, t):
        t = t % self.T
        if t < self.t_ltd:   return "LS", 0.0, 1.0
        if t < self.t_rlo:   return "DS1", 1.0, 1.0
        if t < self.t_rtd:   return "RS", 1.0, 0.0
        return "DS2", 1.0, 1.0

    def cycle_displacement(self):
        p = path_pose(self.vel, self.T)
        return np.array([p[0], p[1], p[2]])


# ----------------------------------------------------------------------------
# 6. Path frame (paper Sec. 4.2) -- constant-velocity arc extrapolation
# ----------------------------------------------------------------------------

def path_pose(vel, t):
    """planar path pose [x, y, theta] = integral of the body-frame velocity"""
    xd, yd, td = vel
    th = td * t
    if abs(td) < 1e-9:
        return np.array([xd * t, yd * t, th])
    s, c = np.sin(th), np.cos(th)
    x = (xd * s - yd * (1 - c)) / td
    y = (xd * (1 - c) + yd * s) / td
    return np.array([x, y, th])


def compose(pa, pb):
    """planar pose composition  pa (+) pb"""
    xa, ya, ta = pa
    xb, yb, tb = pb
    c, s = np.cos(ta), np.sin(ta)
    return np.array([xa + c * xb - s * yb, ya + s * xb + c * yb, ta + tb])
