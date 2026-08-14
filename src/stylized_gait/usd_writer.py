"""
usd_writer.py -- emit one self-contained UsdSkel .usda per gait sample.

Output frame / units match qmini_urdf-2legs.usda exactly:
    centimetres, metersPerUnit = 0.01, upAxis = "Z",
    +X = robot LEFT, +Y = robot BACKWARD, +Z = UP
so the skeleton overlays the existing robot model without any conversion.

Every clip is tagged with the full phase-space blending metadata from the paper:
contact event times, the phi landmarks (0, 0.1, 0.5, 0.6, 1), the monotone
phase-map knots Omega(phi), the Tab. 1 time intervals, the Tab. 3 sample
velocity, the stance attributes, and per-frame phase / contact-state tracks.
"""

import numpy as np
from .core import (
    CHAIN, LIMITS_DEG, SAFE_LIMITS_DEG, PHYSICS_JOINT, PHI_LANDMARKS, EVENT_NAMES,
    mat_to_quat, rot_axis, NOMINAL_HEIGHT, STANCE_X_NEUTRAL, PELVIS_X_OFFSET,
)
from .planner import LEGS, SIDES, stance_pose

M100 = 100.0                      # metres -> centimetres
# character (x fwd, y left, z up) -> robot (x left, y back, z up)
C2R = np.array([[0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]])

JOINT_PATHS = ["Root"]
for _s in ("L", "R"):
    _p = "Root"
    for _j in CHAIN:
        _p = f"{_p}/{_s}_{_j}"
        JOINT_PATHS.append(_p)
JOINT_NAMES = [p.split("/")[-1] for p in JOINT_PATHS]
NJ = len(JOINT_PATHS)              # 11


def _mat4(R, t):
    """USD matrix4d: row-vector convention, translation in the last row."""
    m = np.eye(4)
    m[:3, :3] = R.T
    m[3, :3] = t
    return m


def _fmt_m4(m):
    rows = ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in m)
    return f"( {rows} )"


def _fmt_v3(v):
    return "(" + ", ".join(f"{x:.7g}" for x in v) + ")"


def _fmt_q(q):
    return "(" + ", ".join(f"{x:.9g}" for x in q) + ")"


def skeleton_tables():
    """bind (world) and rest (local) transforms, in robot cm."""
    binds, rests = [], []
    binds.append(_mat4(np.eye(3), np.zeros(3)))      # Root == base_link origin
    rests.append(_mat4(np.eye(3), np.zeros(3)))
    for s in SIDES:
        leg = LEGS[s]
        for j in CHAIN:
            world_c = leg.a[j]
            local_c = leg.local[j]
            binds.append(_mat4(np.eye(3), C2R @ world_c * M100))
            rests.append(_mat4(np.eye(3), C2R @ local_c * M100))
    return binds, rests


def joint_axes_robot():
    out = [np.array([0.0, 0.0, 1.0])]                # Root: unused
    for s in SIDES:
        for j in CHAIN:
            out.append(C2R @ LEGS[s].ax[j])
    return out


def sample_rotations(res, i):
    """local rotation quaternion per joint at frame i (robot frame)."""
    q = [mat_to_quat(C2R @ res["pelvis_R"][i] @ C2R.T)]
    ax = joint_axes_robot()
    k = 1
    for s in SIDES:
        for jj in range(len(CHAIN)):
            q.append(mat_to_quat(rot_axis(ax[k], res["q"][s][i][jj])))
            k += 1
    return q


def write_usda(res, path, fps=60):
    smp = res["sample"]
    n = res["n"]
    T = smp.T
    disp = res["cycle_displacement"]              # (dx, dy, dtheta) char frame, m/rad

    binds, rests = skeleton_tables()
    prim = "".join(w.capitalize() for w in smp.name.split("_"))

    # ---- per-frame data, plus the wrap frame n (== frame 0 advanced) --------
    root_t, root_q, rot_frames = [], [], []
    phase, contact_state, lc, rc = [], [], [], []
    for i in range(n + 1):
        src = i % n
        cyc = i // n
        p_c = res["pelvis_p"][src].copy()
        if cyc:                                    # advance by one cycle
            th = disp[2] * cyc
            c, s = np.cos(th), np.sin(th)
            base = np.array([disp[0], disp[1], 0.0])
            p_c = np.array([c * p_c[0] - s * p_c[1], s * p_c[0] + c * p_c[1], p_c[2]]) + base
            R_c = rot_axis(np.array([0.0, 0.0, 1.0]), th) @ res["pelvis_R"][src]
        else:
            R_c = res["pelvis_R"][src]
        root_t.append(C2R @ p_c * M100)
        qs = sample_rotations(res, src)
        qs[0] = mat_to_quat(C2R @ R_c @ C2R.T)
        rot_frames.append(qs)

        t = (i / fps)
        tm = t % T if i < n else T
        phase.append(float(smp.phase.phase_of(min(tm, T))))
        st, l, r = smp.contact_state(t if i < n else 0.0)
        contact_state.append(st); lc.append(l); rc.append(r)

    ev = smp.phase.events
    om_phi = np.linspace(0.0, 1.0, 21)
    om_t = smp.phase(om_phi)

    L = []
    A = L.append
    A('#usda 1.0')
    A('(')
    A(f'    defaultPrim = "{prim}"')
    A('    metersPerUnit = 0.01')
    A('    kilogramsPerUnit = 1')
    A('    upAxis = "Z"')
    A(f'    timeCodesPerSecond = {fps}')
    A(f'    framesPerSecond = {fps}')
    A('    startTimeCode = 0')
    A(f'    endTimeCode = {n}')
    A('    customLayerData = {')
    A(f'        string generator = "stylized_gait reference animation generator"')
    A(f'        string paper = "Hopkins et al., Interactive Design of Stylized Walking Gaits for Robotic Characters, ACM TOG 2024"')
    A(f'        string robot = "qmini_urdf-2legs"')
    A(f'        string sample = "{smp.name}"')
    A('    }')
    A(')')
    A('')
    A(f'def SkelRoot "{prim}" (')
    A('    prepend apiSchemas = ["SkelBindingAPI"]')
    A(')')
    A('{')

    # ---------------- gait metadata --------------------------------------
    A('    # ---- Tab. 3 sample identity (SI units: m/s, rad/s) ----')
    A(f'    custom string stylizedGait:sampleName = "{smp.name}"')
    A(f'    custom string stylizedGait:notes = "{smp.notes}"')
    A(f'    custom double3 stylizedGait:velocity_mps_radps = ({smp.vel[0]:.6g}, {smp.vel[1]:.6g}, {smp.vel[2]:.6g})')
    A(f'    custom double3 stylizedGait:cycleDisplacement_m_rad = ({disp[0]:.9g}, {disp[1]:.9g}, {disp[2]:.9g})')
    A(f'    custom double stylizedGait:cyclePeriod = {T:.9g}')
    A('')
    A('    # ---- Tab. 1 time intervals (min, max), seconds from cycle start ----')
    A(f'    custom double2 stylizedGait:interval:CY = ({smp.t_llo:.9g}, {smp.t_llo + T:.9g})')
    A(f'    custom double2 stylizedGait:interval:LS = ({smp.t_llo:.9g}, {smp.t_ltd:.9g})')
    A(f'    custom double2 stylizedGait:interval:RS = ({smp.t_rlo:.9g}, {smp.t_rtd:.9g})')
    A('')
    A('    # ---- Sec. 5.1 phase map: contact events tagged for blending ----')
    A('    custom token[] stylizedGait:eventNames = [' + ", ".join(f'"{e}"' for e in EVENT_NAMES) + ']')
    A('    custom double[] stylizedGait:eventTimes = [' + ", ".join(f"{v:.9g}" for v in ev) + ']')
    A('    custom double[] stylizedGait:eventPhases = [' + ", ".join(f"{v:.9g}" for v in PHI_LANDMARKS) + ']')
    A('    custom double[] stylizedGait:phaseMap:phi = [' + ", ".join(f"{v:.6g}" for v in om_phi) + ']')
    A('    custom double[] stylizedGait:phaseMap:time = [' + ", ".join(f"{v:.9g}" for v in om_t) + ']')
    A('    custom string stylizedGait:phaseMap:interpolation = "monotonic cubic (Fritsch-Carlson)"')
    A('')
    A('    # ---- Sec. 4.3 attributes (metres / radians) ----')
    A(f'    custom double stylizedGait:attr:stanceWidth = {smp.stance_width:.9g}')
    A(f'    custom double stylizedGait:attr:stanceXBias = {STANCE_X_NEUTRAL + smp.stance_x_bias:.9g}')
    A(f'    custom double stylizedGait:attr:stanceSplay = {smp.stance_splay:.9g}')
    # evaluate rather than read control points: a blended or library sample's
    # curve has no .v to peek at, and the apex of a Hermite is not necessarily
    # at a knot anyway
    _apex = max(abs(float(smp.c_foot_z["L"](t)))
                for t in np.linspace(smp.t_llo, smp.t_ltd, 129))
    A(f'    custom double stylizedGait:attr:swingHeight = {_apex:.9g}')
    _sc = lambda v: float(v["L"]) if isinstance(v, dict) else float(v)
    A(f'    custom double stylizedGait:attr:psiToeLiftoff = {_sc(smp.psi_toe_lo):.9g}')
    A(f'    custom double stylizedGait:attr:psiHeelTouchdown = {_sc(smp.psi_heel_td):.9g}')
    A(f'    custom double stylizedGait:attr:dtPsiLO = {smp.dt_psi_lo:.9g}')
    A(f'    custom double stylizedGait:attr:dtPsiTD = {smp.dt_psi_td:.9g}')
    A(f'    custom double stylizedGait:attr:pelvisHeight = {smp.pelvis_z_nom:.9g}')
    A('')
    A('    # ---- joint mapping back onto the PhysicsRevoluteJoints ----')
    A('    custom token[] stylizedGait:physicsJoints = [' +
      ", ".join('""' if i == 0 else f'"{PHYSICS_JOINT[(p.split("/")[-1][0], p.split("/")[-1][2:])]}"'
                for i, p in enumerate(JOINT_PATHS)) + ']')
    A('    custom double2[] stylizedGait:jointLimits_rad = [' +
      ", ".join('(0, 0)' if i == 0 else
                "({:.6g}, {:.6g})".format(*np.deg2rad(LIMITS_DEG[p.split("/")[-1][0]][p.split("/")[-1][2:]]))
                for i, p in enumerate(JOINT_PATHS)) + ']')
    A('    custom double2[] stylizedGait:jointLimitsSafe_rad = [' +
      ", ".join('(0, 0)' if i == 0 else
                "({:.6g}, {:.6g})".format(*np.deg2rad(SAFE_LIMITS_DEG[p.split("/")[-1][2:]]))
                for i, p in enumerate(JOINT_PATHS)) + ']')
    A('')

    # ---------------- per-frame phase / contact tracks --------------------
    A('    # ---- per-frame blending tracks ----')
    A('    float primvars:phase = 0 (')
    A('        interpolation = "constant"')
    A('    )')
    A('    float primvars:phase.timeSamples = {')
    for i, v in enumerate(phase):
        A(f'        {i}: {v:.7g},')
    A('    }')
    A('    token primvars:contactState = "DS1" (')
    A('        interpolation = "constant"')
    A('    )')
    A('    token primvars:contactState.timeSamples = {')
    for i, v in enumerate(contact_state):
        A(f'        {i}: "{v}",')
    A('    }')
    A('    float primvars:leftContact = 1 (')
    A('        interpolation = "constant"')
    A('    )')
    A('    float primvars:leftContact.timeSamples = {')
    for i, v in enumerate(lc):
        A(f'        {i}: {v:.0f},')
    A('    }')
    A('    float primvars:rightContact = 1 (')
    A('        interpolation = "constant"')
    A('    )')
    A('    float primvars:rightContact.timeSamples = {')
    for i, v in enumerate(rc):
        A(f'        {i}: {v:.0f},')
    A('    }')
    A('    float primvars:cycleTimeNormalized = 0 (')
    A('        interpolation = "constant"')
    A('    )')
    A('    float primvars:cycleTimeNormalized.timeSamples = {')
    for i in range(n + 1):
        A(f'        {i}: {min(i / fps / T, 1.0):.7g},')
    A('    }')
    A('')

    # ---------------- skeleton --------------------------------------------
    A('    def Skeleton "Skel" (')
    A('        prepend apiSchemas = ["SkelBindingAPI"]')
    A('    )')
    A('    {')
    A('        uniform token[] joints = [' + ", ".join(f'"{p}"' for p in JOINT_PATHS) + ']')
    A('        uniform token[] jointNames = [' + ", ".join(f'"{p}"' for p in JOINT_NAMES) + ']')
    A('        uniform matrix4d[] bindTransforms = [' + ", ".join(_fmt_m4(m) for m in binds) + ']')
    A('        uniform matrix4d[] restTransforms = [' + ", ".join(_fmt_m4(m) for m in rests) + ']')
    A('        rel skel:animationSource = </%s/Skel/Anim>' % prim)
    A('')
    A('        def SkelAnimation "Anim"')
    A('        {')
    A('            uniform token[] joints = [' + ", ".join(f'"{p}"' for p in JOINT_PATHS) + ']')
    A('            float3[] scales = [' + ", ".join("(1, 1, 1)" for _ in range(NJ)) + ']')
    # translations: only Root moves; the rest are the constant rest offsets
    A('            float3[] translations.timeSamples = {')
    for i in range(n + 1):
        vals = [root_t[i]] + [np.array(m[3, :3]) for m in rests[1:]]
        A(f'                {i}: [' + ", ".join(_fmt_v3(v) for v in vals) + '],')
    A('            }')
    A('            quatf[] rotations.timeSamples = {')
    for i in range(n + 1):
        A(f'                {i}: [' + ", ".join(_fmt_q(q) for q in rot_frames[i]) + '],')
    A('            }')
    A('        }')
    A('    }')
    A('}')

    open(path, "w").write("\n".join(L) + "\n")
    return path
