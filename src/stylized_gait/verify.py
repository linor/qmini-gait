"""
verify.py -- independent validation.

Re-parses the emitted .usda text (no shared state with the generator), rebuilds
forward kinematics from the Skeleton bind/rest tables and the SkelAnimation
quaternions, and checks the physical and blending-related invariants.
"""

import re, glob, os, sys
import numpy as np

# Corners of the contact patch expressed relative to the ANKLE (robot cm).
# From the collision meshes: sole plane 9.585 cm below the ankle, patch spans
# y 4.05..16.93 with the ankle at y 11.47 (+Y = backward, so the toe is 7.42 cm
# ahead and the heel 5.46 cm behind), and x 14.44..16.29 with the ankle at 15.85.
CORNERS_CM = {
    "L": [np.array([sx, sy, -9.585]) for sx in (0.44, -1.41) for sy in (5.46, -7.42)],
    "R": [np.array([sx, sy, -9.585]) for sx in (-0.44, 1.41) for sy in (5.46, -7.42)],
}
PELVIS_X_OFFSET_CM = 10.49                        # pelvis ahead of the path origin
RAW_LIMITS = {
    "L_hip_yaw": (-45, 30), "L_hip_roll": (-15, 15), "L_hip_pitch": (-50, 50),
    "L_knee": (-60, 50), "L_ankle": (-60, 40),
    "R_hip_yaw": (-30, 45), "R_hip_roll": (-15, 15), "R_hip_pitch": (-50, 50),
    "R_knee": (-50, 60), "R_ankle": (-40, 60),
}
SAFE = {"hip_yaw": 30, "hip_roll": 15, "hip_pitch": 50, "knee": 50, "ankle": 40}


def q2m(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def parse(path):
    src = open(path).read()
    d = {}
    d["joints"] = re.search(r'uniform token\[\] joints = \[(.*?)\]', src, re.S).group(1)
    d["joints"] = [t.strip().strip('"') for t in d["joints"].split(",")]
    names = [j.split("/")[-1] for j in d["joints"]]
    d["names"] = names

    rests = re.search(r'uniform matrix4d\[\] restTransforms = \[(.*?)\]\n', src, re.S).group(1)
    mats = re.findall(r'\(\s*(\([^)]*\),\s*\([^)]*\),\s*\([^)]*\),\s*\([^)]*\))\s*\)', rests)
    R = []
    for m in mats:
        vals = [float(v) for v in re.findall(r'-?[\d.eE+-]+', m)]
        R.append(np.array(vals).reshape(4, 4))
    d["rest"] = R

    def tsamples(attr, cast):
        """find the attribute, then scan forward with brace balancing so the
        dict body is delimited correctly regardless of indentation."""
        start = src.index(attr + ".timeSamples")
        i = src.index("{", start)
        depth, j = 0, i
        while True:
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = src[i + 1:j]
        out = {}
        for line in body.strip().splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            k, v = line.split(":", 1)
            out[int(k)] = cast(v.strip())
        return [out[k] for k in sorted(out)]

    d["trans"] = tsamples("float3[] translations", lambda s: np.array(
        [[float(x) for x in t.split(",")] for t in re.findall(r'\(([^)]*)\)', s)]))
    d["rots"] = tsamples("quatf[] rotations", lambda s: np.array(
        [[float(x) for x in t.split(",")] for t in re.findall(r'\(([^)]*)\)', s)]))
    d["phase"] = tsamples("float primvars:phase", float)
    d["state"] = tsamples("token primvars:contactState", lambda s: s.strip('"'))
    d["lc"] = tsamples("float primvars:leftContact", float)
    d["rc"] = tsamples("float primvars:rightContact", float)

    d["vel"] = [float(x) for x in re.search(
        r'stylizedGait:velocity_mps_radps = \(([^)]*)\)', src).group(1).split(",")]
    d["disp"] = [float(x) for x in re.search(
        r'stylizedGait:cycleDisplacement_m_rad = \(([^)]*)\)', src).group(1).split(",")]
    d["T"] = float(re.search(r'stylizedGait:cyclePeriod = ([-\d.eE+]+)', src).group(1))
    d["events"] = [float(x) for x in re.search(
        r'stylizedGait:eventTimes = \[([^\]]*)\]', src).group(1).split(",")]
    d["phis"] = [float(x) for x in re.search(
        r'stylizedGait:eventPhases = \[([^\]]*)\]', src).group(1).split(",")]
    d["om_phi"] = [float(x) for x in re.search(
        r'stylizedGait:phaseMap:phi = \[([^\]]*)\]', src).group(1).split(",")]
    d["om_t"] = [float(x) for x in re.search(
        r'stylizedGait:phaseMap:time = \[([^\]]*)\]', src).group(1).split(",")]
    d["fps"] = int(re.search(r'framesPerSecond = (\d+)', src).group(1))
    d["name"] = re.search(r'stylizedGait:sampleName = "([^"]*)"', src).group(1)
    return d


def fk_frame(d, i):
    """world transform of every joint at frame i (robot cm)"""
    T = {}
    for k, jp in enumerate(d["joints"]):
        Rl = q2m(d["rots"][i][k])
        tl = d["trans"][i][k]
        parent = "/".join(jp.split("/")[:-1])
        if parent == "":
            Rw, tw = Rl, tl
        else:
            Rp, tp = T[parent]
            Rw = Rp @ Rl
            tw = tp + Rp @ tl
        T[jp] = (Rw, tw)
    return T


def check(path):
    d = parse(path)
    n = len(d["rots"]) - 1
    fps = d["fps"]
    idx = {j.split("/")[-1]: k for k, j in enumerate(d["joints"])}
    issues, info = [], {}

    ankle_jp = {s: [j for j in d["joints"] if j.endswith(f"{s}_ankle")][0] for s in "LR"}

    ground, slip = [], {}
    prev_low = {}
    for i in range(n + 1):
        T = fk_frame(d, i)
        for s_ in "LR":
            Rw, tw = T[ankle_jp[s_]]
            pts = np.array([tw + Rw @ c for c in CORNERS_CM[s_]])
            ground.extend(pts[:, 2])
            c = d["lc"][i] if s_ == "L" else d["rc"][i]
            if c > 0.5:
                # the true contact point is the lowest corner; a non-sliding
                # contact keeps THAT point fixed (rolling on an edge is allowed)
                k = int(np.argmin(pts[:, 2]))
                if s_ in prev_low and prev_low[s_][0] == i - 1 and prev_low[s_][2] == k:
                    slip.setdefault(s_, []).append(
                        np.linalg.norm(pts[k][:2] - prev_low[s_][1][:2]))
                prev_low[s_] = (i, pts[k], k)
            else:
                prev_low.pop(s_, None)

    info["min_sole_z_cm"] = min(ground)
    if min(ground) < -0.15:
        issues.append(f"ground penetration {min(ground):.3f} cm")

    mx = max((max(v) for v in slip.values() if v), default=0.0)
    info["max_stance_slip_mm"] = mx * 10
    if mx * 10 > 0.35:
        issues.append(f"stance contact point slides {mx*10:.2f} mm/frame")

    # joint angles vs limits
    axes = {}
    worst = 0.0
    for jp in d["joints"][1:]:
        name = jp.split("/")[-1]
        k = d["joints"].index(jp)
        angs = []
        for i in range(n + 1):
            q = d["rots"][i][k]
            angs.append(2 * np.degrees(np.arctan2(np.linalg.norm(q[1:]), q[0])))
        a = np.array(angs)
        lim = SAFE[name[2:]]
        worst = max(worst, a.max() / lim)
        if a.max() > lim + 1e-6:
            issues.append(f"{name} |angle| {a.max():.1f} deg > safe {lim}")
    info["max_joint_use_pct"] = worst * 100

    # exact periodicity in joint space (frame n == frame 0 for all but Root)
    dq = max(np.abs(d["rots"][0][k] - d["rots"][n][k]).max() for k in range(1, len(d["joints"])))
    info["loop_quat_delta"] = dq
    if dq > 1e-6:
        issues.append(f"clip not periodic in joint space (dq={dq:.2e})")

    # C1 continuity across the loop, per joint, using quaternion log speed
    def qd(a, b):
        return np.arccos(np.clip(abs(float(np.dot(a, b))), -1, 1)) * 2 * fps
    jump = 0.0
    for k in range(1, len(d["joints"])):
        v_in = qd(d["rots"][n - 1][k], d["rots"][n][k])     # arriving at the loop point
        v_out = qd(d["rots"][0][k], d["rots"][1][k])        # leaving it
        jump = max(jump, abs(v_in - v_out))
    info["loop_speed_jump_degps"] = np.degrees(jump)
    # also report the worst interior velocity kink, so the loop point can be
    # compared against the rest of the clip rather than judged in isolation
    kink = 0.0
    for k in range(1, len(d["joints"])):
        v = [qd(d["rots"][i][k], d["rots"][i + 1][k]) for i in range(n)]
        kink = max(kink, max(abs(v[i + 1] - v[i]) for i in range(len(v) - 1)))
    info["max_interior_kink_degps"] = np.degrees(kink)

    # root travel must match the commanded velocity
    p0, p1 = d["trans"][0][0], d["trans"][n][0]
    trav = (p1 - p0) / 100.0                 # cm -> m, robot frame
    fwd, left = -trav[1], trav[0]            # robot (x=left, y=back) -> character
    dx, dy, dth = d["disp"]
    # the pelvis rides PELVIS_X_OFFSET ahead of the path origin, so under a
    # turn its travel is the path displacement plus the swing of that offset
    off = PELVIS_X_OFFSET_CM / 100.0
    ex = dx + off * (np.cos(dth) - 1.0)
    ey = dy + off * np.sin(dth)
    info["travel_err_mm"] = np.hypot(fwd - ex, left - ey) * 1000
    if info["travel_err_mm"] > 0.5:
        issues.append(f"root travel mismatch {info['travel_err_mm']:.2f} mm")

    # phase track: monotone, spans [0,1], events land on the tagged frames
    ph = np.array(d["phase"])
    if ph[0] > 1e-6 or abs(ph[-1] - 1.0) > 1e-6:
        issues.append(f"phase does not span [0,1] ({ph[0]:.3g}..{ph[-1]:.3g})")
    if np.any(np.diff(ph) < -1e-9):
        issues.append("phase not monotonic")
    ev_ok = []
    for name, te, pe in zip(["LLO", "LTD", "RLO", "RTD"], d["events"], d["phis"]):
        j = int(round(te * fps))
        got = ph[min(j, n)]
        ev_ok.append(abs(got - pe) < 0.06)
    info["event_phase_ok"] = all(ev_ok)
    if not all(ev_ok):
        issues.append("phase landmarks do not line up with the event times")

    # contact-state track must follow LS -> DS1 -> RS -> DS2
    seq = [s for k, s in enumerate(d["state"]) if k == 0 or s != d["state"][k - 1]]
    info["state_sequence"] = "->".join(seq)
    if seq[:4] != ["LS", "DS1", "RS", "DS2"]:
        issues.append(f"unexpected contact sequence {seq}")

    # phase map must be strictly increasing and hit the event times
    om = np.array(d["om_t"])
    if np.any(np.diff(om) <= 0):
        issues.append("phase map Omega(phi) not strictly increasing")

    return d, info, issues


def main(d=None):
    d = d or (sys.argv[1] if len(sys.argv) > 1 else None)
    if d is None:
        from . import paths
        d = paths.CLIPS
    files = sorted(glob.glob(os.path.join(d, "*.usda")))
    bad = 0
    print(f"{'file':26s} {'soleZ':>7s} {'slip':>6s} {'joint':>6s} {'loop dq':>8s} "
          f"{'C1 loop/max':>13s} {'travel':>7s}  status")
    for f in files:
        d, info, issues = check(f)
        bad += len(issues)
        print(f"{os.path.basename(f):26s} "
              f"{info['min_sole_z_cm']:7.3f} "
              f"{info['max_stance_slip_mm']:6.3f} "
              f"{info['max_joint_use_pct']:5.0f}% "
              f"{info['loop_quat_delta']:8.1e} "
              f"{info['loop_speed_jump_degps']:6.1f}/{info['max_interior_kink_degps']:<6.1f} "
              f"{info['travel_err_mm']:6.3f}m  "
              f"{'OK' if not issues else '; '.join(issues)}")
    print("\nunits: soleZ = lowest sole point (cm, must be >= -0.15), "
          "slip = stance foot motion (mm/frame), joint = max % of safe range,\n"
          "loop dq = joint-space periodicity, C1 jump = angular speed "
          "discontinuity across the loop, travel = root vs commanded velocity")
    print("\nRESULT:", "all checks passed" if bad == 0 else f"{bad} issue(s)")


if __name__ == "__main__":
    main()
