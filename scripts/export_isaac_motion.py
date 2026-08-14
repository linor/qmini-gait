"""
export_isaac_motion.py -- turn a generated .usda clip into an .npz that an
Isaac Lab environment can play back directly.

Two conversions matter here.

1. SIGN.  A UsdPhysics revolute joint measures the rotation of body1 relative
   to body0.  On this robot the right leg (and the left ankle and yaw) have
   body0/body1 swapped relative to the parent->child direction, so PhysX reads
   those joints with the opposite sign to the skeleton convention used when the
   clips were generated.  The mapping is derived from the physics graph, not
   assumed -- and it is also the explanation for the "mirrored" joint limits:
   once the sign is undone, roll/pitch/knee/ankle limits are IDENTICAL on both
   legs and only yaw is mirrored, which is correct for yaw.

2. FRAME.  The clips are authored in a character frame (x forward, y left,
   z up).  The exported root pose is expressed in the robot's own USD axes
   (x left, y backward, z up) so that an identity quaternion is the model's
   rest orientation, which is what Isaac expects for this asset.
"""

import os, re, sys, collections, json
import numpy as np

from stylized_gait import paths

from stylized_gait import paths, verify as V
HERE = paths.DATA

SRC_USD = paths.ROBOT_USD

SKEL2PHYS = {
    "L_hip_yaw": "Revolute_left_yaw",   "L_hip_roll": "Revolute_left_roll",
    "L_hip_pitch": "Revolute_left_pitch", "L_knee": "Revolute_left_knee",
    "L_ankle": "Revolute_left_ankle",
    "R_hip_yaw": "Revolute_right_yaw",  "R_hip_roll": "Revolute_right_roll",
    "R_hip_pitch": "Revolute_right_pitch", "R_knee": "Revolute_right_knee",
    "R_ankle": "Revolute_right_ankle",
}
ORDER = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
         "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"]
REF_AXIS = {"hip_yaw": [0, 0, 1], "hip_roll_L": [0, -1, 0], "hip_roll_R": [0, 1, 0],
            "hip_pitch": [-1, 0, 0], "knee": [-1, 0, 0], "ankle": [-1, 0, 0]}
C2R = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])     # char -> robot axes


# ---------------------------------------------------------------- physics graph
def joint_signs_and_limits(src_path=SRC_USD):
    src = open(src_path).read()

    def block(t, s):
        i = t.index("{", s); d = 0; j = i
        while True:
            if t[j] == "{": d += 1
            elif t[j] == "}":
                d -= 1
                if d == 0: return t[i + 1:j]
            j += 1

    fixed, rev = [], {}
    for m in re.finditer(r'def Physics(Fixed|Revolute)Joint "(\w+)"', src):
        kind, name = m.group(1), m.group(2)
        b = block(src, m.end())
        b0 = re.search(r"body0 = <([^>]*)>", b).group(1).split("/")[-1]
        b1 = re.search(r"body1 = <([^>]*)>", b).group(1).split("/")[-1]
        if "qmini_urdf_2legs" in (b0, b1):
            continue
        if kind == "Fixed":
            fixed.append((b0, b1))
        else:
            rev[name] = (b0, b1,
                         float(re.search(r"lowerLimit = ([-\d.e+]+)", b).group(1)),
                         float(re.search(r"upperLimit = ([-\d.e+]+)", b).group(1)))

    par = {}
    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: par[ra] = rb
    for a, b in fixed: uni(a, b)
    for b0, b1, _, _ in rev.values(): find(b0); find(b1)

    adj = collections.defaultdict(list)
    for n, (b0, b1, _, _) in rev.items():
        adj[find(b0)].append(find(b1)); adj[find(b1)].append(find(b0))
    depth, q = {find("base_link"): 0}, collections.deque([find("base_link")])
    while q:
        c = q.popleft()
        for nb in adj[c]:
            if nb not in depth:
                depth[nb] = depth[c] + 1; q.append(nb)

    signs, limits = {}, {}
    for n, (b0, b1, lo, hi) in rev.items():
        # q rotates body1 relative to body0; +1 when body1 is the child
        signs[n] = 1 if depth[find(b1)] > depth[find(b0)] else -1
        limits[n] = (np.deg2rad(lo), np.deg2rad(hi))
    return signs, limits


# ---------------------------------------------------------------- clip reading
def clip_joint_angles(usda):
    """signed joint angles in the SKELETON convention (child relative to parent
    about the extracted world axis), shape (n+1, 10)"""
    d = V.parse(usda)
    n = len(d["rots"]) - 1
    idx = {jp.split("/")[-1]: k for k, jp in enumerate(d["joints"])}
    out = np.zeros((n + 1, len(ORDER)))
    for c, nm in enumerate(ORDER):
        k = idx[nm]
        jn, side = nm[2:], nm[0]
        ref = np.array(REF_AXIS[f"hip_roll_{side}"] if jn == "hip_roll" else REF_AXIS[jn])
        for i in range(n + 1):
            q = np.asarray(d["rots"][i][k]); v = q[1:]; l = np.linalg.norm(v)
            a = 2 * np.arctan2(l, q[0])
            if a > np.pi: a -= 2 * np.pi
            if l > 1e-12 and np.dot(v / l, ref) < 0: a = -a
            out[i, c] = a
    return d, out


def main(clip="03_step_in_place.usda", out_name=None):
    signs, limits = joint_signs_and_limits()
    d, q_skel = clip_joint_angles(os.path.join(paths.CLIPS, clip))
    n = q_skel.shape[0] - 1
    fps = d["fps"]

    sign_vec = np.array([signs[SKEL2PHYS[nm]] for nm in ORDER], float)
    q_phys = q_skel * sign_vec

    names = [SKEL2PHYS[nm] for nm in ORDER]
    lo = np.array([limits[nm][0] for nm in names])
    hi = np.array([limits[nm][1] for nm in names])

    print(f"clip {clip}   T = {d['T']} s   {n} frames @ {fps} fps")
    print(f"{'physics joint':24s} {'sign':>4s} {'ref min':>8s} {'ref max':>8s}"
          f" {'raw limit lo':>13s} {'raw limit hi':>13s}   fits?")
    allfit = True
    for c, nm in enumerate(names):
        a = np.rad2deg(q_phys[:, c])
        L, H = np.rad2deg(lo[c]), np.rad2deg(hi[c])
        fit = a.min() >= L - 1e-6 and a.max() <= H + 1e-6
        allfit &= fit
        print(f"  {nm:22s} {int(sign_vec[c]):+4d} {a.min():8.2f} {a.max():8.2f}"
              f" {L:13.1f} {H:13.1f}   {'yes' if fit else 'NO'}")
    print("every joint inside its own authored limit:", allfit)

    # ---- root pose, expressed in the robot's USD axes ----
    root_p = np.zeros((n + 1, 3))
    root_q = np.zeros((n + 1, 4))
    for i in range(n + 1):
        pc = np.array(d["trans"][i][0]) / 100.0        # cm -> m, already robot axes
        Rr = V.q2m(d["rots"][i][0])                    # robot-axis rotation
        root_p[i] = pc
        root_q[i] = mat2quat(Rr)
    root_p -= np.array([root_p[0, 0], root_p[0, 1], 0.0])   # start over the origin

    # Structural invariant: for a left/right symmetric gait the PhysX joint
    # values obey q_right(t + T/2) = -q_left(t).  Recording it lets the runtime
    # confirm that the sign map and the column ordering survived loading.
    half = (n + 1) // 2
    anti = 0.0
    for j in ["yaw", "roll", "pitch", "knee", "ankle"]:
        li = names.index(f"Revolute_left_{j}")
        ri = names.index(f"Revolute_right_{j}")
        shifted = np.roll(q_phys[:-1, ri], -half)
        anti = max(anti, np.abs(shifted + q_phys[:-1, li]).max())
    anti_deg = float(np.rad2deg(anti))
    print(f"half-cycle antisymmetry residual: {anti_deg:.4f} deg "
          f"({'symmetric gait' if anti_deg < 0.05 else 'not a symmetric gait'})")

    phase = np.array(d["phase"], float)
    contact = np.stack([np.array(d["lc"], float), np.array(d["rc"], float)], axis=1)
    times = np.arange(n + 1) / fps

    out_name = out_name or ("qmini_" + clip[3:].replace(".usda", "") + ".npz")
    path = os.path.join(HERE, out_name)
    np.savez_compressed(
        path,
        joint_names=np.array(names),
        joint_pos=q_phys.astype(np.float32),
        joint_pos_lower=lo.astype(np.float32),
        joint_pos_upper=hi.astype(np.float32),
        joint_sign=sign_vec.astype(np.float32),
        root_pos=root_p.astype(np.float32),
        root_quat=root_q.astype(np.float32),
        phase=phase.astype(np.float32),
        contact=contact.astype(np.float32),
        times=times.astype(np.float32),
        period=np.float32(d["T"]),
        antisymmetry_deg=np.float32(anti_deg),
        fps=np.int32(fps),
        source=np.array(clip),
        convention=np.array(
            "joint_pos follows the UsdPhysics/PhysX sign (body1 relative to "
            "body0); root_pos/root_quat are in the robot's USD axes "
            "(x left, y backward, z up), root_pos starts over the origin"),
    )
    print(f"\nwrote {path}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


def mat2quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
