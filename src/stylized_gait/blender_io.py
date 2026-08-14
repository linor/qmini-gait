"""
blender_io.py -- Python gait generator  <->  Blender, over USD.

    samples.py ──► export_edit_usd()  ──►  gait_edit/<sample>.usda
                                              │  open in Blender, tune, export .usda
                                              ▼
                   build_library()  ◄──  gait_edit/<sample>.usda
                          │
                          ▼
                   gait_library.json  ──►  parameter blending, clips, Isaac export
                                            (never touches USD again)

WHAT TRAVELS AND WHY
--------------------
The Tab. 2 function curves become transform animation on a small rig, because
that is the only thing a DCC round-trips reliably.  Every scalar that merely
*constructs* one of those curves (swing_height, pelvis_bounce, foot_yaw_amp,
the interval fractions, ...) is therefore redundant and is deliberately left
out.  Three more are derived on import rather than carried, so an edit to the
curve cannot disagree with a stale scalar:

    psi_toe_lo   = psi_toe  curve at the liftoff end of the swing interval
    psi_heel_td  = psi_heel curve at the touchdown end
    pelvis_z_nom = folded into the pelvis z curve, which is written absolute

What is left is exactly what the planner cannot recover from the curves: the
key velocity, the four contact event times and the cycle period, the three
stance attributes, and the two heel/toe support ramp durations.

THE RIG
-------
    /Gait_<name>/Rig/Pelvis                 translate (0,0,height) + rotateXYZ
    /Gait_<name>/Rig/{Left,Right}Foot       swing offset translate + rotateXYZ
                       /HeelPitch           rotateXYZ about the heel point
                              /ToePitch     rotateXYZ about the toe point
    /Gait_<name>/Reference                  skinned robot, for looking at only

The heel/toe pair is the paper's Fig. 5 reverse-foot-lock rig, split into two
nested pivots so an animator can grab them.  Rig children are authored in the
character frame (x forward, y left, z up, centimetres); the Rig prim itself
carries the transform onto the robot's own axes so everything lines up.

BACKENDS
--------
Reading is done with ``pxr`` when it is importable, which is what makes the
round-trip robust: ``UsdGeom.Xformable.GetLocalTransformation`` composes
whatever xform ops Blender wrote -- translate + rotateXYZ, a baked
``xformOp:transform``, an ``orient`` quaternion, any order -- into one matrix
that we simply decompose.  It also reads binary ``.usdc``, which is Blender's
default export format.

Without pxr there is a hand-written ASCII fallback (``usda_min``).  It only
reads ``.usda`` and only the op forms listed above.  ``blender_io.py selftest``
compares the two backends on the same file when both are present.
"""

import json
import os

import numpy as np

from . import core as GC
from . import planner as GP
from . import usda_min as U

try:
    from pxr import Usd, UsdGeom, UsdSkel, Gf, Sdf, Vt      # noqa: F401
    HAVE_PXR = True
except ImportError:                                        # pragma: no cover
    HAVE_PXR = False

from . import paths
HERE = paths.DATA
CM = 100.0
C2R = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])       # char -> robot axes
DEG = 180.0 / np.pi

FOOT_CURVES = ("foot_x", "foot_y", "foot_z", "foot_yaw", "foot_gamma",
               "psi_toe", "psi_heel")
PELVIS_CURVES = ("pelvis_z", "pelvis_roll", "pelvis_pitch", "pelvis_yaw")


# ============================================================== export
def _rig_xform_matrix():
    m = np.eye(4)
    m[:3, :3] = C2R.T          # row-vector convention
    return m


def _foot_channels(s, side, times):
    """the seven swing curves sampled on a time grid"""
    c = {}
    for nm in FOOT_CURVES:
        f = getattr(s, "c_" + nm)[side]
        c[nm] = np.array([float(f(t)) for t in times])
    return c


def export_edit_usd(sample, path, mesh=None, fps=60):
    """write one Blender-editable .usda for a single gait sample"""
    s = sample
    n = int(round(s.T * fps))
    times = np.arange(n + 1) / fps
    key = lambda i: i                                   # timeCode == frame

    L = []
    A = L.append
    A("#usda 1.0")
    A("(")
    A(f'    defaultPrim = "Gait"')
    A("    metersPerUnit = 0.01")
    A('    upAxis = "Z"')
    A(f"    timeCodesPerSecond = {fps}")
    A(f"    framesPerSecond = {fps}")
    A("    startTimeCode = 0")
    A(f"    endTimeCode = {n}")
    A("    customLayerData = {")
    A('        string generator = "stylized_gait blender_io.export_edit_usd"')
    A('        string editable = "everything under /Gait/Rig; /Gait/Reference is display only"')
    A("    }")
    A(")")
    A("")
    A('def Xform "Gait"')
    A("{")
    A("    # ---- the only scalars that cannot be recovered from the curves ----")
    A(f'    custom string gait:sample = "{s.name}"')
    A(f"    custom double3 gait:velocity = {U.fmt(np.asarray(s.vel))}")
    A(f"    custom double gait:cyclePeriod = {s.T:.9g}")
    A("    custom double[] gait:eventTimes = ["
      + ", ".join(f"{v:.9g}" for v in (s.t_llo, s.t_ltd, s.t_rlo, s.t_rtd)) + "]")
    A('    custom token[] gait:eventNames = ["LLO", "LTD", "RLO", "RTD"]')
    A(f"    custom double gait:stanceWidth = {s.stance_width:.9g}")
    A(f"    custom double gait:stanceXBias = {s.stance_x_bias:.9g}")
    A(f"    custom double gait:stanceSplay = {s.stance_splay:.9g}")
    A(f"    custom double gait:dtPsiLO = {s.dt_psi_lo:.9g}")
    A(f"    custom double gait:dtPsiTD = {s.dt_psi_td:.9g}")
    A(f"    custom int gait:fps = {fps}")
    A(f'    custom string gait:units = "translations in cm, rotations in degrees, '
      f'character frame x fwd / y left / z up"')
    A("")
    A('    def Xform "Rig"')
    A("    {")
    A(f"        matrix4d xformOp:transform = {U.fmt(_rig_xform_matrix())}")
    A('        uniform token[] xformOpOrder = ["xformOp:transform"]')
    A("")

    # ---- pelvis ----
    pz = {key(i): np.array([0.0, 0.0,
                            (s.pelvis_z_nom + float(s.c_pelvis_z(t))) * CM])
          for i, t in enumerate(times)}
    pr = {key(i): np.array([float(s.c_pelvis_roll(t)) * DEG,
                            float(s.c_pelvis_pitch(t)) * DEG,
                            float(s.c_pelvis_yaw(t)) * DEG])
          for i, t in enumerate(times)}
    A('        def Xform "Pelvis"')
    A("        {")
    A(U.time_samples("xformOp:translate", "double3", pz, 12))
    A(U.time_samples("xformOp:rotateXYZ", "float3", pr, 12))
    A('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
    A("        }")

    # ---- feet ----
    heel = GC.HEEL_IN_FOOT * CM
    toe = GC.TOE_IN_FOOT * CM
    for side, nm in (("L", "LeftFoot"), ("R", "RightFoot")):
        c = _foot_channels(s, side, times)
        tr = {key(i): np.array([c["foot_x"][i], c["foot_y"][i], c["foot_z"][i]]) * CM
              for i in range(len(times))}
        ro = {key(i): np.array([c["foot_gamma"][i] * DEG, 0.0,
                                c["foot_yaw"][i] * DEG])
              for i in range(len(times))}
        hp = {key(i): np.array([0.0, c["psi_heel"][i] * DEG, 0.0])
              for i in range(len(times))}
        tp = {key(i): np.array([0.0, c["psi_toe"][i] * DEG, 0.0])
              for i in range(len(times))}
        A(f'        def Xform "{nm}"')
        A("        {")
        A(U.time_samples("xformOp:translate", "double3", tr, 12))
        A(U.time_samples("xformOp:rotateXYZ", "float3", ro, 12))
        A('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
        A("")
        A('            def Xform "HeelPitch"')
        A("            {")
        A(f"                double3 xformOp:translate = {U.fmt(heel)}")
        A(U.time_samples("xformOp:rotateXYZ", "float3", hp, 16))
        A('                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
        A("")
        A('                def Xform "ToePitch"')
        A("                {")
        A(f"                    double3 xformOp:translate = {U.fmt(toe - heel)}")
        A(U.time_samples("xformOp:rotateXYZ", "float3", tp, 20))
        A('                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
        A("                }")
        A("            }")
        A("        }")
    A("    }")           # /Rig

    if mesh is not None:
        A("")
        A(_reference_block(sample, mesh, fps))
    A("}")
    open(path, "w").write("\n".join(L) + "\n")
    return path


def _reference_block(sample, mesh, fps):
    """the baked, skinned robot -- display only"""
    from . import usd_writer as UW
    res = GP.solve_sample(sample, fps=fps, passes=3)
    binds, rests = UW.skeleton_tables()
    n = res["n"]

    pos = np.frombuffer(__import__("base64").b64decode(mesh["pos"]),
                        dtype=np.float32).reshape(-1, 3).astype(float)
    idx = np.frombuffer(__import__("base64").b64decode(mesh["idx"]),
                        dtype=np.uint32).reshape(-1, 3).astype(int)

    anchors_cm = {}
    for jname in UW.JOINT_NAMES:
        if jname == "Root":
            anchors_cm[jname] = np.zeros(3)
        else:
            leg = GP.LEGS[jname[0]]
            anchors_cm[jname] = C2R @ leg.a[jname[2:]] * CM

    out = ['    def SkelRoot "Reference" (',
           '        prepend apiSchemas = ["SkelBindingAPI"]',
           "    )", "    {",
           '        custom string note = "display only -- regenerate to refresh"',
           '        rel skel:skeleton = </Gait/Reference/Skel>',
           '        rel skel:animationSource = </Gait/Reference/Skel/Anim>',
           '        def Skeleton "Skel" (',
           '            prepend apiSchemas = ["SkelBindingAPI"]',
           "        )", "        {",
           "            uniform token[] joints = ["
           + ", ".join(f'"{p}"' for p in UW.JOINT_PATHS) + "]",
           "            uniform matrix4d[] bindTransforms = ["
           + ", ".join(UW._fmt_m4(m) for m in binds) + "]",
           "            uniform matrix4d[] restTransforms = ["
           + ", ".join(UW._fmt_m4(m) for m in rests) + "]",
           "            rel skel:animationSource = </Gait/Reference/Skel/Anim>",
           '            def SkelAnimation "Anim"', "            {",
           "                uniform token[] joints = ["
           + ", ".join(f'"{p}"' for p in UW.JOINT_PATHS) + "]",
           "                float3[] scales = ["
           + ", ".join("(1, 1, 1)" for _ in UW.JOINT_PATHS) + "]"]
    tr, ro = [], []
    for i in range(n + 1):
        src = i % n
        cyc = i // n
        p_c = res["pelvis_p"][src].copy()
        R_c = res["pelvis_R"][src]
        if cyc:
            d = res["cycle_displacement"]
            th = d[2] * cyc
            cs, sn = np.cos(th), np.sin(th)
            p_c = np.array([cs * p_c[0] - sn * p_c[1],
                            sn * p_c[0] + cs * p_c[1], p_c[2]]) + np.array([d[0], d[1], 0])
            R_c = GC.rot_z(th) @ R_c
        qs = UW.sample_rotations(res, src)
        qs[0] = GC.mat_to_quat(C2R @ R_c @ C2R.T)
        vals = [C2R @ p_c * CM] + [np.array(m[3, :3]) for m in rests[1:]]
        tr.append(f"                    {i}: ["
                  + ", ".join(UW._fmt_v3(v) for v in vals) + "],")
        ro.append(f"                    {i}: ["
                  + ", ".join(UW._fmt_q(q) for q in qs) + "],")
    out += ["                float3[] translations.timeSamples = {"] + tr + ["                }"]
    out += ["                quatf[] rotations.timeSamples = {"] + ro + ["                }"]
    out += ["            }", "        }"]

    off = 0
    for p in mesh["parts"]:
        j = UW.JOINT_NAMES.index(p["joint"])
        sl = slice(p["v0"], p["v0"] + p["nv"])
        V = pos[sl] + anchors_cm[p["joint"]]
        F = idx[off:off + p["nf"]] - p["v0"]
        off += p["nf"]
        nm = p["body"].replace("-", "_")
        out += [f'        def Mesh "{nm}" (',
                '            prepend apiSchemas = ["SkelBindingAPI"]',
                "        )", "        {",
                "            uniform token subdivisionScheme = \"none\"",
                "            int[] faceVertexCounts = [" + ", ".join(["3"] * len(F)) + "]",
                "            int[] faceVertexIndices = ["
                + ", ".join(str(int(v)) for v in F.reshape(-1)) + "]",
                "            point3f[] points = ["
                + ", ".join(UW._fmt_v3(v) for v in V) + "]",
                "            rel skel:skeleton = </Gait/Reference/Skel>",
                "            int[] primvars:skel:jointIndices = ["
                + ", ".join([str(j)] * len(V)) + "] (elementSize = 1 interpolation = \"vertex\")",
                "            float[] primvars:skel:jointWeights = ["
                + ", ".join(["1"] * len(V)) + "] (elementSize = 1 interpolation = \"vertex\")",
                "        }"]
    out.append("    }")
    return "\n".join(out)


# ============================================================== import
def _xform_channels(prim):
    """(translate, rotateXYZ) time-sample dicts, whatever ops the DCC wrote"""
    tr = prim.samples.get("xformOp:translate")
    ro = prim.samples.get("xformOp:rotateXYZ")
    if tr is None and "xformOp:translate" in prim.attrs:
        tr = {0.0: np.asarray(prim.attrs["xformOp:translate"], float)}
    if ro is None and "xformOp:rotateXYZ" in prim.attrs:
        ro = {0.0: np.asarray(prim.attrs["xformOp:rotateXYZ"], float)}
    mats = prim.samples.get("xformOp:transform")
    if mats:                                      # Blender often bakes a matrix
        tr, ro = {}, {}
        for t, m in mats.items():
            m = np.asarray(m, float)
            R = m[:3, :3].T                       # row-vector convention
            tr[t] = m[3, :3]
            ro[t] = np.degrees(_euler_xyz(R))
    return tr or {}, ro or {}


def _euler_xyz(R):
    """inverse of USD rotateXYZ, i.e. R = Rz(z) Ry(y) Rx(x) -> (x, y, z) rad"""
    sy = -R[2, 0]
    if abs(sy) < 1 - 1e-9:
        return np.array([np.arctan2(R[2, 1], R[2, 2]),
                         np.arcsin(np.clip(sy, -1, 1)),
                         np.arctan2(R[1, 0], R[0, 0])])
    return np.array([0.0, np.arcsin(np.clip(sy, -1, 1)),
                     np.arctan2(-R[0, 1], R[1, 1])])


def _series(d, comp, fps, scale=1.0):
    ts = sorted(d)
    return (np.array([t / fps for t in ts]),
            np.array([float(np.asarray(d[t], float)[comp]) * scale for t in ts]))


def _channels_pxr(stage, prim_path, fps):
    """(translate, rotateXYZ-in-degrees) per time sample, via the composed local
    transform.  Immune to which xform ops the DCC decided to write."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"missing prim {prim_path}")
    xf = UsdGeom.Xformable(prim)
    ts = xf.GetTimeSamples()
    if not ts:
        ts = [0.0]
    tr, ro = {}, {}
    for t in ts:
        m = np.array(xf.GetLocalTransformation(Usd.TimeCode(t)), float)
        tr[float(t)] = m[3, :3]
        ro[float(t)] = np.degrees(_euler_xyz(m[:3, :3].T))   # USD is row-vector
    return tr, ro


def _import_pxr(path):
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError(f"could not open {path}")
    gait = None
    for prim in stage.Traverse():
        if prim.GetAttribute("gait:sample"):
            gait = prim
            break
    if gait is None:
        raise ValueError(f"{path}: no prim carrying gait:sample")
    g = lambda n: gait.GetAttribute(n).Get()
    fps = int(g("gait:fps") or stage.GetTimeCodesPerSecond() or 60)
    ev = [float(x) for x in g("gait:eventTimes")]
    out = {
        "name": str(g("gait:sample")),
        "vel": [float(x) for x in g("gait:velocity")],
        "T": float(g("gait:cyclePeriod")),
        "events": ev,
        "stance_width": float(g("gait:stanceWidth")),
        "stance_x_bias": float(g("gait:stanceXBias")),
        "stance_splay": float(g("gait:stanceSplay")),
        "dt_psi_lo": float(g("gait:dtPsiLO")),
        "dt_psi_td": float(g("gait:dtPsiTD")),
        "fps": fps,
        "curves": {},
    }
    rig = gait.GetPath().AppendChild("Rig")
    ptr, pro = _channels_pxr(stage, rig.AppendChild("Pelvis"), fps)
    t, v = _series(ptr, 2, fps, 1 / CM)
    out["curves"]["pelvis_z"] = [t.tolist(), v.tolist()]
    for comp, nm in ((0, "pelvis_roll"), (1, "pelvis_pitch"), (2, "pelvis_yaw")):
        t, v = _series(pro, comp, fps, np.pi / 180.0)
        out["curves"][nm] = [t.tolist(), v.tolist()]
    for side, nm in (("L", "LeftFoot"), ("R", "RightFoot")):
        fp = rig.AppendChild(nm)
        ftr, fro = _channels_pxr(stage, fp, fps)
        for comp, cn in ((0, "foot_x"), (1, "foot_y"), (2, "foot_z")):
            t, v = _series(ftr, comp, fps, 1 / CM)
            out["curves"][f"{cn}.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(fro, 0, fps, np.pi / 180.0)
        out["curves"][f"foot_gamma.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(fro, 2, fps, np.pi / 180.0)
        out["curves"][f"foot_yaw.{side}"] = [t.tolist(), v.tolist()]
        hp = fp.AppendChild("HeelPitch")
        _, hro = _channels_pxr(stage, hp, fps)
        _, tro = _channels_pxr(stage, hp.AppendChild("ToePitch"), fps)
        t, v = _series(hro, 1, fps, np.pi / 180.0)
        out["curves"][f"psi_heel.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(tro, 1, fps, np.pi / 180.0)
        out["curves"][f"psi_toe.{side}"] = [t.tolist(), v.tolist()]
    return out


def import_edit_usd(path, backend="auto"):
    """read one edited USD layer back into a plain dict of Pi.

    backend: "auto" (pxr if importable), "pxr", or "ascii".
    """
    if backend == "pxr" or (backend == "auto" and HAVE_PXR):
        return _import_pxr(path)
    if path.endswith(".usdc") or path.endswith(".usd"):
        raise RuntimeError(
            f"{path}: binary/ambiguous layer needs pxr, which is not importable "
            f"here. Either install usd-core or export from Blender as .usda.")
    return _import_ascii(path)


def _import_ascii(path):
    """fallback reader -- .usda only, and only the op forms we write"""
    root, meta = U.parse(path)
    prims = {p.path: p for p in root.walk() if p.name}
    gait = next((p for p in prims.values()
                 if p.type == "Xform" and "gait:sample" in p.attrs), None)
    if gait is None:
        raise ValueError(f"{path}: no prim carrying gait:sample -- is this a gait edit file?")
    rig = next((p for p in gait.walk() if p.name == "Rig"), None)
    if rig is None:
        raise ValueError(f"{path}: no /Rig under {gait.path}")

    fps = int(gait.attrs.get("gait:fps", meta.get("framesPerSecond", 60)))
    ev = np.asarray(gait.attrs["gait:eventTimes"], float).ravel()
    out = {
        "name": gait.attrs["gait:sample"],
        "vel": list(np.asarray(gait.attrs["gait:velocity"], float)),
        "T": float(gait.attrs["gait:cyclePeriod"]),
        "events": [float(x) for x in ev],
        "stance_width": float(gait.attrs["gait:stanceWidth"]),
        "stance_x_bias": float(gait.attrs["gait:stanceXBias"]),
        "stance_splay": float(gait.attrs["gait:stanceSplay"]),
        "dt_psi_lo": float(gait.attrs["gait:dtPsiLO"]),
        "dt_psi_td": float(gait.attrs["gait:dtPsiTD"]),
        "fps": fps,
        "curves": {},
    }

    def need(name):
        p = next((q for q in rig.walk() if q.name == name), None)
        if p is None:
            raise ValueError(f"{path}: /Rig is missing '{name}'")
        return p

    pel = need("Pelvis")
    ptr, pro = _xform_channels(pel)
    t, v = _series(ptr, 2, fps, 1 / CM)
    out["curves"]["pelvis_z"] = [t.tolist(), v.tolist()]
    for comp, nm in ((0, "pelvis_roll"), (1, "pelvis_pitch"), (2, "pelvis_yaw")):
        t, v = _series(pro, comp, fps, np.pi / 180.0)
        out["curves"][nm] = [t.tolist(), v.tolist()]

    for side, nm in (("L", "LeftFoot"), ("R", "RightFoot")):
        foot = need(nm)
        ftr, fro = _xform_channels(foot)
        for comp, cn in ((0, "foot_x"), (1, "foot_y"), (2, "foot_z")):
            t, v = _series(ftr, comp, fps, 1 / CM)
            out["curves"][f"{cn}.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(fro, 0, fps, np.pi / 180.0)
        out["curves"][f"foot_gamma.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(fro, 2, fps, np.pi / 180.0)
        out["curves"][f"foot_yaw.{side}"] = [t.tolist(), v.tolist()]
        heel = need("HeelPitch") if side == "L" else None
        heel = next(q for q in foot.walk() if q.name == "HeelPitch")
        toe = next(q for q in heel.walk() if q.name == "ToePitch")
        _, hro = _xform_channels(heel)
        _, tro = _xform_channels(toe)
        t, v = _series(hro, 1, fps, np.pi / 180.0)
        out["curves"][f"psi_heel.{side}"] = [t.tolist(), v.tolist()]
        t, v = _series(tro, 1, fps, np.pi / 180.0)
        out["curves"][f"psi_toe.{side}"] = [t.tolist(), v.tolist()]
    return out


def build_library(layers, out_path=None, backend="auto", write=True):
    """compile edited USD layers into the blend-ready intermediate.

    Returns the library dict.  With ``write=True`` (the default) it is also
    written to ``out_path``, defaulting to data/gait_library.json.
    """
    lib = {"version": 1, "phi_landmarks": list(GC.PHI_LANDMARKS), "samples": []}
    for p in layers:
        lib["samples"].append(import_edit_usd(p, backend=backend))
    if write:
        out_path = out_path or paths.LIBRARY_JSON
        with open(out_path, "w") as f:
            json.dump(lib, f, separators=(",", ":"))
        lib["_path"] = out_path
    return lib


# ============================================================== pxr utilities
def _need_pxr(what):
    if not HAVE_PXR:
        raise RuntimeError(f"{what} needs pxr (pip install usd-core)")


def to_usdc(path, out=None):
    """re-save an ASCII layer as binary .usdc (Blender's default flavour)"""
    _need_pxr("to_usdc")
    out = out or os.path.splitext(path)[0] + ".usdc"
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError(f"could not open {path}")
    stage.Export(out)
    return out


def validate(path):
    """open a layer with pxr and check the things a hand-written writer can get
    wrong.  Returns (list of problems, dict of facts)."""
    _need_pxr("validate")
    problems, facts = [], {}
    stage = Usd.Stage.Open(path)
    if stage is None:
        return [f"{path}: Usd.Stage.Open returned None"], facts
    facts["upAxis"] = UsdGeom.GetStageUpAxis(stage)
    facts["metersPerUnit"] = UsdGeom.GetStageMetersPerUnit(stage)
    facts["timeCodes"] = (stage.GetStartTimeCode(), stage.GetEndTimeCode())
    facts["fps"] = stage.GetTimeCodesPerSecond()

    gait = next((p for p in stage.Traverse() if p.GetAttribute("gait:sample")), None)
    if gait is None:
        problems.append("no prim carrying gait:sample")
        return problems, facts
    facts["sample"] = gait.GetAttribute("gait:sample").Get()

    for rel in ("Rig/Pelvis", "Rig/LeftFoot", "Rig/RightFoot",
                "Rig/LeftFoot/HeelPitch", "Rig/LeftFoot/HeelPitch/ToePitch",
                "Rig/RightFoot/HeelPitch", "Rig/RightFoot/HeelPitch/ToePitch"):
        p2 = stage.GetPrimAtPath(gait.GetPath().AppendPath(rel))
        if not p2 or not p2.IsValid():
            problems.append(f"missing rig prim {rel}")
            continue
        xf = UsdGeom.Xformable(p2)
        ok, _ = xf.GetXformOpOrderAttr().Get() is not None, None
        n = len(xf.GetTimeSamples())
        facts[f"samples:{rel}"] = n
        try:
            xf.GetLocalTransformation(Usd.TimeCode(stage.GetStartTimeCode()))
        except Exception as e:
            problems.append(f"{rel}: local transform failed ({e})")

    ref = stage.GetPrimAtPath(gait.GetPath().AppendChild("Reference"))
    if ref and ref.IsValid():
        cache = UsdSkel.Cache()
        root = UsdSkel.Root(ref)
        if not root:
            problems.append("Reference is not a valid UsdSkelRoot")
        else:
            cache.Populate(root, Usd.TraverseInstanceProxies())
            binding = cache.ComputeSkelBinding(
                root, UsdSkel.Tokens.skelAnimationSource) if hasattr(
                    UsdSkel.Tokens, "skelAnimationSource") else None
            bindings = []
            try:
                bindings = cache.ComputeSkelBindings(
                    root, Usd.TraverseInstanceProxies())
            except Exception as e:
                problems.append(f"ComputeSkelBindings failed: {e}")
            facts["skelBindings"] = len(bindings)
            nskinned = sum(len(b.GetSkinningTargets()) for b in bindings)
            facts["skinnedPrims"] = nskinned
            if nskinned == 0:
                problems.append("no skinned meshes resolved under Reference")
            for b in bindings:
                q = b.GetSkeletonQuery()
                if q is None or not q:
                    problems.append("skeleton query did not resolve")
                    continue
                facts["skelJoints"] = len(q.GetJointOrder())
                if q.GetAnimQuery() is None or not q.GetAnimQuery():
                    problems.append("skeleton has no animation source")
    else:
        facts["skelBindings"] = "no Reference block"
    return problems, facts


def selftest(path):
    """compare the pxr and ASCII readers on the same file"""
    a = import_edit_usd(path, backend="ascii")
    if not HAVE_PXR:
        return None, a
    b = import_edit_usd(path, backend="pxr")
    worst, who = 0.0, ""
    for k in a["curves"]:
        va = np.asarray(a["curves"][k][1], float)
        vb = np.asarray(b["curves"][k][1], float)
        n = min(len(va), len(vb))
        d = float(np.abs(va[:n] - vb[:n]).max()) if n else 0.0
        if d > worst:
            worst, who = d, k
    for k in ("T", "stance_width", "stance_x_bias", "stance_splay",
              "dt_psi_lo", "dt_psi_td"):
        d = abs(float(a[k]) - float(b[k]))
        if d > worst:
            worst, who = d, k
    return (worst, who), a


# ============================================================== cli
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(
        description="gait generator <-> Blender, over USD")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="write Blender-editable .usda per sample")
    e.add_argument("--out", default=paths.BLENDER_EDIT)
    e.add_argument("--overrides", default=paths.OVERRIDES_JSON)
    e.add_argument("--fps", type=int, default=60)
    e.add_argument("--no-reference", action="store_true",
                   help="omit the skinned robot (much smaller files)")
    e.add_argument("samples", nargs="*", help="sample names (default: all)")

    i = sub.add_parser("import", help="compile edited USD into gait_library.json")
    i.add_argument("--dir", default=paths.BLENDER_EDIT)
    i.add_argument("--out", default=paths.LIBRARY_JSON)
    i.add_argument("--backend", choices=("auto", "pxr", "ascii"), default="auto")
    i.add_argument("files", nargs="*")

    v = sub.add_parser("validate", help="open with pxr and check the layer")
    v.add_argument("files", nargs="+")

    t = sub.add_parser("selftest", help="compare the pxr and ASCII readers")
    t.add_argument("files", nargs="+")

    c = sub.add_parser("usdc", help="re-save layers as binary .usdc")
    c.add_argument("files", nargs="+")

    a = ap.parse_args(argv)
    if a.cmd == "export":
        from . import samples as SM
        from . import overrides as GO
        g = GO.apply(SM.build(), a.overrides if os.path.isfile(a.overrides) else None)
        names = a.samples or SM.ORDER
        mesh = None
        if not a.no_reference:
            mesh = json.load(open(paths.MESH_JSON))
        os.makedirs(a.out, exist_ok=True)
        for n in names:
            p = export_edit_usd(g[n], os.path.join(a.out, f"{n}.usda"),
                                mesh=mesh, fps=a.fps)
            print(f"  {os.path.basename(p):28s} {os.path.getsize(p)/1024:7.0f} KB")
        print(f"open these in Blender; export back as .usda (not .usdc)")
    elif a.cmd == "validate":
        bad = 0
        for f in a.files:
            probs, facts = validate(f)
            print(f"{os.path.basename(f)}")
            for k, v2 in facts.items():
                print(f"    {k:24s} {v2}")
            for pmsg in probs:
                print(f"    PROBLEM  {pmsg}")
            bad += len(probs)
        raise SystemExit(1 if bad else 0)
    elif a.cmd == "selftest":
        if not HAVE_PXR:
            raise SystemExit("selftest needs pxr")
        worst_all = 0.0
        for f in a.files:
            (d, who), _ = selftest(f)
            worst_all = max(worst_all, d)
            print(f"  {os.path.basename(f):28s} max pxr-vs-ascii diff {d:.3e}  ({who})")
        print("backends agree" if worst_all < 1e-6 else "BACKENDS DISAGREE")
        raise SystemExit(0 if worst_all < 1e-6 else 1)
    elif a.cmd == "usdc":
        for f in a.files:
            print("  ", to_usdc(f))
    else:
        exts = (".usda", ".usdc", ".usd") if HAVE_PXR else (".usda",)
        files = a.files or sorted(
            os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith(exts))
        if not files:
            raise SystemExit(f"no .usda files in {a.dir}")
        lib = build_library(files, a.out, backend=a.backend)
        out = lib["_path"]
        print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB, "
              f"{len(lib['samples'])} samples)")
        for s in lib["samples"]:
            print(f"  {s['name']:16s} T={s['T']:.3f}s  {len(s['curves'])} curves")


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
