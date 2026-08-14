"""
export_params.py -- dump the nine gait samples as raw parameters (Tab. 1 time
intervals, Sec. 4.3 attributes, and the control points of every Tab. 2 function
curve) plus the robot kinematics, so the browser can do real PARAMETER blending:
interpolate Pi, then re-run the planner and the IK.

This is the input the pose-blending viewer never needed -- it only had baked
joint angles.
"""

import os, json
import numpy as np

from stylized_gait import paths

from stylized_gait import core as GC
from stylized_gait import planner as GP
from stylized_gait import samples as SM
from stylized_gait import overrides as GO

HERE = paths.DATA


def spline(s):
    return {"t": [round(float(x), 8) for x in s.t],
            "v": [round(float(x), 8) for x in s.v],
            "m": [round(float(x), 8) for x in s.m],
            "periodic": bool(s.periodic)}


def dump_sample(s, args=None):
    d = {
        # the GaitSample constructor arguments, so the editor can rebuild every
        # derived curve exactly the way the generator does
        "args": {k: float(v) for k, v in (args or {}).items()
                 if k in GO.ATTR_KEYS},
        "name": s.name,
        "vel": [float(x) for x in s.vel],
        # Tab. 1 time intervals
        "T": float(s.T), "t_llo": float(s.t_llo), "t_ltd": float(s.t_ltd),
        "t_rlo": float(s.t_rlo), "t_rtd": float(s.t_rtd),
        # Sec. 4.3 attributes
        "stance_width": float(s.stance_width),
        "stance_x_bias": float(s.stance_x_bias),
        "stance_splay": float(s.stance_splay),
        "dt_psi_lo": float(s.dt_psi_lo), "dt_psi_td": float(s.dt_psi_td),
        "psi_toe_lo": float(s.psi_toe_lo), "psi_heel_td": float(s.psi_heel_td),
        "pelvis_z_nom": float(s.pelvis_z_nom),
        "notes": s.notes,
        # Tab. 2 function curves
        "curves": {},
    }
    for nm, per_side in (("foot_x", s.c_foot_x), ("foot_y", s.c_foot_y),
                         ("foot_z", s.c_foot_z), ("foot_yaw", s.c_foot_yaw),
                         ("foot_gamma", s.c_foot_gamma),
                         ("psi_toe", s.c_psi_toe), ("psi_heel", s.c_psi_heel)):
        d["curves"][nm] = {k: spline(per_side[k]) for k in "LR"}
    for nm, c in (("pelvis_z", s.c_pelvis_z), ("pelvis_roll", s.c_pelvis_roll),
                  ("pelvis_pitch", s.c_pelvis_pitch), ("pelvis_yaw", s.c_pelvis_yaw)):
        d["curves"][nm] = spline(c)
    return d


def main(overrides=None):
    g = SM.build()
    if overrides:
        g = GO.apply(g, overrides)
    args = GO._authored_args()
    out = {
        "samples": [dump_sample(g[n], args[n]) for n in SM.ORDER],
        "phi": list(GC.PHI_LANDMARKS),
        "keyvel": {"xmax": SM.X_MAX, "xslow": SM.X_SLOW,
                   "ymax": SM.Y_MAX, "thmax": SM.TH_MAX},
        "kin": {
            "chain": GC.CHAIN,
            "legs": {
                s: {
                    "anchor": {j: [float(x) for x in GC.anchors(s)[j]] for j in GC.CHAIN},
                    "local": {j: [float(x) for x in GP.LEGS[s].local[j]] for j in GC.CHAIN},
                    "axis": {j: [float(x) for x in GC.axes(s)[j]] for j in GC.CHAIN},
                    "limits": [[float(np.deg2rad(v)) for v in GC.SAFE_LIMITS_DEG[j]]
                               for j in GC.CHAIN],
                    "corners": [[float(x) for x in c] for c in GC.SOLE_CORNERS[s]],
                } for s in "LR"
            },
            "ankle_in_foot": [float(x) for x in GC.ANKLE_IN_FOOT],
            "heel_in_foot": [float(x) for x in GC.HEEL_IN_FOOT],
            "toe_in_foot": [float(x) for x in GC.TOE_IN_FOOT],
            "pelvis_x_offset": float(GC.PELVIS_X_OFFSET),
            "stance_x_neutral": float(GC.STANCE_X_NEUTRAL),
            "nominal_height": float(GC.NOMINAL_HEIGHT),
        },
    }
    p = paths.PARAMS_JSON
    open(p, "w").write(json.dumps(out, separators=(",", ":")))
    print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB, {len(out['samples'])} samples)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
