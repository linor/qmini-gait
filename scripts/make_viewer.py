"""
make_viewer.py -- bake the nine .usda clips into one standalone HTML viewer.

No server, no CDN, no dependencies: the clip data is embedded as JSON and the
page does its own forward kinematics and software 3D projection on a 2D canvas.
Open gait_viewer.html directly in a browser.
"""

import os, glob, json
import numpy as np

from stylized_gait import paths


def R(name):
    return open(_asset(name)).read()


def _asset(name):
    """web sources live in web/, generated data in data/"""
    for d in (paths.WEB, paths.DATA, paths.DIST):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name} not found in web/ or data/")
from stylized_gait import verify as V

HERE = paths.DATA

# reference axis per joint, robot frame, used to give the quaternion a signed angle
REF_AXIS = {
    "hip_yaw": [0, 0, 1], "hip_roll_L": [0, -1, 0], "hip_roll_R": [0, 1, 0],
    "hip_pitch": [-1, 0, 0], "knee": [-1, 0, 0], "ankle": [-1, 0, 0],
}
SAFE = {"hip_yaw": 30, "hip_roll": 15, "hip_pitch": 50, "knee": 50, "ankle": 40}


def ref_axis(name):
    side, jn = name[0], name[2:]
    return REF_AXIS[f"hip_roll_{side}"] if jn == "hip_roll" else REF_AXIS[jn]


def bake(path):
    d = V.parse(path)
    n = len(d["rots"]) - 1
    joints = []
    for k, jp in enumerate(d["joints"]):
        name = jp.split("/")[-1]
        parent = "/".join(jp.split("/")[:-1])
        joints.append({
            "name": name,
            "parent": d["joints"].index(parent) if parent else -1,
            "t": [round(float(x), 5) for x in d["rest"][k][3, :3]],
            "axis": None if k == 0 else ref_axis(name),
            "limit": None if k == 0 else SAFE[name[2:]],
        })
    return {
        "name": os.path.basename(path).replace(".usda", ""),
        "sample": d["name"],
        "vel": [round(v, 4) for v in d["vel"]],
        "T": d["T"], "fps": d["fps"], "n": n,
        "events": d["events"], "phis": d["phis"],
        "joints": joints,
        "rootT": [[round(float(x), 4) for x in d["trans"][i][0]] for i in range(n + 1)],
        "rot": [[[round(float(x), 6) for x in q] for q in d["rots"][i]] for i in range(n + 1)],
        "phase": [round(float(p), 5) for p in d["phase"]],
        "state": d["state"],
        "lc": [int(x) for x in d["lc"]],
        "rc": [int(x) for x in d["rc"]],
    }


TEMPLATE = _asset("viewer_template.html")


def main():
    clips = [bake(p) for p in sorted(glob.glob(os.path.join(paths.CLIPS, "*.usda")))]
    mesh = R("qmini_mesh.json")
    html = (R(os.path.basename(TEMPLATE))
            .replace("__CLIPS__", json.dumps(clips, separators=(",", ":")))
            .replace("__MESH__", mesh))
    out = os.path.join(paths.ensure(paths.DIST), "gait_viewer.html")
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB, {len(clips)} clips, "
          f"{json.loads(mesh)['nf']} triangles)")


if __name__ == "__main__":
    main()
