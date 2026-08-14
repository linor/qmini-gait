"""
extract_meshes.py -- pull the visual meshes out of qmini_urdf-2legs.usda, work
out which skeleton joint each rigid body is rigidly attached to (from the
physics joint graph, not by name matching), decimate, and emit a compact binary
blob for the WebGL viewer.
"""
import re, os, sys, json, base64, collections
import numpy as np

from stylized_gait import paths

SRC = paths.ROBOT_USD
HERE = paths.DATA
src = open(SRC).read()

# ---------------------------------------------------------------- joint graph
def prim_block(text, start):
    """body of the prim whose header starts at `start`, by brace balancing.
    (A non-greedy regex is wrong here: root_joint is indented differently from
    the joints under /joints, so a fixed-indent terminator swallows the prim
    that follows it -- that silently dropped Revolute_left_ankle.)"""
    i = text.index("{", start)
    d, j = 0, i
    while True:
        if text[j] == "{": d += 1
        elif text[j] == "}":
            d -= 1
            if d == 0: return text[i + 1:j]
        j += 1

fixed, revolute = [], {}
for m in re.finditer(r'def Physics(Fixed|Revolute)Joint "(\w+)"', src):
    kind, name = m.group(1), m.group(2)
    body = prim_block(src, m.end())
    b0 = re.search(r'body0 = <([^>]*)>', body).group(1).split("/")[-1]
    b1 = re.search(r'body1 = <([^>]*)>', body).group(1).split("/")[-1]
    if b0 == "qmini_urdf_2legs" or b1 == "qmini_urdf_2legs":
        continue                      # world articulation root, not a body weld
    if kind == "Fixed":
        fixed.append((b0, b1))
    else:
        revolute[name] = (b0, b1)

# union-find over rigidly-welded bodies
parent = {}
def find(a):
    parent.setdefault(a, a)
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b in fixed:
    union(a, b)
for a, b in revolute.values():
    find(a); find(b)

# BFS from the base_link cluster over the revolute joints -> which cluster each
# revolute joint drives
root_cluster = find("base_link")
adj = collections.defaultdict(list)
for jn, (a, b) in revolute.items():
    adj[find(a)].append((find(b), jn))
    adj[find(b)].append((find(a), jn))

cluster_joint = {root_cluster: "Root"}
NAME = {"left_yaw": "L_hip_yaw", "left_roll": "L_hip_roll", "left_pitch": "L_hip_pitch",
        "left_knee": "L_knee", "left_ankle": "L_ankle",
        "right_yaw": "R_hip_yaw", "right_roll": "R_hip_roll", "right_pitch": "R_hip_pitch",
        "right_knee": "R_knee", "right_ankle": "R_ankle"}
q = collections.deque([root_cluster])
seen = {root_cluster}
while q:
    c = q.popleft()
    for nb, jn in adj[c]:
        if nb in seen: continue
        seen.add(nb)
        cluster_joint[nb] = NAME[jn.replace("Revolute_", "")]
        q.append(nb)

# ---------------------------------------------------------------- joint anchors
ANCH_ROBOT = {
    "Root": (0.0, 0.0, 0.0),
    "L_hip_yaw": (5.50, 10.289148, -2.827945), "L_hip_roll": (10.90, 8.489148, -10.127945),
    "L_hip_pitch": (13.80, 8.639149, -10.127945), "L_knee": (13.77, 16.770876, -18.259672),
    "L_ankle": (15.85, 11.469564, -32.824910),
    "R_hip_yaw": (-5.50, 10.289148, -2.827945), "R_hip_roll": (-10.90, 8.489148, -10.127945),
    "R_hip_pitch": (-13.80, 8.639149, -10.127945), "R_knee": (-13.77, 16.770876, -18.259672),
    "R_ankle": (-15.85, 11.469564, -32.824910),
}

# ---------------------------------------------------------------- mesh reading
def read_mesh(body, kind):
    i = src.index(f'def Xform "{body}"')
    j = src.find('def Xform "', i + 20)
    blk = src[i:(len(src) if j < 0 else j)]
    k = blk.find(f'def "{kind}"')
    if k < 0: return None
    sub = blk[k:]
    counts = [int(x) for x in re.search(r'int\[\] faceVertexCounts = \[(.*?)\]', sub, re.S).group(1).split(",")]
    idx = [int(x) for x in re.search(r'int\[\] faceVertexIndices = \[(.*?)\]', sub, re.S).group(1).split(",")]
    pts = np.array([[float(v) for v in t.split(",")]
                    for t in re.findall(r'\(([^)]*)\)',
                    re.search(r'point3f\[\] points = \[(.*?)\]\n', sub, re.S).group(1))])
    tris, o = [], 0
    for c in counts:
        for t in range(1, c - 1):
            tris.append([idx[o], idx[o + t], idx[o + t + 1]])
        o += c
    return pts, np.array(tris, dtype=np.int64)

# ---------------------------------------------------------------- decimation
from stylized_gait import qem

# Triangle budgets.  These are CAD tessellations -- 50k triangles for a boxy
# torso, 20k for a cylindrical motor can -- so QEM removes ~95 % of them with
# almost no visible change.
BUDGET = {"base_link": 3000, "Left_Foot_1": 1800, "Riggt_Foot_1": 1800,
          "Left_Thigh_1": 1200, "Right_Thigh_1": 1200,
          "Left_Calf_1": 1000, "Right_Calf_1": 1000,
          "Left_Hip_Mount_1": 900, "Right_Hip_Mount_1": 900,
          "Side_Swing_Connector_1": 700, "Right_Side_Swing_Connector_1": 700,
          "Side_Swing_Gear_1": 500, "Right_Side_Swing_Gear_1": 500}
MOTOR_BUDGET = 600

bodies = re.findall(r'def Xform "([^"]+)" \(\s*\n\s*prepend apiSchemas = \["PhysicsRigidBodyAPI"', src)
parts, VS, IS, voff = [], [], [], 0
report = []
for b in bodies:
    jn = cluster_joint.get(find(b))
    if jn is None:
        print("  !! unassigned body", b); continue
    m = read_mesh(b, "visuals") or read_mesh(b, "collisions")
    kind = "visuals" if read_mesh(b, "visuals") else "collisions"
    P, F = m
    budget = BUDGET.get(b, MOTOR_BUDGET)
    V, G = qem.simplify(P, F, budget)
    V = V - np.array(ANCH_ROBOT[jn])          # express in the joint's local frame
    parts.append({"body": b, "joint": jn, "v0": voff, "nv": len(V), "nf": len(G)})
    VS.append(V.astype(np.float32)); IS.append((G + voff).astype(np.uint32))
    voff += len(V)
    report.append((b, jn, kind, len(F), len(G)))
    print(f"    {b[:34]:36s} {len(F):6d} -> {len(G):5d}", flush=True)

V = np.concatenate(VS); I = np.concatenate(IS)
print(f"{'body':34s} {'joint':12s} {'src':>7s} {'->':>6s}")
for b, jn, kind, nf0, nf in report:
    print(f"  {b[:32]:34s} {jn:12s} {nf0:7d} {nf:6d}  {kind}")
print(f"TOTAL  {len(V)} verts  {len(I)} tris")

# smooth-ish normals (area weighted)
N = np.zeros_like(V)
tri = V[I]
fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
for k in range(3):
    np.add.at(N, I[:, k], fn)
N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)

blob = {
    "parts": parts,
    "nv": len(V), "nf": len(I),
    "pos": base64.b64encode(V.astype(np.float32).tobytes()).decode(),
    "nrm": base64.b64encode((N * 32767).astype(np.int16).tobytes()).decode(),
    "idx": base64.b64encode(I.astype(np.uint32).tobytes()).decode(),
}
open(paths.MESH_JSON, "w").write(json.dumps(blob, separators=(",", ":")))
print("wrote qmini_mesh.json",
      round(os.path.getsize(paths.MESH_JSON) / 1024), "KB")
