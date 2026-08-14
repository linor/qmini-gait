"""
qem.py -- quadric error metric mesh simplification (Garland & Heckbert 1997).

Vertex clustering wrecks CAD geometry: it collapses across thin features and
leaves spikes.  QEM edge collapse keeps planar faces and sharp edges, which is
what these machined parts are almost entirely made of.
"""

import heapq
import numpy as np


def weld(P, F, tol=1e-6):
    """merge coincident vertices so edges can actually be collapsed"""
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Pw = P[first]
    Fw = inv[F]
    ok = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    return Pw, Fw[ok]


def _quadrics(P, F):
    n = len(P)
    v0, v1, v2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    area = ln[:, 0] * 0.5
    nrm = nrm / np.maximum(ln, 1e-20)
    d = -np.einsum("ij,ij->i", nrm, v0)
    pl = np.concatenate([nrm, d[:, None]], axis=1)           # (m,4)
    K = pl[:, :, None] * pl[:, None, :] * area[:, None, None]
    Q = np.zeros((n, 4, 4))
    for k in range(3):
        np.add.at(Q, F[:, k], K)
    return Q


def simplify(P, F, target, boundary_weight=8.0):
    """collapse edges until <= target faces.  returns (P', F')"""
    P, F = weld(P.astype(np.float64), F)
    if len(F) <= target:
        return P, F
    n = len(P)
    Q = _quadrics(P, F)
    P = P.copy()

    faces = F.copy()
    f_alive = np.ones(len(F), bool)
    v_alive = np.ones(n, bool)
    vf = [set() for _ in range(n)]
    for i, f in enumerate(faces):
        for v in f:
            vf[v].add(i)

    # penalise open boundaries so silhouettes are preserved
    edge_count = {}
    for f in faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (a, b) if a < b else (b, a)
            edge_count[e] = edge_count.get(e, 0) + 1
    for (a, b), c in edge_count.items():
        if c == 1:                                   # boundary edge
            d = P[b] - P[a]
            ln = np.linalg.norm(d)
            if ln < 1e-12:
                continue
            # plane perpendicular to the surface through the boundary edge
            nb = np.cross(d / ln, [0.0, 0.0, 1.0])
            if np.linalg.norm(nb) < 1e-9:
                nb = np.cross(d / ln, [0.0, 1.0, 0.0])
            nb /= np.linalg.norm(nb)
            pl = np.array([*nb, -nb @ P[a]])
            K = np.outer(pl, pl) * boundary_weight * ln
            Q[a] += K
            Q[b] += K

    def opt_pos(a, b):
        Qs = Q[a] + Q[b]
        A = Qs.copy()
        A[3] = [0.0, 0.0, 0.0, 1.0]
        try:
            v = np.linalg.solve(A, np.array([0.0, 0.0, 0.0, 1.0]))
            if not np.all(np.isfinite(v)):
                raise np.linalg.LinAlgError
            p = v[:3]
        except np.linalg.LinAlgError:
            cand = [P[a], P[b], 0.5 * (P[a] + P[b])]
            p = min(cand, key=lambda c: _err(Qs, c))
        return p, _err(Qs, p)

    def _err(Qs, p):
        v = np.array([p[0], p[1], p[2], 1.0])
        return float(v @ Qs @ v)

    heap = []
    for (a, b) in edge_count:
        p, c = opt_pos(a, b)
        heapq.heappush(heap, (c, a, b))

    def neighbours(v):
        out = set()
        for i in vf[v]:
            out.update(faces[i])
        out.discard(v)
        return out

    def flips(v_keep, v_drop, p):
        """reject a collapse that would fold a face over"""
        for i in vf[v_keep] | vf[v_drop]:
            if not f_alive[i]:
                continue
            f = faces[i]
            if v_keep in f and v_drop in f:
                continue                              # face disappears
            q = [p if (x == v_keep or x == v_drop) else P[x] for x in f]
            n0 = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
            n1 = np.cross(q[1] - q[0], q[2] - q[0])
            l0, l1 = np.linalg.norm(n0), np.linalg.norm(n1)
            if l1 < 1e-14:
                return True
            if l0 > 1e-14 and (n0 @ n1) / (l0 * l1) < -0.2:
                return True
        return False

    nf = len(F)
    while nf > target and heap:
        c, a, b = heapq.heappop(heap)
        if not (v_alive[a] and v_alive[b]):
            continue
        if b not in neighbours(a):
            continue
        p, c2 = opt_pos(a, b)
        if c2 > c + 1e-12:                            # stale cost, re-queue
            heapq.heappush(heap, (c2, a, b))
            continue
        if flips(a, b, p):
            continue

        # collapse b into a
        P[a] = p
        Q[a] = Q[a] + Q[b]
        for i in list(vf[b]):
            if not f_alive[i]:
                vf[b].discard(i)
                continue
            f = faces[i]
            if a in f:                                # degenerate -> remove
                f_alive[i] = False
                nf -= 1
                for v in f:
                    vf[v].discard(i)
                continue
            faces[i] = np.where(f == b, a, f)
            vf[a].add(i)
            vf[b].discard(i)
        v_alive[b] = False
        for w in neighbours(a):
            pp, cc = opt_pos(a, w)
            heapq.heappush(heap, (cc, min(a, w), max(a, w)))

    F2 = faces[f_alive]
    used = np.unique(F2)
    remap = -np.ones(n, np.int64)
    remap[used] = np.arange(len(used))
    return P[used], remap[F2]
