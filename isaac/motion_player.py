import math
import os

import numpy as np
import torch


class MotionPlayer:
    """Cyclic reference-motion sampler.

    Two ways to build one:

      MotionPlayer(keyframes=[(t, [q...]), ...], device=...)   # as before
      MotionPlayer.from_npz("qmini_step_in_place.npz", device) # generated clip

    Both expose ``sample(t)``; the clip form additionally exposes
    ``sample_vel``, ``sample_root`` and ``sample_phase``, and carries the joint
    names so the caller can line the columns up with the articulation instead of
    trusting index order.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, keyframes=None, device="cpu", degrees=False, *,
                 times=None, poses=None, joint_names=None, period=None,
                 fps=None, root_pos=None, root_quat=None, phase=None,
                 contact=None, meta=None):
        self.device = device

        if keyframes is not None:
            times = torch.tensor([k[0] for k in keyframes], device=device,
                                 dtype=torch.float32)
            poses = [[math.radians(a) for a in p] if degrees else list(p)
                     for _, p in keyframes]
            poses = torch.tensor(poses, device=device, dtype=torch.float32)
            period = float(times[-1])
            fps = None

        self.times = times
        self.poses = poses
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.period = float(period)
        self.length = self.period          # kept for backwards compatibility
        self.fps = fps
        self.root_pos = root_pos
        self.root_quat = root_quat
        self.phase_track = phase
        self.contact = contact
        self.meta = meta or {}

        # Uniform-grid clips get an O(1) sampler.  The table is a readout of an
        # analytic C1 spline, so interpolate it with a cyclic Catmull-Rom
        # Hermite rather than linearly: position and velocity then come from the
        # same polynomial and agree exactly, and both sit closer to the curve the
        # clip was generated from.
        self._uniform = fps is not None
        if self._uniform:
            self._n = self.poses.shape[0] - 1        # intervals; frame n == frame 0
            self._fps = float(fps)
            fwd = torch.roll(self.poses[:-1], -1, dims=0)
            bwd = torch.roll(self.poses[:-1], 1, dims=0)
            vel = (fwd - bwd) * (0.5 * self._fps)     # cyclic central difference
            self.vels = torch.cat([vel, vel[:1]], dim=0)

    # ----------------------------------------------------------------- clips
    @classmethod
    def from_npz(cls, path, device="cpu", joint_order=None):
        """Load a clip exported by ``export_isaac_motion.py``.

        ``joint_order`` reorders the columns to the caller's joint naming, so
        the articulation's own ordering is authoritative rather than assumed.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"reference motion not found: {path}")
        d = np.load(path, allow_pickle=False)
        names = [str(n) for n in d["joint_names"]]
        pos = d["joint_pos"]

        if joint_order is not None:
            joint_order = list(joint_order)
            missing = [n for n in joint_order if n not in names]
            if missing:
                raise KeyError(
                    f"{path} has no column(s) for {missing}. It provides: {names}")
            cols = [names.index(n) for n in joint_order]
            pos = pos[:, cols]
            names = joint_order

        t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=device,
                                                     dtype=dt)
        return cls(
            device=device,
            times=t(d["times"]), poses=t(pos), joint_names=names,
            period=float(d["period"]), fps=int(d["fps"]),
            root_pos=t(d["root_pos"]), root_quat=t(d["root_quat"]),
            phase=t(d["phase"]), contact=t(d["contact"]),
            meta={"source": str(d["source"]), "convention": str(d["convention"]),
                  "lower": t(d["joint_pos_lower"]), "upper": t(d["joint_pos_upper"]),
                  "antisymmetry_deg": float(d["antisymmetry_deg"])
                                      if "antisymmetry_deg" in d.files else None},
        )

    # -------------------------------------------------------------- sampling
    def _frac(self, t):
        """-> (i0, i1, alpha) for a cyclic uniform grid"""
        x = (t % self.period) * self._fps
        i0 = torch.clamp(x.floor().long(), 0, self._n - 1)
        return i0, i0 + 1, (x - i0.to(x.dtype)).unsqueeze(-1)

    def sample(self, t):
        if not self._uniform:
            t = t % self.period
            i = torch.clamp(torch.searchsorted(self.times, t), 1, len(self.times) - 1)
            t0, t1 = self.times[i - 1], self.times[i]
            q0, q1 = self.poses[i - 1], self.poses[i]
            a = ((t - t0) / (t1 - t0)).unsqueeze(-1)
            return q0 + a * (q1 - q0)
        i0, i1, a = self._frac(t)
        h = 1.0 / self._fps
        a2 = a * a
        a3 = a2 * a
        return ((2 * a3 - 3 * a2 + 1) * self.poses[i0]
                + (a3 - 2 * a2 + a) * h * self.vels[i0]
                + (-2 * a3 + 3 * a2) * self.poses[i1]
                + (a3 - a2) * h * self.vels[i1])

    def sample_vel(self, t):
        """Reference joint velocity [rad/s] -- the analytic derivative of
        ``sample``, so the two are consistent to machine precision."""
        if not self._uniform:
            dt = 1e-3
            return (self.sample(t + dt) - self.sample(t)) / dt
        i0, i1, a = self._frac(t)
        h = 1.0 / self._fps
        a2 = a * a
        return ((6 * a2 - 6 * a) * self.poses[i0]
                + (3 * a2 - 4 * a + 1) * h * self.vels[i0]
                + (-6 * a2 + 6 * a) * self.poses[i1]
                + (3 * a2 - 2 * a) * h * self.vels[i1]) / h

    def sample_phase(self, t):
        i0, i1, a = self._frac(t)
        return self.phase_track[i0] * (1 - a.squeeze(-1)) + self.phase_track[i1] * a.squeeze(-1)

    def sample_contact(self, t):
        i0, _, _ = self._frac(t)
        return self.contact[i0]

    def sample_root(self, t):
        """(position [m], quaternion wxyz) of the base, in the robot's USD axes."""
        i0, i1, a = self._frac(t)
        p = self.root_pos[i0] * (1 - a) + self.root_pos[i1] * a
        q0, q1 = self.root_quat[i0], self.root_quat[i1]
        flip = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        q = q0 * (1 - a) + q1 * flip * a          # nlerp: rotations here are small
        return p, q / q.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------ diagnostics
    def symmetry_residual(self):
        """max |q_right(t + T/2) + q_left(t)| in degrees.

        For a left/right symmetric gait this is zero once the PhysX sign
        convention is applied, so it is a sharp check that the sign map and the
        column ordering both survived loading and reordering.  It is much more
        sensitive than comparing against the joint limits, which on this robot
        are wide enough to accept a wholly sign-flipped reference.
        """
        if self.joint_names is None or not self._uniform:
            return None
        half = self._n // 2
        worst = 0.0
        pos = np.asarray(self.poses.cpu() if hasattr(self.poses, "cpu") else self.poses)
        for j in ("yaw", "roll", "pitch", "knee", "ankle"):
            ln, rn = f"Revolute_left_{j}", f"Revolute_right_{j}"
            if ln not in self.joint_names or rn not in self.joint_names:
                return None
            li, ri = self.joint_names.index(ln), self.joint_names.index(rn)
            shifted = np.roll(pos[:-1, ri], -half)
            worst = max(worst, float(np.abs(shifted + pos[:-1, li]).max()))
        return math.degrees(worst)

    def limit_report(self, lower, upper, names=None):
        """Compare the reference against the articulation's own joint limits.
        Catches gross mapping errors, but note the limits here are wide relative
        to this clip, so it will NOT catch a uniform sign flip -- use
        ``symmetry_residual`` for that."""
        names = names or self.joint_names or [f"j{i}" for i in range(self.poses.shape[1])]
        lo = self.poses.min(dim=0).values
        hi = self.poses.max(dim=0).values
        lines, bad = [], False
        for i, n in enumerate(names):
            ok = bool(lo[i] >= lower[i] - 1e-4 and hi[i] <= upper[i] + 1e-4)
            bad |= not ok
            lines.append(
                f"  {n:24s} ref [{math.degrees(lo[i]):7.2f},{math.degrees(hi[i]):7.2f}]"
                f"  limit [{math.degrees(lower[i]):7.2f},{math.degrees(upper[i]):7.2f}]"
                f"  {'ok' if ok else '<-- OUTSIDE'}")
        return "\n".join(lines), bad
