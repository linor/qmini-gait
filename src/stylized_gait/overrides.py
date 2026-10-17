"""
gait_overrides.py -- apply hand-tuned edits on top of the authored samples.

The editor (gait_editor.html) writes a sparse ``gait_overrides.json``; only the
values you actually touched appear in it, so it stays diffable and any single
tweak can be reverted by deleting one line.  ``samples.py`` remains the authored
baseline and is never rewritten.

Curve knots are stored in NORMALISED interval coordinates (x in 0..1 across the
curve's own LS / RS / CY interval), and tangents per unit of x.  That way an edit
to the cycle period or the contact event times rescales an edited curve instead
of leaving it stranded at the old absolute times.

    {
      "version": 1,
      "samples": {
        "forward_fast": {
          "attrs":  {"swing_height": 0.052, "psi_toe_lo": 0.32},
          "curves": {"foot_z.L": {"x": [...], "v": [...], "m": [...]}}
        }
      }
    }
"""

import json
import os

import numpy as np

from .core import HermiteSpline

# constructor arguments of GaitSample that the editor may override
ATTR_KEYS = (
    "T", "f_ltd", "f_rlo", "f_rtd",
    "stance_width", "stance_x_bias", "stance_splay",
    "swing_height", "psi_toe_lo", "psi_heel_td", "dt_psi_lo", "dt_psi_td",
    "pelvis_z", "pelvis_bounce", "pelvis_roll", "pelvis_pitch",
    "swing_x_lead", "swing_y_lead", "foot_yaw_amp", "pelvis_sway",
)

# curve name -> (attribute on the sample, is it per-side?)
CURVES = {
    "foot_x": ("c_foot_x", True), "foot_y": ("c_foot_y", True),
    "foot_z": ("c_foot_z", True), "foot_yaw": ("c_foot_yaw", True),
    "foot_gamma": ("c_foot_gamma", True),
    "psi_toe": ("c_psi_toe", True), "psi_heel": ("c_psi_heel", True),
    "pelvis_z": ("c_pelvis_z", False), "pelvis_roll": ("c_pelvis_roll", False),
    "pelvis_pitch": ("c_pelvis_pitch", False), "pelvis_yaw": ("c_pelvis_yaw", False),
}


def curve_interval(sample, name, side):
    """the (start, end) time span a curve lives on, on THIS sample"""
    if name.startswith("pelvis"):
        return 0.0, sample.T
    return (0.0, sample.t_ltd) if side == "L" else (sample.t_rlo, sample.t_rtd)


def load(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    if d.get("version") != 1:
        raise ValueError(f"{path}: unsupported override version {d.get('version')}")
    return d.get("samples", {})


def apply(samples, path_or_dict=None, verbose=True):
    """Return a new dict of samples with the overrides applied.

    ``samples`` is what ``samples.build()`` returns.  Attribute edits are fed
    back through the GaitSample constructor so every curve derived from them is
    rebuilt consistently; explicit curve edits are then laid on top.
    """
    from . import samples as SM

    ov = path_or_dict if isinstance(path_or_dict, dict) else load(path_or_dict)
    # accept either the on-disk form {"version":1,"samples":{...}} or the bare
    # {name: edit} mapping, so a parsed override file and a hand-built dict
    # behave the same.  dump_template writes the former.
    if isinstance(ov, dict) and "samples" in ov and "version" in ov:
        if ov.get("version") != 1:
            raise ValueError(f"unsupported override version {ov.get('version')}")
        ov = ov.get("samples", {})
    if not ov:
        return samples

    # the constructor arguments each sample was originally built with
    base_args = _authored_args()
    out, touched = dict(samples), []

    for name, edit in ov.items():
        if name not in samples:
            raise KeyError(f"override for unknown sample '{name}'")
        args = dict(base_args[name])
        attrs = edit.get("attrs", {}) or {}
        for k, v in attrs.items():
            if k not in ATTR_KEYS:
                raise KeyError(f"{name}: unknown attribute '{k}'")
            args[k] = float(v)
        _check_intervals(name, args)

        s = SM.GaitSample(name, args.pop("vel"), **args)

        for key, cv in (edit.get("curves", {}) or {}).items():
            cname, _, side = key.partition(".")
            if cname not in CURVES:
                raise KeyError(f"{name}: unknown curve '{cname}'")
            attr, per_side = CURVES[cname]
            if per_side and side not in ("L", "R"):
                raise KeyError(f"{name}: curve '{cname}' needs a .L or .R suffix")
            a, b = curve_interval(s, cname, side or "L")
            dur = b - a
            x = np.asarray(cv["x"], float)
            t = a + x * dur
            m = np.asarray(cv["m"], float) / max(dur, 1e-9)
            sp = HermiteSpline(t, np.asarray(cv["v"], float), tangents=m,
                               periodic=cname.startswith("pelvis"))
            if per_side:
                getattr(s, attr)[side] = sp
            else:
                setattr(s, attr, sp)

        s.notes = samples[name].notes + "  [hand tuned]"
        out[name] = s
        touched.append(f"{name}({len(attrs)} attr, "
                       f"{len(edit.get('curves', {}) or {})} curve)")

    if verbose and touched:
        print("applied overrides: " + ", ".join(touched))
    return out


def _check_intervals(name, args):
    f = (0.0, args["f_ltd"], args["f_rlo"], args["f_rtd"], 1.0)
    if not all(f[i] < f[i + 1] for i in range(4)):
        raise ValueError(
            f"{name}: contact events out of order "
            f"(0 < {args['f_ltd']} < {args['f_rlo']} < {args['f_rtd']} < 1)")
    if args["psi_heel_td"] > 1e-9:
        raise ValueError(f"{name}: psi_heel_td must be <= 0 (heel strike lifts the toe)")
    if args["psi_toe_lo"] < -1e-9:
        raise ValueError(f"{name}: psi_toe_lo must be >= 0 (toe off lifts the heel)")


def _authored_args():
    """Recover the constructor arguments samples.build() used, by re-running it
    with a GaitSample stand-in that records what it was called with.

    Defaults are filled in explicitly: samples.py leaves several arguments at
    their default (swing_y_lead, foot_yaw_amp, ...) and a partial dict would
    otherwise reach the browser as undefined and poison the curves with NaN.
    """
    import inspect
    from . import samples as SM
    from . import core as GC

    captured = {}
    real = GC.GaitSample
    defaults = {k: v.default for k, v in
                inspect.signature(real.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty}

    class Spy(real):
        def __init__(self, name, vel, **kw):
            captured[name] = dict(defaults, **dict(kw, vel=vel))
            super().__init__(name, vel, **kw)

    GC.GaitSample = Spy
    SM.GaitSample = Spy
    try:
        SM.build()
    finally:
        GC.GaitSample = real
        SM.GaitSample = real
    return captured


def dump_template(samples, path):
    """Write an override file that restates every sample's current values --
    handy as a starting point for editing by hand rather than in the browser."""
    base = _authored_args()
    out = {"version": 1, "samples": {}}
    for name in samples:
        a = {k: float(v) for k, v in base[name].items() if k in ATTR_KEYS}
        out["samples"][name] = {"attrs": a, "curves": {}}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path
