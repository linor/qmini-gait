# CLAUDE.md

Context for Claude Code. Read this before changing anything.

## What this is

`stylized_gait` implements Hopkins, Wiedebach, Cesare, Bishop, Knoop & Bächer,
*"Interactive Design of Stylized Walking Gaits for Robotic Characters"*
(ACM TOG 2024, Disney Research / WDI R&D), for **QMini**, a BDX-like bipedal
droid, so its gaits can drive RL training in Isaac Lab.

Section and table numbers in comments refer to that paper. The mapping:

| paper | here |
|---|---|
| Sec. 4.2 motion planning, footstep plan, path frame | `planner.py` |
| Sec. 4.3 animation parameters, Tab. 1 intervals, Tab. 2 curves | `core.py`, `samples.py` |
| Fig. 5 heel–toe "reverse foot lock" rig | `planner.heel_toe_transform`, and the `HeelPitch`/`ToePitch` prims in `blender_io.py` |
| Sec. 5.1 phase map Ω(φ) | `core.PhaseMap` |
| Sec. 5.2 two-sample blending | `blend.py` |
| Sec. 5.3 omnidirectional interpolation (Eqs 8–10) | `web/param_core.js`, `core.f_soft` |
| Sec. 6 interactive editor | `web/gait_editor.js` → `dist/gait_editor.html` |
| Sec. 7.1 gait feedback | `web/param_step.js` |
| Tab. 3 nine key velocities | `samples.py` |

## The pipeline

```
samples.py            authored Π for the nine key velocities
   │
   ├── overrides.py   sparse hand tuning from the HTML editor
   │
   ├── blender_io.py  Π ⇄ USD, for tuning in Blender  ──► library.py ──► gait_library.json
   │
   ▼
blend.py              Sec. 5.2 — interpolate Π, do NOT interpolate poses
   │
   ▼
planner.py            footstep plan + 5-DoF leg IK  ──►  joint angles
   │
   ├── usd_writer.py  ──► data/clips/*.usda        phase-tagged UsdSkel
   └── scripts/export_isaac_motion.py ──► data/qmini_step_in_place.npz
```

`verify.py` re-parses the emitted `.usda` from scratch and checks every
invariant. It shares no code with the writer on purpose — a bug common to both
would otherwise be invisible.

## Things that are easy to get wrong

These each cost real debugging time. Please don't rediscover them.

**Blend Π, never poses.** Foot height is `FK(q)`, nonlinear in the joint
angles, so averaging two clips' joint angles does not keep the foot on the
ground. Averaging Π does, because ground contact is a linear property of the
control points and a convex combination inherits it. Measured: parameter
blending 0.00 mm contact error, pose blending −0.24 to −11.12 mm, and it tracks
a 0.25 command at 0.10. `tests/test_blend.py` pins both halves — including a
test that *fails if pose blending stops being wrong*, since at that point the
comparison would be broken rather than fixed.

**Joint sign conventions.** A `UsdPhysics` revolute joint measures body1
relative to body0. Six of QMini's joints (the whole right leg, plus left ankle
and left yaw) have body0/body1 swapped relative to parent→child, so PhysX reads
them **negated**. Undo that and the limits are identical left/right except yaw,
which is genuinely mirrored.

> I got this wrong once and told Krijn the limits "cannot both be right for a
> mirrored mechanism", then clamped to their intersection. That was wrong. The
> clips are valid but conservative as a result: knee is really (−60, 50) and
> ankle (−40, 60), while `core.SAFE_LIMITS_DEG` still carries the intersection.
> Widening it buys ~10° on the ankle, which is the joint limiting reverse
> walking. Not yet done.

**A joint-limit check cannot catch a uniform sign flip** — negating every joint
still fits inside a symmetric envelope; verified empirically. What catches it is
half-cycle antisymmetry, `q_R(t + T/2) = −q_L(t)`: 0.0000° when right, ~48.8°
with a flipped right leg. A pure L/R swap is invisible to it, correctly — for a
symmetric gait that is just a half-cycle time shift.

**IK targets the sole, not the ankle.** There is no ankle roll, so the sole
tilts and drags the contact point by `ankle_height · sin(roll)`. Targeting the
ankle gave 1.4 mm/frame of stance slip; targeting the sole gives 0.014.

**The pelvis reference point is offset.** `base_link`'s origin sits 0.105 m
ahead of the sole centres in the CAD rest pose — hence `PELVIS_X_OFFSET` and
`STANCE_X_NEUTRAL` in `core.py`. Without it every clip floats forward.

**Robot axes are not the character axes.** Derived from the USD, not assumed:
robot **+X = left, +Y = backward, +Z = up**. Internally everything uses a
character frame (x forward, y left, z up) and `C2R = [[0,1,0],[-1,0,0],[0,0,1]]`
converts. The legs are reverse-knee (digitigrade), 5 DoF each, no ankle roll.

**Periodic splines duplicate the last knot.** Compute tangents over the `n−1`
independent nodes, then set `m[-1] = m[0]`. Doing it naively divides by zero.

**NaN hides from `>` comparisons.** A round-trip check written as
`if (diff > worst)` silently passed while every curve was NaN, because two
attributes reached the browser as `undefined`. `test_web_parity.py` now counts
non-finite comparisons explicitly. Keep it that way.

## Verified vs not

Verified, and re-checked by the test suite:

- nine clips: 0.000 mm ground penetration, ≤0.014 mm/frame stance slip, ≤89% of
  the safe joint envelope, exact periodicity, exact root travel
- the Blender round-trip **on the ASCII backend**, end to end: export → edit →
  import → library → plan → blend → regenerate → `verify.py`, all passing
- the browser port of the curve builder against Python (~1e-9 rad)
- the Isaac `.npz`: sign map, antisymmetry 0.0000°, sampler vs its own analytic
  derivative (1.35e−3 rad/s)

**Not executed, at all:**

- **anything requiring `pxr`.** `usd-core` could not be installed in the sandbox
  this was built in (no package index). `_import_pxr`, `validate`, `to_usdc` and
  `selftest` in `blender_io.py` are written but have never run. The two tests
  covering them skip automatically. Run these first:
  ```
  pip install usd-core
  python3 -m stylized_gait.blender_io selftest data/blender_edit/*.usda   # expect <1e-6
  python3 -m stylized_gait.blender_io validate data/blender_edit/*.usda
  ```
- **real Blender.** The round-trip was exercised by rewriting the USD in code,
  which proves the reader and writer agree — it does not prove Blender's
  importer/exporter preserves the rig. This is the largest untested assumption
  in the project.

## Conventions

- SI throughout in Python: metres, radians, seconds. USD layers are authored in
  **centimetres and degrees** (`metersPerUnit = 0.01`) because that is what DCCs
  expect; `blender_io.py` converts at the boundary and nowhere else.
- Phase landmarks φ = 0 / 0.1 / 0.5 / 0.6 / 1 → LLO / LTD / RLO / RTD / cycle end.
- Sample keys are unprefixed (`forward_fast`); clip *filenames* carry a sort
  prefix (`01_forward_fast.usda`). They are not the same string.
- `data/` holds generated artefacts. All of it is reproducible via `make`.

## Where things live

```
src/stylized_gait/   the library
scripts/             CLI entry points (run via make)
web/                 JS sources + HTML templates; make web bakes dist/*.html
dist/                standalone single-file HTML tools, open directly in a browser
isaac/               motion_player.py + qmini_leg_env.py for Isaac Lab
data/                clips, meshes, library, params, npz, the source robot USD
tests/               see tests/README.md
docs/GAIT_NOTES.md   the long-form design notes written during development
```

## Open threads

1. Run the pxr path (above). Highest value, lowest effort.
2. Try a real Blender round-trip.
3. Point the HTML viewers/editors at `gait_library.json` — currently they read
   `gait_params.json`, so Blender edits do not reach the browser tools. Krijn
   deferred this deliberately ("worry about the html visualizers and editors
   later"), so confirm before starting.
4. Optionally widen `SAFE_LIMITS_DEG` to the true per-side limits.

## Working agreements

- `make test` before and after any change to `src/`.
- New invariants belong in `tests/`, not in a throwaway script — the entire
  reason this repo has a test suite is that the original checks lived in `/tmp`
  and nearly did not survive.
- If a test fails, read the comment above the assertion before touching the
  tolerance. Each one records a specific bug.
- Say plainly when something is untested. There is a lot of verified work here
  and a couple of genuine gaps; blurring the two is worse than either.
