# The papers, and how they map onto this code

## 1. Stylized gaits — `stylized-gaits-2024.pdf`

Hopkins, Wiedebach, Cesare, Bishop, Knoop & Bächer, *Interactive Design of
Stylized Walking Gaits for Robotic Characters*, ACM TOG 43(4), 2024.
Disney Research / Walt Disney Imagineering R&D.

**This is the paper this repo implements.** Everything in `src/stylized_gait/`
traces back to a numbered part of it:

| paper | code |
|---|---|
| Sec. 4.2 motion planning: footstep plan, nominal support/swing trajectories, path frame | `planner.py` — `foot_pose_at`, `pelvis_pose_at`, `core.path_pose` |
| Sec. 4.3 animation parameters | `core.GaitSample.__init__`, `samples.py` |
| Tab. 1 time intervals | `core.PHI_LANDMARKS`, the `f_ltd` / `f_rlo` / `f_rtd` fractions |
| Tab. 2 function curves | the `c_*` attributes on a sample; ported to JS in `web/gait_curves.js` |
| Fig. 5 heel–toe "reverse foot lock" rig | `planner.heel_toe_transform`; the `HeelPitch`/`ToePitch` prims in `blender_io.py` |
| Sec. 5.1 phase map Ω(φ) | `core.PhaseMap` (monotone Fritsch–Carlson, so it inverts) |
| Sec. 5.2 two-sample blending | `blend.py` — `BlendedCurve`, `BlendedSample` |
| Sec. 5.3 omnidirectional interpolation, Eqs 8–10 | `web/param_core.js`; `f_soft` (Eq. 8) is in `core.py` |
| Sec. 6 interactive gait editor | `web/gait_editor.js` → `dist/gait_editor.html` |
| Sec. 7.1 gait feedback | `web/param_step.js` |
| Tab. 3 nine key velocities | `samples.py` — `X_MAX`, `X_SLOW`, `Y_MAX`, `TH_MAX` |

### Where this implementation knowingly departs

- **QMini, not the paper's robot.** All kinematics are extracted from
  `data/qmini_urdf-2legs.usda`: reverse-knee legs, 5 DoF per leg, **no ankle
  roll**. The missing ankle roll is why IK targets the sole rather than the
  ankle — see `CLAUDE.md`.
- **No QP.** The paper solves a quadratic program for the CoM/CoP trajectory
  (Appendix). This repo has no CoM/CoP planner at all: the pelvis follows the
  Tab. 2 curves directly and the legs are solved by damped-least-squares IK.
  Fine for generating reference animation; **not** a balance controller.
- **Joint-space output.** Clips are baked to joint angles in UsdSkel, per the
  choice made early on, rather than kept as task-space targets.

## 2. Bipedal robotic character — `bipedal-character-2024.pdf`

Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller & Bächer,
*Design and Control of a Bipedal Robotic Character*, RSS 2024.

**Nothing in this repo was written with reference to this paper.** It was not
part of the context when the code was built — it only surfaced when the project
was packaged. It is the companion hardware/control paper for the BDX-style
droid: the robot design, and the RL-based control stack that would consume
reference motion of the kind this repo produces.

That makes it the relevant paper for the *next* stage — the Isaac Lab side in
`isaac/` — rather than for the gait generator. Read it before making decisions
about the RL environment, reward shaping, or how reference motion is tracked;
`isaac/qmini_leg_env.py` was written without it and its choices (residual
actions, `action_scale=0.15`, kinematic base) were not checked against it.
