# QMini reference gait animations

Nine phase-tagged UsdSkel walk cycles for `qmini_urdf-2legs.usda`, one per key
velocity in Table 3 of Hopkins et al., *Interactive Design of Stylized Walking
Gaits for Robotic Characters* (ACM TOG 2024). They are the sample set the gait
sampler blends between with the three-step omnidirectional interpolation of
Sec. 5.3.

| file | sample | ẋ (m/s) | ẏ (m/s) | θ̇ (rad/s) | T (s) | frames |
|---|---|---|---|---|---|---|
| `01_forward_fast.usda` | Forward Fast | +0.35 | 0 | 0 | 0.80 | 49 |
| `02_forward_slow.usda` | Forward Slow | +0.15 | 0 | 0 | 1.00 | 61 |
| `03_step_in_place.usda` | Step In Place | 0 | 0 | 0 | 1.10 | 67 |
| `04_reverse_slow.usda` | Reverse Slow | −0.15 | 0 | 0 | 1.05 | 64 |
| `05_reverse_fast.usda` | Reverse Fast | −0.35 | 0 | 0 | 0.85 | 52 |
| `06_left_strafe.usda` | Left Strafe | 0 | +0.16 | 0 | 0.78 | 48 |
| `07_right_strafe.usda` | Right Strafe | 0 | −0.16 | 0 | 0.78 | 48 |
| `08_left_turn.usda` | Left Turn | 0 | 0 | +0.60 | 1.00 | 61 |
| `09_right_turn.usda` | Right Turn | 0 | 0 | −0.60 | 1.00 | 61 |

60 fps. Each file is self-contained: `SkelRoot` → `Skeleton` (bind + rest
tables) → `SkelAnimation`. Units and axes match the robot model exactly
(centimetres, `metersPerUnit = 0.01`, Z-up, **+X = left, +Y = backward,
+Z = up**), so a clip overlays the model with no conversion.

## No keyframes

Nothing is hand-keyed. Every quantity comes from an analytically evaluated
spline:

* **Swing curves** (foot *x, y, z*, and the rotation angles θ, ψ_heel, ψ_toe, γ)
  are piecewise cubic Hermite splines on the LS/RS intervals, value-clamped to
  zero at the interval ends so the foot starts and ends exactly at the nominal
  foothold pose.
* **Cycle curves** (pelvis *z*, yaw/pitch/roll) are periodic Hermite splines
  with wrapped tangents, so they are C¹ across the loop point.
* **Nominal swing interpolation** uses the paper's cubic `f_soft(α) = −2α³+3α²`
  (Eq. 8), which has zero derivative at both ends.
* **The phase map Ω(φ)** is a monotone Fritsch–Carlson cubic through the five
  contact-event way-points, so it stays strictly invertible.

The frame samples in the USD are just a 60 Hz readout of those curves; the
curve parameters themselves are preserved in metadata (below) so the clips can
be re-evaluated at any rate or blended as parameters rather than as poses.

## Phase tagging

Every clip carries the full Sec. 5.1 phase-space blending data on the
`SkelRoot`:

```
stylizedGait:eventNames    = ["LLO", "LTD", "RLO", "RTD", "CYCLE_END"]
stylizedGait:eventTimes    = [ ... seconds from cycle start ... ]
stylizedGait:eventPhases   = [0, 0.1, 0.5, 0.6, 1]
stylizedGait:phaseMap:phi  = [ ... 21 samples of Omega(phi) ... ]
stylizedGait:phaseMap:time = [ ... ]
stylizedGait:interval:CY / :LS / :RS      # Tab. 1 time intervals (min, max)
stylizedGait:attr:*                        # stance width / x-bias / splay,
                                           # swing height, psi_toe, psi_heel,
                                           # dtPsiLO, dtPsiTD, pelvis height
stylizedGait:velocity_mps_radps            # Tab. 3 key velocity
stylizedGait:cycleDisplacement_m_rad       # travel per cycle, for looping
stylizedGait:physicsJoints                 # map back to the PhysicsRevoluteJoints
stylizedGait:jointLimits_rad / :jointLimitsSafe_rad
```

and per-frame tracks:

```
primvars:phase                # phi in [0, 1], monotone, landmarks land on the events
primvars:contactState         # "LS" | "DS1" | "RS" | "DS2"
primvars:leftContact          # 0 / 1
primvars:rightContact         # 0 / 1
primvars:cycleTimeNormalized  # t / T
```

Two clips are registered for blending with `t_B = Ω_B(Ω_A⁻¹(t_A))`; because the
landmarks are fixed at 0 / 0.1 / 0.5 / 0.6 / 1, contact events always map onto
contact events, which is what preserves the contact constraints through a blend.

## Robot findings that shaped the result

Extracted from the USD rather than assumed:

* The legs are **reverse-knee (digitigrade)** — the thigh runs back-and-down,
  the shank forward-and-down. The model frame is +X left, +Y **backward**, +Z up.
* `hip_pitch`, `knee` and `ankle` share the *same* world axis (−X) on both legs;
  their `localRot0` quaternions match to 1e−18, yet the USD authors mirror-image
  limits per side. I first read that as a CAD-export artifact and clamped the
  animations to the **intersection** — yaw ±30°, roll ±15°, pitch ±50°,
  knee ±50°, ankle ±40°.

  **That was wrong, and `export_isaac_motion.py` resolves it.** A UsdPhysics
  revolute joint measures body1 relative to body0, and the right leg plus the
  left ankle and left yaw have body0/body1 swapped relative to parent→child. So
  those six joints are simply expressed in the opposite sign convention. Undo it
  and roll/pitch/knee/ankle limits are *identical* on both legs, with only yaw
  mirrored — which is correct for yaw:

  | joint | left | right | |
  |---|---|---|---|
  | yaw | (−30, 45) | (−45, 30) | mirrored, as it should be |
  | roll | (−15, 15) | (−15, 15) | identical |
  | pitch | (−50, 50) | (−50, 50) | identical |
  | knee | (−60, 50) | (−60, 50) | identical |
  | ankle | (−40, 60) | (−40, 60) | identical |

  The shipped clips are therefore **valid but conservative**: the knee really has
  (−60, 50) rather than ±50 and the ankle (−40, 60) rather than ±40. Widening
  `SAFE_LIMITS_DEG` in `gait_core.py` to the per-side values would buy roughly
  10° more on the ankle, which is the joint that limits reverse walking.
  Both the raw and the intersected limits are in the clip metadata.
* There is **no ankle roll**. Foot roll can only come from the hip, so the sole
  necessarily tilts and rides an edge during lateral motion. Two consequences:
  * The IK is driven from the **sole reference point**, not the ankle. Targeting
    the ankle (from an assumed-flat foot) drags the contact point sideways by
    `ankle_height · sin(roll)`; that showed up as 1.4 mm/frame of stance slip
    before the fix, and is 0.014 mm/frame after it.
  * The paper's heel/toe contact-height rig is extended with a lateral term: the
    foot is lifted so the lowest sole *corner* sits exactly on the ground.
* **ẏ_max is 0.16 m/s, not the 0.20 m/s requested.** Lateral travel is bounded by
  the ±15° hip roll; 0.20 m/s saturates it. 0.16 m/s peaks at ~86% of the roll
  range. Everything else hit the requested values. Turning is the robot's
  cheapest direction — θ̇ = 0.6 rad/s uses only 69%, and ~1.1 rad/s is reachable
  if you want a more aggressive turn sample later.

## Style

Neutral / mechanically clean, per the brief: flat-foot contacts with a modest
heel-toe roll that scales with speed (16° toe-off at ẋ_max, none in place or
strafing), a small two-per-cycle pelvis bounce, 2–2.5° of pelvis lean toward the
support foot, symmetric stance.

Pelvis *x/y* follows the path frame exactly, as in the paper — Table 2 has no
pelvis translation curves, since the CoM/CoP planner downstream is what shifts
weight over the support polygon. These clips are kinematic references, not
dynamically balanced trajectories.

## Verification

`verify.py` re-parses the emitted `.usda` text from scratch, rebuilds forward
kinematics from the bind/rest tables and the animation quaternions, and checks
the invariants. Current results — all nine pass:

| check | result |
|---|---|
| ground penetration | 0.000 cm (lowest sole corner never below the floor) |
| stance contact slip | ≤ 0.014 mm/frame |
| joint range used | ≤ 89% of the safe envelope (`reverse_fast`, ankle) |
| joint-space periodicity | exact (Δq = 0) |
| root travel vs commanded velocity | 0.000 mm error |
| phase track | monotone over [0, 1], landmarks on the event frames |
| contact sequence | LS → DS1 → RS → DS2 in every clip |
| Ω(φ) | strictly increasing in every clip |

`gait_reference_overview.png` plots joint angles, sole clearance and the phase
map for all nine.

## Viewer

`gait_viewer.html` — open it directly in a browser. No server, no network, no
CDN: the clips *and* the robot geometry are baked into the single 758 KB file.
Rendering is raw WebGL2 (no library), with a 2D canvas overlay for the skeleton
and HUD.

**The robot is the actual `qmini_urdf-2legs.usda` geometry**, not a stick figure:

* the visual meshes of all 23 rigid bodies, simplified from 318k triangles to
  20.8k with quadric error metric edge collapse (`qem.py`). Vertex clustering
  was tried first and shredded the CAD features — QEM keeps the planar faces and
  sharp edges these machined parts are made of.
* each body is bound to a skeleton joint by walking the file's **own physics
  joint graph** — union-find over the `PhysicsFixedJoint` welds, then a BFS out
  from `base_link` over the `PhysicsRevoluteJoint`s. Nothing is matched by name,
  so a renamed or re-welded part still binds correctly.
* four motor bodies have no visual mesh in the USD; those fall back to their
  collision mesh.
* shading is exact flat shading from screen-space derivatives, so no normals are
  stored.

Controls:

* click a clip, or press `1`–`9`
* drag to orbit, scroll to zoom, shift-drag to pan, `R` to reset the view
* space to play/pause, `←`/`→` to step, scrub the timeline or frame slider
* the timeline shows the phase curve, the LLO/LTD/RLO/RTD markers with their φ
  values, and a contact band (red = left swing, blue = right swing, grey = double
  support)
* signed joint angles as bars over the safe envelope, amber past 90%
* toggles: Robot mesh, Skeleton overlay, follow-root vs world camera, footprint
  history, ghost trail, axes
* drag any regenerated `.usda` onto the view to load it — the page parses the
  format itself

Rebuild with `python3 extract_meshes.py` (a few minutes; only needed if the robot
model changes) then `python3 make_viewer.py`.
`mesh_preview.png` shows four frames of what it renders.

Checks run on the generated page, in Node:

| check | result |
|---|---|
| viewer FK vs the Python verifier | identical sole clearance, slip and joint use |
| GPU model matrix vs overlay FK | agree to 1.8e−5 px over 5562 sampled vertices |
| index buffer | all indices in range, part ranges tile the buffer exactly |
| body→joint binding | every part's joint exists in every clip's skeleton |
| drag-drop `.usda` parser | round-trips to the baked data |

## Interactive blending viewer

`gait_blend.html` — drive the blended gait from the keyboard. Same self-contained
setup, same real `qmini_urdf-2legs.usda` geometry and WebGL2 renderer.

**W/S** forward / reverse · **A/D** strafe · **Q/E** turn · **shift** full speed ·
**space** stop · drag to orbit, **R** reset view.

**Gamepad** (both interactive viewers, standard mapping):

| control | action |
|---|---|
| left stick | ẋ forward / ẏ strafe — analog, so it sweeps the blend space continuously |
| right stick X | θ̇ turn |
| A / cross | toggle slow ↔ fast |
| B / circle | stop |
| start / back | reset |
| D-pad | digital forward/back + turn |

Keyboard stays live alongside it; whichever axis is pushed further wins, and
holding shift forces fast regardless of the toggle. The bottom bar shows both
stick positions and the pad name. Browsers only expose a pad after it sends an
event, so press a button if it reads "no gamepad". Sticks use a rescaled radial
deadzone (0.12 / 0.14), so the command starts from zero just outside it rather
than jumping.

What runs each frame:

1. The key state becomes a velocity command, rate-limited (0.55 m/s², 1.4 rad/s²).
2. **Weights** from the paper's three-step interpolation (Eqs 8–10): nearest
   forward/reverse pair on ẋ; nearest turn/strafe pair by polar angle β in the
   normalised (ẏₙ, θ̇ₙ) plane; the two intermediates combined by
   α<sub>turn-strafe</sub> = f_soft(ρₙ)·(1 − η·f_soft(|ẋₙ|)). η is exposed as a
   slider so you can see it attenuate the turn sample at speed.
3. **Blended timing** — event times are the convex combination of the samples'
   (Sec 5.2), so the ordering constraint survives. A monotone Ω is rebuilt from
   them every frame; φ is the state and gait time is stepped through Ω.
4. **Registered sampling** — each contributing clip is read at *its own*
   t<sub>i</sub> = Ω<sub>i</sub>(φ), so LTD lines up with LTD regardless of the
   clips' different periods. Joint quaternions are blended by weighted slerp.
5. **Root from the planted foot**, not by integrating the command: the support
   foot's pose is anchored at touchdown and the root is solved to keep it there,
   transferring to the leading foot across double support. So *achieved* velocity
   is whatever the blended pose genuinely produces, shown next to the command.
6. **Velocity feedback** (Sec 7.1, toggleable): a slow integral term nudges the
   sampler input until achieved matches commanded.

The panel shows commanded vs achieved velocity, the live sampler input, the nine
sample weights, the blended period and event times, a Fig. 9 blend-space diagram
with the input marker, per-joint angles against the safe envelope, and swing-foot
ground clearance.

### The honest caveat

This blends **poses**; the paper blends gait **parameters** and re-runs the
planner. That difference is measurable and the viewer surfaces it rather than
hiding it:

| | result |
|---|---|
| at each of the 9 key velocities | that sample gets weight 1.000 — the blend reproduces the clip exactly |
| Σ weights | 1 ± 2e−16 over 1377 sampled commands |
| weight continuity | ≤ 0.028 change per step around a closed command loop (no seam pops) |
| planted-foot slip, steady state | ≤ 1.5 mm (exactly 0 in single support; all of it is the double-support transfer) |
| planted-foot slip, during a hard command change | up to ~3.5 cm for one step |
| planted-foot ground penetration | 0.00 mm — exact by construction of the root solve |
| **swing-foot** clearance | dips to −3.7 mm on forward+turn, −9.2 mm on reverse+turn |
| velocity tracking, single axis | within 0.005 m/s and 0.01 rad/s |
| velocity tracking, ẋ between samples | 0.237 → 0.250 m/s with feedback on (exact) |
| velocity tracking, forward+turn together | 0.25 commanded → 0.10 achieved; saturates |
| joint use | ≤ 89% of the safe envelope across all tested commands |

The negative swing clearance is the contact constraint not being preserved —
precisely the reason the paper blends parameters. The forward+turn shortfall is
Eq. 10 working as designed (η halves the turn contribution at ẋ<sub>max</sub>);
feedback can't recover it because pushing either axis harder suppresses the
other. Both are worth knowing if `stylized_gait` ever grows a pose-blending
fast path — the parameter-space route in the library will not have either.

`blend_preview.png` shows four blended states with their active weights.

## Interactive viewer, parameter space

`gait_param_blend.html` — the same keyboard-driven viewer, the same robot
geometry and renderer, but blending **Π instead of poses**. Nothing in it is a
baked animation: the browser runs the paper's planner and the leg IK every frame.

Per frame:

1. keys → velocity command, rate-limited
2. weights from Eqs 8–10 (unchanged)
3. **`blendSamples`** — alpha-blend the Tab. 1 intervals and Sec. 4.3
   attributes, and wrap every Tab. 2 curve so that evaluating it at a time t of
   the blended cycle maps t → φ with the blended Ω, then back into each source's
   own cycle time. The result is a valid `GaitSample`.
4. **planner (Sec. 4.2)** — footholds sampled from the predicted path frame at
   the midpoint of each upcoming contact interval, re-evaluated every frame
   during swing and latched at touchdown; nominal swing interpolation with
   `f_soft`; stylized swing curves; the heel-toe rig with its automated support
   transitions
5. **IK** — 5-DoF damped least squares per leg, driven from the sole point, with
   the contact-height correction loop
6. render

`gait_params.json` (23 KB) carries the raw sample parameters — intervals,
attributes, and the knots/values/tangents of all 11 curve families per sample —
which the pose viewer never needed.

Measured on the built page, core extracted and simulated in Node:

| | pose blending | parameter blending |
|---|---|---|
| ẋ 0.25 (between samples) | 0.237 → 0.250 only with feedback | **0.250 exact, open loop** |
| ẋ 0.25 + θ̇ 0.35 together | 0.10 achieved (saturates) | **0.250 / 0.350 exact** |
| worst velocity error, 10 commands | 0.147 | **0.0004** |
| swing-foot clearance | −0.24 to −11.12 mm | **≥ −0.00 mm** |
| planted-foot slip | ≤ 1.5 mm steady, ~3.5 cm on a hard command change | **≤ 0.38 mm, no transient** |
| IK residual | n/a (no IK) | **0.000 mm** |
| joint use | ≤ 89% | ≤ 89% |
| needs Sec. 7.1 feedback | yes | no |

### The one corner that saturates

Sweeping the sticks to the extremes found a genuine limit. Every *forward*
combination is comfortable — full ẋ with full ẏ reaches 83% of the joint
envelope, full ẋ with full θ̇ 89%, all three at once 87%, all with 0.000 mm IK
residual. But **full reverse with full strafe and full turn together**
(−0.35, −0.16, −0.60) drives the right ankle to 100% and leaves a 2.25 mm IK
residual: the planner is asking for a foot pose the leg cannot reach.

That is exactly the failure mode parameter blending is supposed to expose, so
the viewer peak-holds the IK residual and calls it out in words when it goes
above 1 mm rather than silently clipping. Reverse walking already sits at 89% on
its own (the ankle is this robot's tightest joint), so there is little headroom
left to add turning to it.

Velocity tracking is exact because the path frame *is* the integral of the
command — the planner places footholds to match, rather than the motion emerging
from averaged angles. There is no velocity-feedback toggle because there is
nothing to correct. The failure mode moves to the IK residual: if a blended gait
ever asked for a foot pose the legs could not reach, that number would rise and
the joint bars would saturate. Across every command tested it stays at 0.000 mm.

Both viewers still reproduce each key velocity's own sample at weight 1.000, and
the JS planner matches the Python planner to within 0.73° per joint at the
sample velocities.

`param_blend_preview.png` shows four blended states with their weights.

## Blender round-trip (USD)

```
samples.py ──► blender_io.py export ──► gait_edit/<sample>.usda
                                            │  open in Blender, tune, export .usda
                                            ▼
              blender_io.py import  ◄── gait_edit/<sample>.usda
                       │
                       ▼
              gait_library.json ──► gait_blend.py / generate.py / export_isaac_motion.py
                                     (never touches USD again)
```

```
python3 blender_io.py export          # 9 files, ~800 KB each
python3 blender_io.py import          # -> gait_library.json (204 KB)
python3 generate.py --library gait_library.json
```

### What is in the edit file

`/Gait/Rig` is the editable part — the Tab. 2 curves as transform animation,
because transforms are the one thing a DCC round-trips reliably:

| prim | carries |
|---|---|
| `Rig/Pelvis` | translate z = absolute pelvis height, rotateXYZ = roll / pitch / yaw |
| `Rig/{Left,Right}Foot` | translate = swing x/y/z offset, rotateXYZ = γ roll and θ yaw |
| `…/HeelPitch` | rotateXYZ.y = ψ_heel, pivoted at the heel point |
| `…/HeelPitch/ToePitch` | rotateXYZ.y = ψ_toe, pivoted at the toe |

That heel/toe pair is the paper's Fig. 5 reverse-foot-lock rig split into two
nested pivots, so an animator can grab them like bones. Rig children are
authored in the character frame (x fwd, y left, z up, cm); the `Rig` prim itself
carries the transform onto the robot's axes so everything lines up.

`/Gait/Reference` is the skinned robot — all 23 bodies rigidly bound to the
skeleton — for looking at while you work. It does **not** update as you edit,
because Blender cannot re-run the planner and the IK; regenerate to refresh it.

### What travels, and what deliberately does not

You asked that only what is needed should be in the file. Working through what
the planner actually reads:

* **Dropped** — `swing_height`, `swing_x_lead`, `swing_y_lead`, `foot_yaw_amp`,
  `pelvis_bounce`, `pelvis_roll`, `pelvis_pitch`, and the interval fractions.
  Every one of these exists only to *construct* a curve that now travels as
  animation, so carrying them would create a second, stale source of truth.
* **Derived on import** — `psi_toe_lo` and `psi_heel_td` are read off the
  ψ curves at the liftoff / touchdown ends, and `pelvis_z_nom` is folded into
  the pelvis z curve, which is written absolute. This is what an animator would
  expect: raise the toe-off in the curve and the support-phase ramp follows.
  These are now **per side**, so an asymmetric edit stays asymmetric.
* **Carried** — the nine things not recoverable from the curves: `vel`,
  `T`, the four event times, the three stance attributes, `dt_psi_lo/td`.

### Verified

| check | result |
|---|---|
| export → import → plan, vs the original sample | 0.143° max joint difference (resampling at 60 fps) |
| edit landed (swing apex ×1.40, toe-off ×1.30) | 4.80 → 6.72 cm, 16.0 → 20.8° |
| asymmetric edit stays asymmetric | ψ_toe came back `{L: 20.8°, R: 16.0°}` |
| pelvis rewritten as a baked `xformOp:transform` matrix | decoded identically — nominal height and roll range unchanged |
| planner on an edited library sample | IK residual 0.0000 mm, swing clearance −0.000 mm, joint 94% |
| parameter blending two library samples | IK 0.0000 mm, clearance −0.000 mm, ψ_toe blends per side |
| `generate.py --library` → `verify.py` | all nine clips pass every check |

### Backends

Reading uses **pxr** when it is importable, which is what makes the round-trip
robust: `UsdGeom.Xformable.GetLocalTransformation` composes whatever xform ops
Blender wrote — translate + rotateXYZ, a baked `xformOp:transform`, an `orient`
quaternion, in any order — into one matrix that gets decomposed. It also reads
binary `.usdc`, which is Blender's default export format.

```
python3 blender_io.py import                 # pxr if available, else ascii
python3 blender_io.py import --backend pxr   # force
python3 blender_io.py validate gait_edit/*.usda
python3 blender_io.py selftest gait_edit/*.usda   # pxr vs ascii, needs both
python3 blender_io.py usdc gait_edit/*.usda       # re-save as binary
```

`usda_min.py` remains as a fallback for environments without pxr. It reads
`.usda` only, and only the op forms we write. `selftest` compares the two
readers on the same file and should report agreement below 1e−6.

`validate` opens a layer through pxr and checks the things a hand-written writer
can get wrong: stage up-axis / metersPerUnit / time range, that every rig prim
exists and its local transform composes, and that the `Reference` block resolves
as a real `UsdSkelRoot` with skinned meshes and an animation source.

### What has and has not been executed

**Not run here:** everything on the pxr path — `_import_pxr`, `validate`,
`to_usdc`, `selftest`. This sandbox has no package index (PyPI and
files.pythonhosted.org both return 403 through the proxy) and `usd-core` is not
vendored, so `pxr` could not be installed. That code is written against stable,
conventional API but it is unexecuted; `blender_io.py selftest` and
`blender_io.py validate` are there so you can shake it out in one command.

**Run here:** the ASCII path end to end — export, a simulated Blender edit
(including the baked-matrix form), import, library build, planning, blending,
and clip regeneration through `verify.py`. Those are the numbers in the table
above.

**Also untested:** real Blender itself. The round-trip was exercised by
rewriting the USD in code, not by opening it in Blender.

## Hand tuning

`gait_editor.html` — the paper's Sec. 6 editor. Pick a sample, watch it walk at
its own velocity, and tune it; the planner and the IK re-run on every change, so
the constraints are re-imposed rather than assumed.

* **Attributes and timing** as sliders: stance width / x-bias / splay, swing
  height, swing x/y lead, foot yaw, toe-off and heel-strike angles and their
  ramps, pelvis height / bounce / lean / pitch, cycle period, and the three
  contact event fractions. Click a label to revert that one value.
* **Function curves** with draggable knots and tangent handles, one at a time
  from a dropdown. Interior knots move in time as well as value.
* **Constraint guards are built into the UI**, which is the point of editing Π
  rather than the output: swing-curve endpoints are pinned at zero, ψ_toe is
  clamped ≥ 0 and ψ_heel ≤ 0, periodic pelvis curves keep their ends tied
  together, and the event ordering is validated. The live read-outs (swing
  clearance, planted-foot slip, IK residual, joint bars) tell you immediately if
  an edit has gone past what the legs can do.
* **Blend preview tab** — switch to gamepad/keyboard driving with the edited
  samples, to check a tuned clip still behaves when it is mixed with others.

Edits export as a **sparse** `gait_overrides.json`: only what you touched, so it
diffs cleanly and any single tweak can be reverted by deleting a line.
`samples.py` stays the authored baseline and is never rewritten.

```
python3 generate.py [gait_overrides.json]   # picks up ./gait_overrides.json by default
python3 export_params.py gait_overrides.json
python3 export_isaac_motion.py              # then the Isaac .npz
```

Curve knots are stored in **normalised interval coordinates** (x in 0..1 across
the curve's own LS / RS / CY span, tangents per unit x), so changing the cycle
period or a contact event rescales an edited curve instead of stranding it at
the old absolute times.

`gait_overrides.example.json` is the file used for the round-trip test below;
`gait_overrides.json` ships empty.

### Verified round-trip

The browser and the generator must agree, or tuning by eye is worthless. Edits
made in the editor were exported, fed to `gait_overrides.py`, regenerated in
Python, and compared against the browser's live state at matched phase:

| case | max joint disagreement |
|---|---|
| no edits (control) | 0.017° |
| five attribute edits, incl. cycle period 1.00 → 0.94 s | 0.059° |
| two curve edits (swing apex + periodic pelvis roll) | 0.027° |

and the regenerated clips still pass every check in `verify.py`, including the
edited one (`forward_slow` went from 61% to 87% of the joint envelope with a
7 cm swing apex, and stayed valid).

**A bug this caught.** My first equivalence check between the browser's curve
builder and the generator reported a clean match, but it was comparing with
`if (diff > worst)` — which is false for `NaN`. `_authored_args` only captured
the kwargs `samples.py` passes explicitly, so defaulted ones (`swing_y_lead`,
`foot_yaw_amp`) never reached the browser, became `undefined`, and poisoned
those curves. The round-trip test exposed it as a 26° disagreement. Both sides
now fill defaults explicitly, `buildCurves` throws on a non-finite attribute,
and the equivalence check counts non-finite comparisons rather than ignoring
them. The `.usda` clips were never affected — the bug lived only in the
browser-facing export path.

## Parameter blending vs pose blending

`blend_param_vs_pose.png` (from `blend_compare.py` + `blend_compare_fig.py`) runs
both methods on the same source samples, the same weights and the same phase-map
registration, so the *only* difference is which quantity gets interpolated.

`BlendedSample` implements the paper's Sec. 5.2 properly: alpha-blended time
intervals and attributes, and function curves evaluated through the registration
(for a time t of the blended cycle, map t → φ, map φ back into each source
cycle's own time, combine). It quacks like a `GaitSample`, so the unmodified
planner runs on it.

Worst swing-foot clearance over the cycle, α = 0.5:

| blend | parameter | pose |
|---|---|---|
| forward_slow × forward_fast (adjacent ẋ pair) | −0.00 mm | −0.24 mm |
| forward_fast × left_strafe | −0.00 mm | −2.59 mm |
| forward_fast × left_turn | −0.00 mm | −3.81 mm |
| reverse_fast × left_turn | −0.00 mm | −11.12 mm |

Parameter blending is exact at every blend, because the swing curve is
*value-clamped to zero at the interval endpoints* — a linear property of the
control points, so any convex combination inherits it. Pose blending degrades in
proportion to how different the two samples are: foot height is `FK(q)`, and
`lerp(q_A, q_B) ≠ FK⁻¹(lerp(FK(q_A), FK(q_B)))`.

Note the first row is the benign case, and it is also the only kind of blend the
ẋ axis ever performs — the three-step scheme only ever mixes *nearest neighbours*
on that axis. The large errors come from mixing a forward/reverse sample with a
turn/strafe one, which α_turn-strafe does routinely.

## Regenerating

```
python3 generate.py      # writes the nine .usda files
python3 verify.py .      # re-parses and validates them
python3 plot.py          # overview figure
python3 make_viewer.py   # rebuilds gait_viewer.html
python3 make_blend_viewer.py   # rebuilds gait_blend.html
python3 blend_compare_fig.py   # parameter-vs-pose comparison figure
python3 export_params.py       # gait_params.json for the parameter viewer
python3 make_param_viewer.py   # rebuilds gait_param_blend.html
python3 make_editor.py         # rebuilds gait_editor.html
```

`gait_core.py` (kinematics, splines, phase map, sample parameterisation),
`gait_planner.py` (footstep plan, swing/pelvis trajectories, heel-toe rig, IK)
and `samples.py` (the nine Table 3 samples) follow the paper's structure, so
they should drop alongside the existing `stylized_gait` modules.
