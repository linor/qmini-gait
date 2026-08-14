# Step-in-place reference for `qmini_leg_env`

Replaces the hand-keyframed single-leg motion with the generated
**step-in-place** clip (Table 3, `ẋ = ẏ = θ̇ = 0`), driving all **10 leg joints**.

## Files

| file | what changed |
|---|---|
| `qmini_step_in_place.npz` | the clip: 67 frames @ 60 fps, T = 1.10 s |
| `motion_player.py` | keyframe constructor unchanged; adds `from_npz` + clip sampling |
| `qmini_leg_env.py` | 10 DoF, name-based joint resolution, reference-relative actions |

`export_isaac_motion.py` (in the generator folder) rebuilds the `.npz` from
`03_step_in_place.usda`.

## Two conversions that matter

**Sign.** A `UsdPhysics` revolute joint measures body1 relative to body0. On this
robot the right leg — plus the left ankle and left yaw — have body0/body1 swapped
relative to the parent→child direction, so PhysX reads those six joints with the
*opposite* sign to the skeleton the clips were authored in. The export derives
the ±1 per joint from the physics graph and bakes it in, so `joint_pos` is
already in PhysX convention.

This also corrects something I got wrong earlier. I had called the mirrored joint
limits a CAD-export artifact and clamped the animations to the intersection of
the two. They are not an artifact — they are the *same* limits expressed in
opposite conventions. Undo the sign and roll/pitch/knee/ankle are identical on
both legs, and only yaw is mirrored, which is correct for yaw:

| joint | left | right | |
|---|---|---|---|
| yaw | (−30, 45) | (−45, 30) | mirrored, as it should be |
| roll | (−15, 15) | (−15, 15) | identical |
| pitch | (−50, 50) | (−50, 50) | identical |
| knee | (−60, 50) | (−60, 50) | identical |
| ankle | (−40, 60) | (−40, 60) | identical |

So the clips have more headroom than I claimed: knee is (−60, 50) not ±50, ankle
(−40, 60) not ±40. The existing animations are valid, just conservative.

**Consequence for you:** because of the sign flip, a symmetric gait produces
*antisymmetric* joint targets between the legs. Left knee runs +7.7…+24.4°,
right knee −24.4…−7.7° for the same physical bend. Do not "fix" that.

## Joint ordering

PhysX orders DoFs by articulation-tree traversal, not USD declaration order, so
index 0 is very unlikely to be the hip. The env therefore resolves the ten joints
**by name** via `find_joints(..., preserve_order=True)` and reorders the clip
columns to match. Your existing code assumed `joint_pos[:, :3]` was
hip/knee/ankle — worth checking against the joint list the env prints at startup.

## Self-checks at startup

1. Reference vs the articulation's own joint limits. Catches a gross mapping
   error — but *not* a uniform sign flip, because the limits are wide relative to
   this gentle clip. Verified: negating every joint still fits.
2. **Half-cycle antisymmetry**, `q_right(t + T/2) = −q_left(t)`. Holds to
   0.0000° for this clip, and a wrong right-leg sign shows up as 48.8°. This is
   the check that actually bites; the env raises if it fails.

A pure left/right column swap is invisible to (2) — correctly, since for a
symmetric gait that is exactly a half-cycle time shift. Name-based resolution
rules it out anyway.

## Behaviour changes worth knowing

- **Amplitudes are much smaller.** The hand-keyed motion swung ±50°; the real
  step-in-place is hip_pitch −8…+2°, knee 8…24°, ankle −16…−10°, and hip_yaw is
  identically zero. Peak joint speed 201 °/s, 4.0° per 50 Hz control step.
- **Actions are residuals**: `target = reference + action_scale · action`, with
  `action_scale = 0.15` rad. The previous code drove `target = action` directly.
- **`kinematic_base = True`** pins the base to the clip's own root trajectory
  (which bounces 409→419 mm). The clip is a kinematic reference with no balance
  controller, so a free base just falls over and the tracking test tells you
  nothing. Set it `False` once you add balance.
- **Observation is 32**: `joint_pos(10) + joint_vel(10) + sin/cos φ(2) +
  reference(10)`. Drop the reference block for 22 if you would rather the policy
  infer it from phase.
- Resets place the robot *on* the reference pose at a random phase, rather than
  at the default pose.

## Sampling

The clip is a 60 Hz readout of an analytic C¹ spline, so `MotionPlayer`
interpolates it with a cyclic Catmull-Rom Hermite rather than linearly.
`sample_vel` is the analytic derivative of `sample`, so position and velocity
references are consistent — with the earlier linear sampler plus a central
difference they disagreed by up to 1.3 rad/s, which would have fought the
velocity reward.

`_shim/` is a numpy stand-in for torch used only to test `motion_player.py`
without Isaac Lab installed. It is not needed at runtime.
