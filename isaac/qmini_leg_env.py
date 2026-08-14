from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from droidplayground.assets.qmini import QMINI_CFG

from .motion_player import MotionPlayer

# Reference clip, generated from the paper's gait sampler and exported by
# export_isaac_motion.py.  Sits next to this module.
MOTION_FILE = os.path.join(os.path.dirname(__file__), "qmini_step_in_place.npz")

# Column order of the clip == the USD PhysicsRevoluteJoint names.  The
# articulation is asked for these by name, so PhysX's own DoF ordering (which is
# a tree traversal, not the USD declaration order) cannot silently scramble them.
MOTION_JOINTS = [
    "Revolute_left_yaw", "Revolute_left_roll", "Revolute_left_pitch",
    "Revolute_left_knee", "Revolute_left_ankle",
    "Revolute_right_yaw", "Revolute_right_roll", "Revolute_right_pitch",
    "Revolute_right_knee", "Revolute_right_ankle",
]


@configclass
class QminiLegEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 5.0            # ~4.5 cycles of the 1.10 s clip

    # - spaces definition
    #   action  : residual on top of the reference, one per driven joint
    #   obs     : joint_pos(10) + joint_vel(10) + sin/cos(phi)(2) + reference(10)
    action_space = 10
    observation_space = 32
    state_space = 0
    action_scale = 0.15               # rad of authority around the reference

    # reference motion
    motion_file: str = MOTION_FILE
    motion_joints: tuple = tuple(MOTION_JOINTS)

    # Hold the base on the clip's own root trajectory.  The clip is a kinematic
    # reference with no balance controller, so a free base just falls over and
    # the tracking test tells you nothing.  Set False once you add balance.
    kinematic_base: bool = True

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.0, replicate_physics=True)


class QminiLegEnv(DirectRLEnv):
    cfg: QminiLegEnvCfg

    def __init__(self, cfg: QminiLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print("\nRobot joints:")
        for joint_id, joint_name in enumerate(self.robot.joint_names):
            print(f"  {joint_id:2d}: {joint_name}")

        # ---- resolve the clip's joints against the articulation, by name ----
        wanted = list(self.cfg.motion_joints)
        try:
            self.motion_joint_ids, resolved = self.robot.find_joints(
                wanted, preserve_order=True)
        except Exception as exc:
            raise RuntimeError(
                f"could not resolve the reference joints {wanted} on this "
                f"articulation. It has: {self.robot.joint_names}") from exc
        if len(self.motion_joint_ids) != len(wanted):
            raise RuntimeError(
                f"expected {len(wanted)} joints, matched {len(resolved)}: {resolved}")

        self.motion = MotionPlayer.from_npz(
            self.cfg.motion_file, device=self.device, joint_order=resolved)

        print(f"\nReference motion: {self.motion.meta['source']} "
              f"({self.motion.period:.3f} s, {self.motion.fps} fps)")
        print(f"  {self.motion.meta['convention']}")
        print("  articulation DoF index per reference column: "
              f"{list(self.motion_joint_ids)}")

        # ---- sanity checks -------------------------------------------------
        # (a) does the reference fit the articulation's own limits?  Catches a
        #     gross mapping error, though the limits are wide enough here that
        #     it would not catch a uniform sign flip on its own.
        lim = self.robot.data.soft_joint_pos_limits[0, self.motion_joint_ids]
        report, bad = self.motion.limit_report(lim[:, 0], lim[:, 1], resolved)
        print("\nReference vs articulation joint limits:")
        print(report)
        if bad:
            print("  !! the reference leaves the joint limits -- check the joint "
                  "mapping before trusting the results")

        # (b) sharper: a symmetric gait obeys q_right(t+T/2) = -q_left(t) once
        #     the PhysX sign convention is applied.  If the sign map or the
        #     column order were wrong, this residual blows up.
        stored = self.motion.meta.get("antisymmetry_deg")
        live = self.motion.symmetry_residual()
        if stored is not None and live is not None:
            print(f"\nHalf-cycle antisymmetry: {live:.4f} deg "
                  f"(exported {stored:.4f} deg)")
            if stored < 0.05 and live > 0.05:
                raise RuntimeError(
                    "the reference lost its left/right antisymmetry during "
                    "loading -- the joint mapping or sign convention is wrong")

        self.motion_time = torch.zeros(self.num_envs, device=self.device,
                                       dtype=torch.float32)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.motion_time = (self.motion_time + self.step_dt) % self.motion.period

    def _apply_action(self) -> None:
        reference = self.motion.sample(self.motion_time)
        position_targets = reference + self.cfg.action_scale * self.actions

        self.robot.set_joint_position_target(
            position_targets, joint_ids=self.motion_joint_ids)

        if self.cfg.kinematic_base:
            self._drive_base()

    def _drive_base(self) -> None:
        """Pin the base to the clip's root trajectory (animation playback)."""
        root_pos, root_quat = self.motion.sample_root(self.motion_time)
        pose = torch.cat([root_pos + self.scene.env_origins, root_quat], dim=-1)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((self.num_envs, 6), device=self.device))

    # ----------------------------------------------------------- observations
    def _get_observations(self):
        reference = self.motion.sample(self.motion_time)
        phase = self.motion.sample_phase(self.motion_time) * (2.0 * math.pi)

        observations = torch.cat(
            (
                self.robot.data.joint_pos[:, self.motion_joint_ids],   # 10
                self.robot.data.joint_vel[:, self.motion_joint_ids],   # 10
                torch.sin(phase).unsqueeze(-1),                        # 1
                torch.cos(phase).unsqueeze(-1),                        # 1
                reference,                                             # 10
            ),
            dim=-1,
        )
        return {"policy": observations}

    # --------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        reference = self.motion.sample(self.motion_time)
        reference_velocity = self.motion.sample_vel(self.motion_time)

        joint_pos = self.robot.data.joint_pos[:, self.motion_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.motion_joint_ids]

        error = joint_pos - reference
        velocity_error = joint_vel - reference_velocity

        tracking_reward = torch.exp(-5.0 * torch.sum(error ** 2, dim=1))
        velocity_reward = torch.exp(-0.05 * torch.sum(velocity_error ** 2, dim=1))
        effort_penalty = 0.001 * torch.sum(self.actions ** 2, dim=1)

        reward = 2.0 * tracking_reward + velocity_reward - effort_penalty

        # ---- logging -------------------------------------------------------
        deg = torch.rad2deg
        log = {
            "tracking/reward": reward.mean(),
            "tracking/mean_abs_error_deg": deg(error.abs().mean()),
            "tracking/max_abs_error_deg": deg(error.abs().max()),
        }
        for i, name in enumerate(self.motion.joint_names):
            short = name.replace("Revolute_", "")
            log[f"error/{short}"] = deg(error[:, i].abs().mean())
        self.extras["log"] = log

        if self.common_step_counter % 100 == 0:
            ref = deg(reference[0]).cpu().numpy()
            act = deg(joint_pos[0]).cpu().numpy()
            print(f"\n---- step {self.common_step_counter}  "
                  f"t={self.motion_time[0]:.2f}s  "
                  f"phi={self.motion.sample_phase(self.motion_time)[0]:.3f} ----")
            for i, name in enumerate(self.motion.joint_names):
                print(f"  {name.replace('Revolute_',''):16s} "
                      f"ref {ref[i]:7.2f}   act {act[i]:7.2f}   "
                      f"err {act[i]-ref[i]:6.2f}")

        return reward

    # ----------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ----------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # start each env at a random point in the cycle
        self.motion_time[env_ids] = torch.rand(
            len(env_ids), device=self.device) * self.motion.period

        # and put the robot on the reference pose it is about to be asked to track
        reference = self.motion.sample(self.motion_time[env_ids])
        reference_velocity = self.motion.sample_vel(self.motion_time[env_ids])
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.motion_joint_ids] = reference
        joint_vel[:, self.motion_joint_ids] = reference_velocity
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root_pos, root_quat = self.motion.sample_root(self.motion_time[env_ids])
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = root_pos + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_quat
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
