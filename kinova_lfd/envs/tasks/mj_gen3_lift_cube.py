from __future__ import annotations

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.core.tensor_math import (
    quat_from_euler_xyz,
    sample_uniform,
    subtract_frame_transforms,
    tf_combine,
)
from kinova_lfd.envs.base.mujoco.assets.kinova_gen3.gen3_constants import get_gen3_robot_cfg
from kinova_lfd.envs.base.mujoco.configclass import configclass

from .mj_gen3 import MjGen3Env, MjGen3EnvCfg, validate_scene_entities


@configclass
class MjGen3LiftCubeEnvCfg(MjGen3EnvCfg):
    action_scale = 0.5

    # reward scales
    min_height = 0.04
    cube_reach_reward_scale = 1.0
    cube_reach_reward_std = 0.1
    cube_lift_reward_scale = 15.0
    cube_goal_reward_scale = 16.0
    cube_goal_reward_std = 0.3
    cube_goal_fine_grained_scale = 5.0
    cube_goal_fine_grained_std = 0.05
    action_rate_reward_scale = -0.0001
    joint_vel_reward_scale = -0.0001

    # Cube pose initial ranges
    # init_pos_x = [0.2, 0.6]
    init_pos_x = [0.5, 0.5]
    # init_pos_y = [-0.3, 0.3]
    init_pos_y = [0.1, 0.1]
    init_pos_z = [0.0, 0.0]
    init_roll = [0.0, 0.0]
    init_pitch = [0.0, 0.0]
    init_yaw = [0.0, 0.0]
    cube_init_ranges = [init_pos_x, init_pos_y, init_pos_z, init_roll, init_pitch, init_yaw]

    # Cube pose initial ranges
    goal_pos_x = [0.4, 0.6]
    goal_pos_y = [-0.25, 0.25]
    goal_pos_z = [0.25, 0.5]
    goal_roll = [0.0, 0.0]
    goal_pitch = [0.0, 0.0]
    goal_yaw = [0.0, 0.0]
    cube_goal_ranges = [goal_pos_x, goal_pos_y, goal_pos_z, goal_roll, goal_pitch, goal_yaw]


class MjGen3LiftCubeEnv(MjGen3Env):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: MjGen3LiftCubeEnvCfg

    def __init__(self, cfg: MjGen3LiftCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.prev_actions = torch.zeros_like(self.actions)
        self.cube_init_ranges = torch.tensor(cfg.cube_init_ranges, device=self.device)
        self.cube_goal_ranges = torch.tensor(cfg.cube_goal_ranges, device=self.device)
        self.cube_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.cube_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.cube_goal_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_goal_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_tcp_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_ee_pose_w = torch.zeros((self.num_envs, 7), device=self.device)

    def _setup_scene(self):
        def get_cube_spec(cube_size: float = 0.025, mass: float = 0.06) -> mujoco.MjSpec:
            """Create a cube object specification."""
            spec = mujoco.MjSpec()
            body = spec.worldbody.add_body(name="cube")
            body.add_freejoint(name="cube_joint")
            body.add_geom(
                name="cube_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(cube_size,) * 3,
                mass=mass,
                rgba=(0.2, 0.6, 0.9, 1.0),
            )
            spec.worldbody.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(2, 2, 0.1),
                pos=(0, 0, 0),
                rgba=(0.8, 0.8, 0.8, 1.0),
            )

            return spec

        self.cfg.scene.entities = {"robot": get_gen3_robot_cfg(), "cube": EntityCfg(spec_fn=get_cube_spec)}
        validate_scene_entities(self.cfg.scene, type(self).__name__)

        # Initialize scene and simulation.
        self.scene = Scene(self.cfg.scene, device=self.cfg.sim.device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=self.cfg.sim.device,
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )

        # Get the robot from the scene
        self._robot = self.scene.entities["robot"]
        self._cube = self.scene.entities["cube"]
        self._cubes = {"cube": self._cube}
        self._tabletop_camera = self.scene.sensors["tabletop_camera"]
        self._wrist_camera = self.scene.sensors["wrist_camera"]

    def _pre_physics_step(self, actions: torch.Tensor, action_spec: ActionSpec = None) -> None:
        del action_spec
        self.actions = actions.clone().clamp(-5.0, 5.0)
        self.robot_dof_targets = torch.clamp(self.actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.robot_dof_targets[:, -1] = (
            self.robot_dof_targets[:, -1] / (self.robot_dof_upper_limits[-1:] - self.robot_dof_lower_limits[-1:])
            + self.robot_dof_lower_limits[-1:]
        ) * 255

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets[:, :-1], joint_ids=self.cfg.joint_ids[:-1])
        self.robot.set_tendon_len_target(self.robot_dof_targets[:, -1:], tendon_ids=[0])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate when the cube falls or the episode reaches its time limit."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = (self.cube_pose_w[:, 2] < -0.05) | torch.isnan(self.robot.data.body_link_pose_w).any(dim=(1, 2))
        return terminated.to(torch.bool), truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        return torch.nan_to_num(
            compute_rewards(
                self.actions,
                self.prev_actions,
                self.robot.data.joint_pos[:, self.cfg.joint_ids],
                self.robot.data.joint_vel[:, self.cfg.joint_ids],
                self.robot_tcp_pose_w[:, 0:3],
                self.cube_pose_w[:, 0:3],
                self.cube_goal_pose_w[:, 0:3],
                self.cfg.min_height,
                self.cfg.cube_reach_reward_scale,
                self.cfg.cube_reach_reward_std,
                self.cfg.cube_lift_reward_scale,
                self.cfg.cube_goal_reward_scale,
                self.cfg.cube_goal_reward_std,
                self.cfg.cube_goal_fine_grained_scale,
                self.cfg.cube_goal_fine_grained_std,
                self.cfg.action_rate_reward_scale,
                self.cfg.joint_vel_reward_scale,
            ),
            nan=0.0,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super(MjGen3Env, self)._reset_idx(env_ids)
        joint_pos = (
            self.robot.data.default_joint_pos[env_ids]
            + sample_uniform(-0.125, 0.125, (len(env_ids), self.robot.num_joints), self.device)
        )[:, self.cfg.joint_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)[env_ids]
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids.unsqueeze(-1), joint_ids=self.cfg.joint_ids)
        self.robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids, joint_ids=self.cfg.joint_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
        self._reset_cube_pose(env_ids)
        self._cube.write_root_link_pose_to_sim(self.cube_pose_w[env_ids], env_ids)
        self._cube.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
        self._compute_intermediate_values(env_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = torch.nan_to_num(
            torch.cat(
                (
                    self.robot.data.joint_pos[:, self.cfg.joint_ids],
                    self.robot.data.joint_vel[:, self.cfg.joint_ids],
                    self.prev_actions,
                    self.cube_pose_b[:, 0:3],
                    self.cube_goal_xyz_b[:, 0:3],
                ),
                dim=-1,
            ),
            nan=0.0,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.cube_pose_w[env_ids] = self._cube.data.root_link_pose_w[env_ids]
        self.robot_ee_pose_w[env_ids] = self.robot.data.body_link_pose_w[env_ids, self.ee_link_idx]
        self.prev_actions[env_ids] = self.actions[env_ids].clone()

        cube_pos_b, cube_quat_b = subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids],
            self.robot.data.root_link_quat_w[env_ids],
            self.cube_pose_w[env_ids, 0:3],
        )
        self.cube_pose_b[env_ids] = torch.cat((cube_pos_b, cube_quat_b), dim=-1)
        robot_tcp_quat_w, robot_tcp_pos_w = tf_combine(
            self.robot_ee_pose_w[env_ids, 3:7],
            self.robot_ee_pose_w[env_ids, 0:3],
            self.tcp_offset[env_ids, 3:7],
            self.tcp_offset[env_ids, 0:3],
        )
        self.robot_tcp_pose_w[env_ids] = torch.cat((robot_tcp_pos_w, robot_tcp_quat_w), dim=-1)

    def _reset_cube_pose(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        base_pos_w = self.robot.data.root_link_pos_w[env_ids]
        base_quat_w = self.robot.data.root_link_quat_w[env_ids]
        self.cube_xyz_b[env_ids] = self.cube_init_ranges[:, 0] + (
            self.cube_init_ranges[:, 1] - self.cube_init_ranges[:, 0]
        ) * torch.rand((len(env_ids), len(self.cube_init_ranges)), device=self.device)
        cube_quat_b = quat_from_euler_xyz(
            self.cube_xyz_b[env_ids, 3],
            self.cube_xyz_b[env_ids, 4],
            self.cube_xyz_b[env_ids, 5],
        )
        self.cube_pose_b[env_ids] = torch.cat((self.cube_xyz_b[env_ids, :3], cube_quat_b), dim=-1)
        cube_quat_w, cube_pos_w = tf_combine(base_quat_w, base_pos_w, cube_quat_b, self.cube_xyz_b[env_ids, :3])
        self.cube_pose_w[env_ids] = torch.cat((cube_pos_w, cube_quat_w), dim=-1)
        self.cube_goal_xyz_b[env_ids] = self.cube_goal_ranges[:, 0] + (
            self.cube_goal_ranges[:, 1] - self.cube_goal_ranges[:, 0]
        ) * torch.rand((len(env_ids), len(self.cube_goal_ranges)), device=self.device)
        cube_goal_quat_b = quat_from_euler_xyz(
            self.cube_goal_xyz_b[env_ids, 3],
            self.cube_goal_xyz_b[env_ids, 4],
            self.cube_goal_xyz_b[env_ids, 5],
        )
        cube_goal_quat_w, cube_goal_pos_w = tf_combine(
            base_quat_w,
            base_pos_w,
            cube_goal_quat_b,
            self.cube_goal_xyz_b[env_ids, :3],
        )
        self.cube_goal_pose_w[env_ids] = torch.cat((cube_goal_pos_w, cube_goal_quat_w), dim=-1)


@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tcp_pos_w: torch.Tensor,
    cube_pos_w: torch.Tensor,
    cube_goal_pos_w: torch.Tensor,
    min_height: float,
    cube_reach_reward_scale: float,
    cube_reach_reward_std: float,
    cube_lift_reward_scale: float,
    cube_goal_reward_scale: float,
    cube_goal_reward_std: float,
    cube_goal_fine_grained_scale: float,
    cube_goal_fine_grained_std: float,
    action_rate_reward_scale: float,
    joint_vel_reward_scale: float,
) -> torch.Tensor:
    cube_tcp_distance = torch.norm(cube_pos_w - tcp_pos_w, dim=1)
    cube_tcp_distance_reward = cube_reach_reward_scale * (1 - torch.tanh(cube_tcp_distance / cube_reach_reward_std))
    cube_lifted_reward = (
        cube_lift_reward_scale * torch.where(cube_pos_w[:, 2] > min_height, 1.0, 0.0) * (cube_tcp_distance < 0.1)
    )
    cube_distance = torch.norm(cube_goal_pos_w - cube_pos_w, dim=1)
    cube_dist_reward = (
        cube_goal_reward_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_reward_std))
    )
    cube_dist_reward_fine_grained = (
        cube_goal_fine_grained_scale
        * (cube_pos_w[:, 2] > min_height)
        * (1 - torch.tanh(cube_distance / cube_goal_fine_grained_std))
    )
    action_rate_reward = (action_rate_reward_scale * torch.sum(torch.square(actions - prev_actions), dim=1)).clamp(
        -1, 1
    )
    joint_vel_reward = (joint_vel_reward_scale * torch.sum(torch.square(joint_vel), dim=1)).clamp(-1, 1)
    return (
        cube_tcp_distance_reward
        + cube_lifted_reward
        + cube_dist_reward
        + cube_dist_reward_fine_grained
        + action_rate_reward
        + joint_vel_reward
    )
