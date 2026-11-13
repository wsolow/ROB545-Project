import math

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.sim import MujocoCfg
from mjlab.sim.sim import Simulation, SimulationCfg

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.core.mj_tensor_math import (
    quat_from_euler_xyz,
    subtract_frame_transforms,
    tf_combine,
)
from kinova_lfd.envs.base.mujoco.assets.kinova_gen3.gen3_constants import get_gen3_robot_cfg
from kinova_lfd.envs.base.mujoco.configclass import configclass
from kinova_lfd.envs.tasks.lift_cube_curriculum import LiftCubeCurriculumCfg, LiftCubeCurriculumManager

from .mj_gen3 import MjGen3Env, MjGen3EnvCfg, validate_scene_entities


@configclass
class MjGen3LiftCubeEnvCfg(MjGen3EnvCfg):
    enable_cameras = False

    episode_length_s = 3.0
    decimation = 2

    observation_space = 30

    # reward scales
    min_height = 0.06
    cube_reach_reward_scale = 1.0
    cube_reach_reward_std = 0.1
    cube_lift_reward_scale = 15.0
    cube_goal_reward_scale = 16.0
    cube_goal_reward_std = 0.3
    cube_goal_fine_grained_scale = 5.0
    cube_goal_fine_grained_std = 0.05
    action_rate_reward_scale = -0.0001
    joint_vel_reward_scale = -0.0001
    cube_goal_distance_threshold = 0.03
    curriculum = LiftCubeCurriculumCfg()

    # Keep the arm in a deterministic, camera-facing ready pose.  The final
    # entry is the gripper command and is repeated in ``joint_ids``.
    robot_init_joint_pos = [0.0, 0.25, -0.05, 1.60, 0.0, 1.20, -1.60, 0.0]

    # Cube pose initial ranges
    # Keep the cube centered in the wrist-camera workspace.  The z coordinate
    # is the cube center, so it must be half the cube size above the floor.
    init_pos_x = [0.45, 0.45]
    init_pos_y = [0.0, 0.0]
    init_pos_z = [0.025, 0.025]
    init_roll = [0.0, 0.0]
    init_pitch = [0.0, 0.0]
    init_yaw = [-3.141592653589793, 3.141592653589793]
    cube_init_ranges = [init_pos_x, init_pos_y, init_pos_z, init_roll, init_pitch, init_yaw]

    # The goal is vertically above the initial cube position.
    goal_height = 0.35

    sim = SimulationCfg(
        nconmax=55,
        njmax=600,
        mujoco=MujocoCfg(
            timestep=0.01,
            iterations=10,
            ls_iterations=20,
            impratio=10,
            cone="elliptic",
        ),
    )


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

        self.gripper_joint_idx = self.robot.find_joints("right_driver_joint")[0][0]
        self.left_gripper_joint_idx = self.robot.find_joints("left_driver_joint")[0][0]
        # Device index tensor for the gripper finger joints (see MjGen3Env for why not a Python list).
        self._gripper_joint_ids = torch.as_tensor(
            [self.gripper_joint_idx, self.left_gripper_joint_idx], device=self.device, dtype=torch.long
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self._camera_view_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cube_init_ranges = torch.tensor(cfg.cube_init_ranges, device=self.device)
        self.cube_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        # Internal pose buffers below hold mjlab-convention (w, x, y, z) quaternions; they are
        # written to / computed from the sim directly. Only their xyz parts are read outside.
        self.cube_pose_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.cube_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.cube_goal_xyz_b = torch.zeros((self.num_envs, 6), device=self.device)
        self.cube_goal_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.last_cube_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.last_cube_goal_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_tcp_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_ee_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.curriculum_manager = LiftCubeCurriculumManager(self.cfg.curriculum)

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

        if not self.cfg.enable_cameras:
            self.cfg.scene.sensors = ()

        # Initialize scene and simulation.
        self.scene = Scene(self.cfg.scene, device=self.cfg.sim.device if hasattr(self.cfg.sim, "device") else "cuda")
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=self.cfg.sim.device if hasattr(self.cfg.sim, "device") else "cuda",
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
        if self.cfg.enable_cameras:
            self._tabletop_camera = self.scene.sensors["tabletop_camera"]
            self._wrist_camera = self.scene.sensors["wrist_camera"]

    def _pre_physics_step(self, actions: torch.Tensor, action_spec: ActionSpec = None) -> None:
        del action_spec
        self.actions = actions.clone().clamp(-5.0, 5.0)
        arm_targets = torch.clamp(
            self.actions[:, :-1],
            self.robot_dof_lower_limits[:-1],
            self.robot_dof_upper_limits[:-1],
        )
        # The policy's final action is normalized: -1 is open and +1 is
        # closed.  The MuJoCo tendon actuator uses [0, 255].
        gripper_target = ((self.actions[:, -1:] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0
        self.robot_dof_targets = torch.cat((arm_targets, gripper_target), dim=-1)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets[:, :-1], joint_ids=self._arm_joint_ids)
        self.robot.set_tendon_len_target(self.robot_dof_targets[:, -1:], tendon_ids=self._gripper_tendon_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate when the cube falls or the episode reaches its time limit."""
        self.last_cube_pose_w.copy_(self.cube_pose_w)
        self.last_cube_goal_pose_w.copy_(self.cube_goal_pose_w)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        cube_goal_distance = torch.norm(self.cube_goal_pose_w[:, 0:3] - self.cube_pose_w[:, 0:3], dim=1)
        terminated = (
            (self.cube_pose_w[:, 2] < -0.05)
            | (cube_goal_distance <= self.cfg.cube_goal_distance_threshold)
            | torch.isnan(self.robot.data.body_link_pose_w).any(dim=(1, 2))
        )
        return terminated.to(torch.bool), truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        curriculum_values = self.curriculum_manager.values(self.common_step_counter)
        rewards = torch.nan_to_num(
            compute_rewards(
                self.actions,
                self.prev_actions,
                self.robot.data.joint_pos[:, self._arm_joint_ids],
                self.robot.data.joint_vel[:, self._arm_joint_ids],
                self.robot_tcp_pose_w[:, 0:3],
                self.cube_pose_w[:, 0:3],
                self.cube_goal_pose_w[:, 0:3],
                self.cfg.min_height,
                curriculum_values["cube_reach_reward_scale"],
                self.cfg.cube_reach_reward_std,
                curriculum_values["cube_lift_reward_scale"],
                curriculum_values["cube_goal_reward_scale"],
                self.cfg.cube_goal_reward_std,
                curriculum_values["cube_goal_fine_grained_scale"],
                self.cfg.cube_goal_fine_grained_std,
                curriculum_values["action_rate_reward_scale"],
                curriculum_values["joint_vel_reward_scale"],
            ),
            nan=0.0,
        )
        self.extras.setdefault("log", {}).update(self.curriculum_manager.log_values(self.common_step_counter))
        self.prev_actions[:] = self.actions
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super(MjGen3Env, self)._reset_idx(env_ids)
        # Do not randomize the arm pose: even small changes at the wrist can
        # move the cube outside the camera frustum.
        joint_pos = torch.as_tensor(self.cfg.robot_init_joint_pos, device=self.device).repeat(len(env_ids), 1)
        joint_pos = torch.clamp(
            joint_pos[:, :-1],
            self.robot_dof_lower_limits[:-1],
            self.robot_dof_upper_limits[:-1],
        )
        joint_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)[env_ids]
        # With tensor joint ids mjlab builds the (env, joint) outer index itself, so env_ids stays 1-D.
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids, joint_ids=self._arm_joint_ids)
        self.robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids, joint_ids=self._arm_joint_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim(
            torch.zeros((len(env_ids), 2), device=self.device),
            env_ids=env_ids,
            joint_ids=self._gripper_joint_ids,
        )
        self.prev_actions[env_ids] = 0.0
        self._reset_cube_pose(env_ids)
        self._cube.write_root_link_pose_to_sim(self.cube_pose_w[env_ids], env_ids)
        self._cube.write_root_link_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
        self._compute_intermediate_values(env_ids)
        self._camera_view_pending[env_ids] = True

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pending_env_ids = self._camera_view_pending.nonzero(as_tuple=False).squeeze(-1)
        if len(pending_env_ids) > 0:
            # Reset writes qpos first; by this point the environment has
            # already called sim.forward(), so camera transforms are current.
            self._compute_intermediate_values(pending_env_ids)
            self._validate_wrist_camera_view(pending_env_ids)
            self._camera_view_pending[pending_env_ids] = False

        arm_joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        arm_joint_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]
        gripper_pos = self.robot.data.joint_pos[:, self.gripper_joint_idx : self.gripper_joint_idx + 1] / 0.8
        gripper_vel = self.robot.data.joint_vel[:, self.gripper_joint_idx : self.gripper_joint_idx + 1]
        obs = torch.nan_to_num(
            torch.cat(
                (
                    arm_joint_pos,
                    gripper_pos,
                    arm_joint_vel,
                    gripper_vel,
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

    def _validate_wrist_camera_view(self, env_ids: torch.Tensor) -> None:
        """Validate that the fixed initial pose points the wrist camera at the cube."""
        camera_idx = mujoco.mj_name2id(self.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        camera_pos = torch.as_tensor(self.sim.data.cam_xpos[env_ids, camera_idx], device=self.device)
        camera_mat = torch.as_tensor(self.sim.data.cam_xmat[env_ids, camera_idx], device=self.device).reshape(-1, 3, 3)

        camera_forward_w = -camera_mat[:, :, 2]
        downward = camera_forward_w[:, 2] < -0.9

        cube_delta_w = self.cube_pose_w[env_ids, :3] - camera_pos
        cube_camera = torch.bmm(camera_mat.transpose(1, 2), cube_delta_w.unsqueeze(-1)).squeeze(-1)
        cube_depth = -cube_camera[:, 2]

        vertical_half_fov = math.radians(float(self.sim.mj_model.cam_fovy[camera_idx])) / 2.0
        horizontal_half_fov = math.atan(
            math.tan(vertical_half_fov)
            * float(self.sim.mj_model.cam_resolution[camera_idx, 0])
            / float(self.sim.mj_model.cam_resolution[camera_idx, 1])
        )
        in_frustum = (
            (cube_depth > 0.0)
            & (torch.abs(cube_camera[:, 0] / cube_depth) < math.tan(horizontal_half_fov) * 0.9)
            & (torch.abs(cube_camera[:, 1] / cube_depth) < math.tan(vertical_half_fov) * 0.9)
        )
        if not torch.all(downward & in_frustum):
            invalid_envs = env_ids[~(downward & in_frustum)].detach().cpu().tolist()
            raise RuntimeError(
                f"Wrist camera must face down with the initial cube in view; invalid envs: {invalid_envs}"
            )

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

        # Keep the goal directly above the sampled starting position, with no
        # roll, pitch, or yaw rotation.
        self.cube_goal_xyz_b[env_ids] = self.cube_xyz_b[env_ids]
        self.cube_goal_xyz_b[env_ids, 2] = self.cfg.goal_height
        self.cube_goal_xyz_b[env_ids, 3:] = 0.0
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
