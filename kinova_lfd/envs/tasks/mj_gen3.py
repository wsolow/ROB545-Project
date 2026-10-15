from __future__ import annotations

import time

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from PIL import Image

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.core.tensor_math import quat_from_matrix
from kinova_lfd.envs.base.mujoco.assets.kinova_gen3.gen3_constants import get_gen3_robot_cfg
from kinova_lfd.envs.base.mujoco.configclass import configclass
from kinova_lfd.envs.base.mujoco.controllers.differential_ik import MjDifferentialIk
from kinova_lfd.envs.base.mujoco.controllers.waypoints import WaypointCfg
from kinova_lfd.envs.base.mujoco.mj_env import MjDirectRlEnv, MjDirectRlEnvCfg


@configclass
class MjGen3EnvCfg(MjDirectRlEnvCfg):
    decimation = 2

    episode_length_s = 100.0
    action_space = 8
    observation_space = 8
    tool_site_name = "pinch_site"
    ee_link_name = "end_effector_link"
    base_link_name = "base_link"
    gripper_joint_names = ["right_driver_joint"]
    joint_ids = [0, 1, 2, 3, 4, 5, 6, 0]
    tcp_offset = [0.0, 0.0, 0.120, 1.0, 0.0, 0.0, 0.0]

    sim = SimulationCfg(njmax=300, nconmax=50, mujoco=MujocoCfg(integrator="implicitfast", gravity=(0, 0, -9.81)))

    scene = SceneCfg(
        num_envs=1,
        env_spacing=2,
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            textures=(),
            materials=(),
            # lights=(LightCfg(name="sun", pos=(0.5, -0.5, 1.5), type="directional", castshadow=False),),
        ),
        sensors=(
            CameraSensorCfg(
                name="tabletop_camera",
                width=640,
                height=480,
                pos=[0.4, -0.5, 0.3],
                quat=[0.8192, 0.5736, 0.0, 0.0],
                fovy=90,
                data_types=("rgb", "depth"),
                use_shadows=True,
                use_textures=True,
                clone_data=True,
            ),
            CameraSensorCfg(
                name="wrist_camera",
                camera_name="robot/wrist",
                width=640,
                height=480,
                data_types=("rgb", "depth"),
                use_shadows=True,
                use_textures=True,
                clone_data=True,
            ),
        ),
    )


class MjGen3Env(MjDirectRlEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    def __init__(self, cfg: MjGen3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.mujoco.timestep * self.cfg.decimation

        # Goal poses and end effector positions
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._current_prev_actions = torch.zeros_like(self.actions, device=self.device)

        # Limits and targets
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)[
            self.cfg.joint_ids
        ]
        self.robot_dof_lower_limits[self.robot_dof_lower_limits == -float("inf")] = -torch.pi
        self.robot_dof_upper_limits[self.robot_dof_upper_limits == float("inf")] = torch.pi

        self.robot_dof_targets = self.robot.data.default_joint_pos[:, self.cfg.joint_ids].clone()
        self._curr_gripper_goal = None

        self.ee_link_idx = self.robot.find_bodies(self.cfg.ee_link_name)[0][0]
        self.tool_site_idx = self.robot.find_sites(self.cfg.tool_site_name)[0][0]

        self.tcp_offset = torch.as_tensor(self.cfg.tcp_offset, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self._diff_ik = MjDifferentialIk(self)
        self._waypoints = WaypointCfg().build()

    def _setup_scene(self):

        def get_cube_spec(
            name: str = "red_block",
            rgba: tuple[int] = (0.8, 0.1, 0.1, 1.0),
            cube_size: float = 0.02,
            mass: float = 0.06,
        ) -> mujoco.MjSpec:
            """Create a cube object specification."""
            spec = mujoco.MjSpec()
            body = spec.worldbody.add_body(name=name)
            body.add_freejoint(name=f"{name}_joint")
            geom = body.add_geom(
                name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=(cube_size,) * 3, mass=mass, rgba=rgba, group=2
            )
            geom.solref = (0.005, 1.0)
            geom.solimp = (0.99, 0.9999, 0.001, 0.5, 2)
            return spec

        self.cfg.scene.entities = {
            "robot": get_gen3_robot_cfg(),
            "pink_block": EntityCfg(spec_fn=lambda: get_cube_spec(name="pink_block", rgba=(1.0, 0.4, 0.7, 1.0))),
        }

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
        self._cubes = {
            "pink_block": self.scene.entities["pink_block"],
        }
        self._robot = self.scene.entities["robot"]
        self._tabletop_camera = self.scene.sensors["tabletop_camera"]
        self._wrist_camera = self.scene.sensors["wrist_camera"]

    def _pre_physics_step(self, actions: torch.Tensor, action_spec: ActionSpec = None):
        if action_spec == ActionSpec.MOVE_TCP:
            # Obtain waypoints that should be followed by the arm
            next_action = self._waypoints.get_next_point(self._robot_tool_pose_b, actions[:, :6])
            arm_targets = self._joint_positions[:, self.cfg.joint_ids[:-1]] + self._diff_ik.compute_joint_vel(
                next_action, smoothing=False
            )
            targets = torch.cat((arm_targets, actions[:, -1:]), dim=1)
            self.actions = targets.clone()
        elif action_spec == ActionSpec.TWIST_TCP:
            twist = actions[:, :6]
            dq = self._diff_ik.compute_joint_pos(twist, smoothing=False)
            if self._diff_ik.active:
                arm_targets = self._joint_positions[:, self.cfg.joint_ids[:-1]] + dq
            else:
                arm_targets = self.robot_dof_targets[:, :-1]
            targets = torch.cat((arm_targets, actions[:, -1:]), dim=1)
            self.actions = targets.clone()

        elif action_spec in [ActionSpec.JOINTS_POS, ActionSpec.JOINTS_VEL]:
            self.actions = actions.clone()
        else:
            raise NotImplementedError
        self.robot_dof_targets = self.actions.clone()

    def _apply_action(self) -> None:
        """Send the action to the robot."""
        self.robot.set_tendon_len_target(
            torch.where(self.robot_dof_targets[:, -1:] > 0, 255, self.robot_dof_targets[:, -1:]),
            tendon_ids=self.cfg.joint_ids[-1:],
        )
        self.robot.set_joint_position_target(self.robot_dof_targets[:, :-1], joint_ids=self.cfg.joint_ids[:-1])

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate if max length is reached."""
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros((self.num_envs), dtype=torch.bool), truncated

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return the observations as a vector."""
        return {"policy": self._joint_positions}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if hasattr(self, "_diff_ik"):
            self._diff_ik.reset()
        # robot state
        joint_pos = (self.robot.data.default_joint_pos[env_ids])[:, self.cfg.joint_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids.unsqueeze(-1), joint_ids=self.cfg.joint_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.cfg.joint_ids)

        # Cubes
        self._init_cube_poses = {
            "pink_block": [0.35, 0, 0.021],
        }
        for cube in self._cubes:
            self._cubes[cube].write_root_link_pose_to_sim(
                torch.tensor([[*self._init_cube_poses[cube], 1, 0, 0, 0]], device=self.device).repeat(self.num_envs, 1),
                env_ids,
            )
            self._cubes[cube].write_root_link_velocity_to_sim(
                torch.zeros((len(env_ids), 6), device=self.device), env_ids
            )

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """Docstring for _compute_intermediate_values.

        :param self: Description
        :param env_ids: Description
        :type env_ids: torch.Tensor | None

        Compute vlaues in the world frame
        """

    def _rgbd_from_sensor(self, camera, camera_pose: torch.Tensor) -> dict:
        """Pack RGB, depth, intrinsics, and pose for one camera sensor."""
        height = camera.cfg.height
        width = camera.cfg.width
        fovy_deg = camera.cfg.fovy
        if fovy_deg is None:
            fovy_deg = float(self.sim.mj_model.cam_fovy[camera.camera_idx])

        fy = (height / 2.0) / np.tan(np.radians(fovy_deg / 2.0))
        fx = fy  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        intrinsic_k = torch.as_tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=self.device)

        camera_data = camera.data
        return {
            "rgb": camera_data.rgb.permute(0, 3, 1, 2),
            # (camera_data.depth.permute(0, 3, 1, 2) * 1000).to(torch.uint16)[:, 0] # TODO changing this here has downstream effects?
            "depth": camera_data.depth.permute(0, 3, 1, 2)[:, 0],
            "timestamp": time.perf_counter(),
            "camera_pose": camera_pose,
            "intrinsic_k": intrinsic_k,
        }

    def _get_latest_rgbd(self):
        """Grab the latest RGBD image from the Mujoco simulator.

        Returns:
        A dictionary containing:
          - ``rgb``: (H, W, 3) uint8 RGB image
          - ``depth``: (H, W) uint16 depth image
          - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
          - ``camera_pose``: 7D float64 array (x, y, z, qw, qx, qy, qz) in Isaac wxyz
          - ``timestamp``: float timestamp in seconds

        """
        latest = self._rgbd_from_sensor(
            self._tabletop_camera,
            torch.as_tensor(self._tabletop_camera.cfg.pos + self._tabletop_camera.cfg.quat, device=self.device),
        )
        # camera_data = self._tabletop_camera.data

        # def colorize_depth(depth: np.ndarray, min_depth=None, max_depth=None, colormap="plasma") -> Image.Image:
        #     """Colorize a depth image and return a PIL Image.

        #     Args:
        #         depth:     HxW float array (meters or raw units)
        #         min_depth: clip minimum (defaults to array min) in meters
        #         max_depth: clip maximum (defaults to array max) in meters
        #         colormap:  any matplotlib colormap name

        #     """
        #     import matplotlib.pyplot as plt

        #     if depth.dtype == np.uint16:
        #         depth = depth.astype(np.float32) / 1000

        #     min_d = min_depth if min_depth is not None else np.nanmin(depth)
        #     max_d = max_depth if max_depth is not None else np.nanmax(depth)

        #     # Normalize to [0, 1]
        #     normalized = (depth - min_d) / (max_d - min_d + 1e-8)
        #     normalized = np.clip(normalized, 0, 1)

        #     # Apply colormap → RGBA float array → uint8
        #     cmap = plt.get_cmap(colormap)
        #     colored = (cmap(normalized) * 255).astype(np.uint8)

        #     return Image.fromarray(colored.squeeze(), mode="RGBA").convert("RGB")

        # Image.fromarray(camera_data.rgb[0].cpu().numpy()).save("out.png")
        # colorize_depth(latest["depth"].cpu().numpy(), max_depth=1).save("depth.png")

        return latest

    def _get_latest_rgbd_wrist(self):
        """Grab the latest RGBD image from the wrist camera on the Gen3.

        Same dictionary as ``_get_latest_rgbd``. ``camera_pose`` is the wrist
        camera's current world pose, because that camera is fixed to the arm.
        """
        cam_idx = self._wrist_camera.camera_idx
        pos = torch.as_tensor(self.sim.data.cam_xpos[:, cam_idx], device=self.device, dtype=torch.float32)
        xmat = torch.as_tensor(self.sim.data.cam_xmat[:, cam_idx], device=self.device, dtype=torch.float32)
        pose = torch.cat((pos, quat_from_matrix(xmat)), dim=-1).squeeze(0)
        camera_data = self._wrist_camera.data
        latest = self._rgbd_from_sensor(self._wrist_camera, pose)

        def colorize_depth(depth: np.ndarray, min_depth=None, max_depth=None, colormap="plasma") -> Image.Image:
            """Colorize a depth image and return a PIL Image.

            Args:
                depth:     HxW float array (meters or raw units)
                min_depth: clip minimum (defaults to array min) in meters
                max_depth: clip maximum (defaults to array max) in meters
                colormap:  any matplotlib colormap name

            """
            import matplotlib.pyplot as plt

            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000

            min_d = min_depth if min_depth is not None else np.nanmin(depth)
            max_d = max_depth if max_depth is not None else np.nanmax(depth)

            # Normalize to [0, 1]
            normalized = (depth - min_d) / (max_d - min_d + 1e-8)
            normalized = np.clip(normalized, 0, 1)

            # Apply colormap → RGBA float array → uint8
            cmap = plt.get_cmap(colormap)
            colored = (cmap(normalized) * 255).astype(np.uint8)

            return Image.fromarray(colored.squeeze(), mode="RGBA").convert("RGB")

        Image.fromarray(camera_data.rgb[0].cpu().numpy()).save("out.png")
        colorize_depth(latest["depth"].cpu().numpy(), max_depth=1).save("depth.png")
        return latest

    @property
    def _robot_tool_pose_b(self) -> torch.Tensor:
        """Return the tool site of the robot."""
        return self.robot.data.site_pose_w[:, self.tool_site_idx]

    @property
    def _robot_tool_vel_b(self) -> torch.Tensor:
        """Return the tool site of the robot."""
        return self.robot.data.site_vel_w[:, self.tool_site_idx]

    @property
    def _robot_tool_wrench_b(self) -> torch.Tensor:
        """Return the tool wrench forces appleid to the robot."""
        return self.robot.data.body_external_wrench[:, self.ee_link_idx]

    @property
    def _n_objects(self) -> int:
        """Return the number of objects in the scene."""
        return len(self._cubes)

    @property
    def _object_names(self) -> list[str]:
        return list(self._cubes.keys())

    @property
    def _object_poses(self) -> torch.Tensor:
        return torch.stack([c.data.root_link_pose_w for c in self._cubes.values()]).reshape(
            self.sim.num_envs, self._n_objects, -1
        )
