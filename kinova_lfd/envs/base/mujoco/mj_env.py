import math
from abc import abstractmethod
from dataclasses import MISSING, field
from typing import Any

import gymnasium as gym
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.envs import types
from mjlab.managers.event_manager import EventManager
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.utils import random as random_utils
from mjlab.utils.logging import print_info
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from prettytable import PrettyTable
from skillet.core.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul
from skillet.core.spaces import ActionSpec
from skillet.envs.compatibility import SkilletGymEnv
from skillet.envs.mujoco.mj_viewer import NativeMujocoViewer
from skillet.envs.util import configclass
from skillet.rl.s2r import ObservationManager

from kinova_lfd.envs.base import mujoco


@configclass
class MjDirectRlEnvCfg:
    """Configuration for a manager-based RL environment.

    This config defines all aspects of an RL environment: the physical scene,
    observations, actions, rewards, terminations, and optional features like
    commands and curriculum learning.

    The environment step size is ``sim.mujoco.timestep * decimation``. For example,
    with a 2ms physics timestep and decimation=10, the environment runs at 50Hz.
    """

    # Base environment configuration.

    obs_terms: list[str] = MISSING

    decimation: int = MISSING
    """Number of physics simulation steps per environment step. Higher values mean
  coarser control frequency. Environment step duration = physics_dt * decimation."""

    scene: SceneCfg = MISSING
    """Scene configuration defining terrain, entities, and sensors. The scene
  specifies ``num_envs``, the number of parallel environments."""

    seed: int | None = None
    """Random seed for reproducibility. If None, a random seed is used. The actual
  seed used is stored back into this field after initialization."""

    sim: SimulationCfg = field(default_factory=SimulationCfg)
    """Simulation configuration including physics timestep, solver iterations,
  contact parameters, and NaN guarding."""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Viewer configuration for rendering (camera position, resolution, etc.)."""

    # RL-specific configuration.

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

  Episode length in steps is computed as:
    ceil(episode_length_s / (sim.mujoco.timestep * decimation))
  """
    is_finite_horizon: bool = MISSING
    """Whether the task has a finite or infinite horizon. Defaults to False (infinite).

  - **Finite horizon (True)**: The time limit defines the task boundary. When reached,
    no future value exists beyond it, so the agent receives a terminal done signal.
  - **Infinite horizon (False)**: The time limit is an artificial cutoff. The agent
    receives a truncated done signal to bootstrap the value of continuing beyond the
    limit.
  """

    events: object | None = None
    """Event settings. Defaults to None, in which case no events are applied through the event manager.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    scale_rewards_by_dt: bool = MISSING
    """Whether to multiply rewards by the environment step duration (dt).

  When True (default), reward values are scaled by step_dt to normalize cumulative
  episodic rewards across different simulation frequencies. Set to False for
  algorithms that expect unscaled reward signals (e.g., HER, static reward scaling).
  """


class MjDirectRlEnv(SkilletGymEnv):
    """Manager-based RL environment."""

    is_vector_env = True
    metadata = {
        "render_modes": [None, "rgb_array", "human"],
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.config.version,
    }
    cfg: MjDirectRlEnvCfg

    def __init__(
        self,
        cfg: MjDirectRlEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        # Initialize base environment state.
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self.extras = {}
        self.obs_buf = {}

        self._setup_scene()

        # Wire sensor context to simulation for sense_graph.
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        # Print environment info.
        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        # Initialize RL-specific state.
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)

        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene)
            renderer.initialize()
            self._offline_renderer = renderer
        if self.render_mode == "human":
            self._mj_renderer = NativeMujocoViewer(self)
            self._mj_renderer.initialize()
        self.metadata["render_fps"] = 1.0 / self.step_dt

        # Configure spaces for the environment.
        self._configure_gym_env_spaces()

        self.obs_manager = ObservationManager(self.cfg.obs_terms, self)

        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)
            print_info(f"[INFO] {self.event_manager}")

            # Initialize startup events if defined.
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    # Properties.

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """Physics simulation step size."""
        return self.cfg.sim.mujoco.timestep

    @property
    def step_dt(self) -> float:
        """Environment step size (physics_dt * decimation)."""
        return self.cfg.sim.mujoco.timestep * self.cfg.decimation

    @property
    def device(self) -> str:
        """Device for computation."""
        return self.sim.device

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def unwrapped(self) -> "MjDirectRlEnv":
        """Get the unwrapped environment (base case for wrapper chains)."""
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[types.VecEnvObs, dict]:
        del options  # Unused.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if seed is not None:
            self.seed(seed)
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.sense()
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor, action_spec: ActionSpec = None) -> types.VecEnvStepReturn:
        """Run one environment step: apply actions, simulate, compute RL signals.

        **Forward-call placement.** MuJoCo's ``mj_step`` runs forward kinematics
        *before* integration, so after stepping, derived quantities (``xpos``,
        ``xquat``, ``site_xpos``, ``cvel``, ``sensordata``) lag ``qpos``/``qvel``
        by one physics substep. Rather than calling ``sim.forward()`` twice (once
        after the decimation loop and once after the reset block), this method
        calls it **once**, right before observation computation. This single call
        refreshes derived quantities for *all* envs: non-reset envs pick up
        post-decimation kinematics, reset envs pick up post-reset kinematics.

        The tradeoff is that termination and reward managers see derived
        quantities that are stale by one physics substep (the last ``mj_step``
        ran ``mj_forward`` from *pre*-integration ``qpos``). In practice, the
        staleness is negligible for reward shaping and termination
        checks. Critically, the staleness is *consistent*: every env,
        every step, always sees the same lag, so the MDP is well-defined
        and the value function can learn the correct mapping.

        .. note::

          Event and command authors do not need to call ``sim.forward()``
          themselves. This method handles it. The only constraint is: do not
          read derived quantities (``root_link_pose_w``, ``body_link_vel_w``,
          etc.) in the same function that writes state
          (``write_root_state_to_sim``, ``write_joint_state_to_sim``, etc.).
          See :ref:`faq` for details.
        """
        action = action.to(self.device)

        # process actions
        self._pre_physics_step(action, action_spec=action_spec)

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        # Update env counters.
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Check terminations and compute rewards.
        # Note: Derived quantities (xpos, xquat, ...) are stale by one physics
        # substep here. See the docstring above for why this is acceptable.
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.reward_buf = self._get_rewards()

        # Reset envs that terminated/timed-out and log the episode info.
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()

        # Single forward() call: recompute derived quantities from current
        # qpos/qvel for every env. For non-reset envs this resolves the
        # one-substep staleness left by mj_step; for reset envs it picks up
        # the freshly written reset state.
        self.sim.forward()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.sim.sense()
        self.obs_buf = self._get_observations()

        self.render()

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            return self._mj_renderer.render()
        if self.render_mode == "rgb_array":
            if self._offline_renderer is None:
                raise ValueError("Offline renderer not initialized")
            debug_callback = self.update_visualizers if hasattr(self, "update_visualizers") else None
            self._offline_renderer.update(self.sim.data, debug_vis_callback=debug_callback)
            return self._offline_renderer.render()
        raise NotImplementedError(
            f"Render mode {self.render_mode} is not supported. Please use: {self.metadata['render_modes']}."
        )

    def close(self) -> None:
        if self._offline_renderer is not None:
            self._offline_renderer.close()

    @staticmethod
    def seed(seed: int = -1) -> int:
        if seed == -1:
            seed = np.random.randint(0, 10_000)
        print_info(f"Setting seed: {seed}")
        random_utils.seed_rng(seed)
        return seed

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        for sensor in self.scene.sensors.values():
            sensor.debug_vis(visualizer)

    # Private methods.

    def _configure_gym_env_spaces(self) -> None:
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space.spaces["policy"] = spec_to_gym_space(self.cfg.observation_space)
        action_dim = self.cfg.action_space
        self.single_action_space = gym.spaces.Box(shape=(action_dim,), low=-math.inf, high=math.inf)

        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)

        if self.cfg.events and "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # NONoteTE: This is order sensitive.
        self.extras["log"] = dict()
        # reset the episode length buffer.
        self.episode_length_buf[env_ids] = 0

    @abstractmethod
    def _setup_scene(self):
        """Render the scene for the environment.

        This function is responsible for creating the scene objects and setting up the scene for the environment.
        The scene creation can happen through :class:`isaaclab.scene.InteractiveSceneCfg` or through
        directly creating the scene objects and registering them with the scene manager.

        We leave the implementation of this function to the derived classes. If the environment does not require
        any explicit scene setup, the function can be left empty.
        """
        raise NotImplementedError(f"Please implement the '_setup_scene' method for {self.__class__.__name__}.")

    @abstractmethod
    def _pre_physics_step(self, actions: torch.Tensor, action_spec: ActionSpec = None):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).

        """
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self) -> types.VecEnvStepReturn:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.

        """
        raise NotImplementedError

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

    """
    Skillet interface properties
    """

    def _find_link_idx(self, link: str) -> int:
        """Find the link index of the robot link."""
        return self.robot.find_bodies(link)[0][0]

    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index robot joint."""
        return self.robot.find_joints(joint)[0][0]

    def _get_latest_rgbd() -> torch.Tensor:
        """Get the latest RGBD information from the camera in the environment."""
        raise NotImplementedError

    @property
    def _prev_actions(self) -> torch.Tensor:
        """Return the previous actions."""
        return self._current_prev_actions

    @property
    def robot(self) -> Entity:
        """Return the robot entity."""
        if hasattr(self, "_robot"):
            return self._robot
        if hasattr(self, "scene"):
            if hasattr(self.scene, "entities") and "robot" in self.scene.entities:
                return self.scene.entities["robot"]
        else:
            raise ValueError(
                f"Environment `{self}` has no attribute `scene['robot']`. Unable to parse robot Articulation."
            )
        return None

    @property
    def _tcp_pose_b(self) -> torch.Tensor:
        """Return current tool frame position."""
        ee_pose_w = self._robot_body_pose_w[:, self._find_link_idx(self.cfg.ee_link_name)]
        ee_pos_b = quat_apply_inverse(
            self._robot_root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - self._robot_root_pose_w[:, 0:3]
        )
        ee_quat_b = quat_mul(quat_inv(self._robot_root_pose_w[:, 3:7]), ee_pose_w[:, 3:7])

        tcp_pos_b = ee_pos_b + quat_apply(ee_quat_b, self.tcp_offset[:, 0:3])
        tcp_quat_b = quat_mul(ee_quat_b, self.tcp_offset[:, 3:7])
        self.robot_tcp_pose_b = torch.concatenate(
            (tcp_pos_b, tcp_quat_b),
            dim=1,
        )

        tcp_pos = ee_pos_b + quat_apply(ee_quat_b, self.tcp_offset[:, 0:3])
        tcp_quat = quat_mul(ee_quat_b, self.tcp_offset[:, 3:7])
        return torch.concatenate(
            (tcp_pos, tcp_quat),
            dim=1,
        )

    @property
    def _ee_vel_b(self) -> torch.Tensor:
        """Return current TCP velocity in robot base frame."""
        base_link_idx = self._find_link_idx(self.cfg.base_link_name)
        ee_vel_w = self.robot.data.body_link_vel_w[:, self._find_link_idx(self.cfg.ee_link_name), :]
        root_vel_w = self.robot.data.body_link_vel_w[:, base_link_idx, :]
        relative_vel_w = ee_vel_w - root_vel_w
        ee_lin_vel_b = quat_apply_inverse(
            self.robot.data.body_link_pose_w[:, base_link_idx][:, 3:7],
            relative_vel_w[:, 0:3],
        )  # From world to root frame
        ee_ang_vel_b = quat_apply_inverse(
            self.robot.data.body_link_pose_w[:, base_link_idx][:, 3:7],
            relative_vel_w[:, 3:6],
        )
        return torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    @property
    def _robot_tool_pose_b(self):
        """Return the tool position."""
        return None

    @property
    def _robot_tool_vel_b(self):
        """Return the tool velocity."""
        return None

    @property
    def _robot_tool_wrench_b(self):
        """Return the tool wrench forces."""
        return None

    @property
    def _joint_positions(self) -> torch.Tensor:
        """Return current joint positions."""
        return self.robot.data.joint_pos

    @property
    def _joint_velocities(self) -> torch.Tensor:
        """Return current joint velocities."""
        return self.robot.data.joint_vel

    @property
    def _joint_efforts(self) -> torch.Tensor:
        """Return current joint efforts (torques)."""
        return self.robot.data.qfrc_actuator

    @property
    def _robot_body_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        return self.robot.data.body_link_pose_w

    @property
    def _robot_root_pose_w(self) -> torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        return self.robot.data.root_link_pose_w

    @property
    def _jacobians(self) -> torch.Tensor:
        """Return the jacobian frame transforms of the robot."""
        nv = self.robot.data.model.nv
        num_links = self.robot.data.data.xpos.shape[1]
        jacobians = torch.zeros((self.num_envs, num_links, 6, nv), device=self.device)
        for i in range(num_links):
            jacp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
            jacr = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32)
            mjw.jac(
                self.robot.data.model,
                self.robot.data.data,
                jacp,
                jacr,
                self.robot.data.data.xpos[:, i],
                wp.array(np.full(self.num_envs, i), dtype=wp.int32),
            )
            jacobians[:, i, 0:3] = wp.to_torch(jacp)
            jacobians[:, i, 3:6] = wp.to_torch(jacr)
        return jacobians

    @property
    def _robot_dof_lower_limits(self) -> torch.Tensor:
        """Return the lower limits of the robot joints."""
        lower_lim = self.robot.data.soft_joint_pos_limits[0, :, 0]
        lower_lim[lower_lim == -float("inf")] = -2 * torch.pi
        return lower_lim

    @property
    def _robot_dof_upper_limits(self) -> torch.Tensor:
        """Return the upper limits of the robot joints."""
        upper_lim = self.robot.data.soft_joint_pos_limits[0, :, 1]
        upper_lim[upper_lim == float("inf")] = 2 * torch.pi
        return upper_lim

    @property
    def _gravity_vector(self) -> torch.Tensor:
        """Return the gravity compenstation vector."""
        raise NotImplementedError

    @property
    def _mass_matrices(self) -> torch.Tensor:
        """Return the mass matrices."""
        raise NotImplementedError

    @property
    def _robot_body_vel_w(self) -> torch.Tensor:
        """Return body velocity in XYZ + Quaternion."""
        return self.robot.data.body_link_vel_w

    @property
    def _joint_centers(self) -> torch.Tensor:
        """Return joint centers."""
        return torch.nan_to_num(
            torch.mean(self.robot.data.soft_joint_pos_limits[:, :, :], dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


def spec_to_gym_space(spec) -> gym.spaces.Space:
    """Generate an appropriate Gymnasium space according to the given space specification.

    Args:
        spec: Space specification.

    Returns:
        Gymnasium space.

    Raises:
        ValueError: If the given space specification is not valid/supported.

    """
    if isinstance(spec, gym.spaces.Space):
        return spec
    # fundamental spaces
    # Box
    if isinstance(spec, int):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(spec,))
    if isinstance(spec, list) and all(isinstance(x, int) for x in spec):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=spec)
    if isinstance(spec, dict):
        return gym.spaces.Dict({k: spec_to_gym_space(v) for k, v in spec.items()})
    raise ValueError(f"Unsupported space specification: {spec}")
