"""kortex_env.py.

Main Kortex

Written by Will Solow, 2026

"""

import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

from kinova_lfd.core.math import np_quat_from_euler_xyz
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.envs.base.kortex.kortex_bridge import DeviceConnection
from kinova_lfd.envs.base.seed import configure_seed


@dataclass
class KortexEnvCfg:
    """The configuration class for Kortex Envs."""

    robot_ip: str = "www.xxx.yyy.zzz"
    """IP of the robot"""

    use_tabletop_camera: bool = False

    use_wrist_camera: bool = True

    dt: float = 1.0 / 30.0

    decimation: float = 1.0

    use_udp: bool = False
    """If to use UDP for arm commands and feedback."""


class KortexEnv(GymEnv):
    """The superclass for the Kortex workflow to design environments.

    This class implements the core functionality for reinforcement learning (RL)
    environments. However, it is designed to interface with Kortex and thus calls the corresponding
    API calls etc.

    """

    _current_robot_tool_pose_b: np.ndarray
    _current_robot_tool_vel_b: np.ndarray
    _current_robot_tool_wrench_b: np.ndarray
    _current_joint_positions: np.ndarray
    _current_joint_velocities: np.ndarray
    _current_joint_efforts: np.ndarray
    _robot_links: list[str]
    _robot_joints: list[str]
    _current_robot_body_pose_w: np.ndarray
    _current_robot_root_pose_w: np.ndarray
    _current_jacobians: np.ndarray
    _current_upper_joint_limits: np.ndarray
    _current_lower_joint_limits: np.ndarray
    _current_gravity_vector: np.ndarray
    _current_mass_matrices: np.ndarray
    _current_robot_body_vel_w: np.ndarray
    _current_joint_centers: np.ndarray
    _current_prev_actions: np.ndarray
    _current_wrist_rgb: np.ndarray

    def __init__(
        self,
        cfg: KortexEnvCfg,
        kortex_tcp_udp: tuple[DeviceConnection, DeviceConnection],
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
            kwargs: Additoinal arguments

        """
        self.cfg = cfg

        self.kortex_tcp_udp = kortex_tcp_udp
        self.kortex = self.kortex_tcp_udp[0].open_base()
        self.kortex_c = self.kortex_tcp_udp[1].open_cyclic()

        if self.cfg.use_udp:  # Configure UDP command for real time control
            self._cyclic_cmd = BaseCyclic_pb2.Command()
            self._cyclic_cmd.frame_id = 0
            self._cyclic_cmd.interconnect.command_id.identifier = 0
            self._cyclic_cmd.interconnect.gripper_command.command_id.identifier = 0
            self._feedback = self.kortex_c.RefreshFeedback()

            # Add motor command to interconnect's cyclic and set initial position
            self._cyclic_gripper_cmd = self._cyclic_cmd.interconnect.gripper_command.motor_cmd.add()
            self._cyclic_gripper_cmd.position = self._feedback.interconnect.gripper_feedback.motor[0].position
            self._cyclic_gripper_cmd.velocity = 0
            self._cyclic_gripper_cmd.force = 100

            # Add actuators to base cmd
            # TODO this is probably wrong...
            for actuator in self._feedback.actuators:
                self._cyclic_actuator_cmd = self._cyclic_cmd.actuators.add()
                self._cyclic_actuator_cmd.position = actuator.position
                self._cyclic_actuator_cmd.velocity = 0.0
                self._cyclic_actuator_cmd.torque_joint = 0.0
                self._cyclic_actuator_cmd.command_id = 0

        self.render_mode = render_mode
        self._sim_step_counter = 0
        # -- counter for curriculum
        self._common_step_counter = 0
        # -- init buffers
        self._episode_length_buf = 0
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False
        # allocate dictionary to store metrics
        self.extras: dict[str, Any] = {}

        # setup the action and observation spaces for Gym
        self._next_step_time = time.perf_counter()

        print("[INFO][KortexEnv] Completed Environment Setup")

    def __del__(self) -> None:
        """Cleanup for the environment."""
        if self.kortex_connection is not None:
            self.kortex_connection.close()
        self.close()

    @property
    def episode_length_buf(self) -> np.ndarray:
        return np.array([self._episode_length_buf])

    @episode_length_buf.setter
    def episode_length_buf(self, value: np.ndarray) -> None:
        self._episode_length_buf = value.squeeze().item()

    """
    Gymansium Interface Properties.
    """

    @property
    def _robot_tool_wrench_b(self) -> np.ndarray:
        return self._current_robot_tool_wrench_b

    @property
    def _robot_tool_vel_b(self) -> np.ndarray:
        return self._current_robot_tool_vel_b

    @property
    def _robot_tool_pose_b(self) -> np.ndarray:
        return self._current_robot_tool_pose_b

    @property
    def _wrist_rgb(self) -> np.ndarray:
        return self._current_wrist_rgb

    @property
    def _prev_actions(self) -> np.ndarray:
        """Return the previous action taken in the environment."""
        return self._current_prev_actions

    @property
    def _joint_positions(self) -> np.ndarray:
        """Return current joint positions."""
        return self._current_joint_positions

    @property
    def _joint_velocities(self) -> np.ndarray:
        """Return current joint velocities."""
        return self._current_joint_velocities

    @property
    def _joint_efforts(self) -> np.ndarray:
        """Return current joint efforts (torques)."""
        return self._current_joint_efforts

    @property
    def _robot_body_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        return self._current_robot_body_pose_w

    @property
    def _robot_root_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        return self._current_robot_root_pose_w

    @property
    def _jacobians(self) -> np.ndarray:
        """Return the jacobian frame transforms of the robot."""
        return self._current_jacobians

    @property
    def _robot_dof_lower_limits(self) -> np.ndarray:
        """Return the lower limits of the robot joints."""
        return self._current_lower_joint_limits

    @property
    def _robot_dof_upper_limits(self) -> np.ndarray:
        """Return the upper limits of the robot joints."""
        return self._current_upper_joint_limits

    @property
    def _gravity_vector(self) -> np.ndarray:
        """Return the gravity compenstation vector."""
        return self._current_gravity_vector

    @property
    def _mass_matrices(self) -> np.ndarray:
        """Return the mass matrices."""
        return self._current_mass_matrices

    @property
    def _robot_body_vel_w(self) -> np.ndarray:
        """Return body velocity in XYZ + Quaternion."""
        return self._current_robot_body_vel_w

    @property
    def _joint_centers(self) -> np.ndarray:
        """Return joint centers."""
        return self._current_joint_centers

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which we publish to Kortex.
        """
        return self.cfg.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return float(self.cfg.dt * self.cfg.decimation)

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.max_episode_length_s / (self.cfg.dt * self.cfg.decimation))

    def _find_link_idx(self, link: str) -> int:
        """Find the link index in the robot urdf.

        Args:
            link: string of link name

        Returns:
            int of index

        """
        return self._robot_links.index(link)

    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index in the robot urdf.

        Args:
            joint: string of joint name

        Returns:
            int of index

        """
        return self._robot_joints.index(joint)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations happen during reset() which are not repeated

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.


        Returns:
            A tuple containing the observations and extras.

        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx()

        self._episode_length_buf += 1
        self._common_step_counter += 1
        self._feedback = self.kortex_c.RefreshFeedback()
        self._update_robot_info()
        self._next_step_time = time.perf_counter()

        # return observations
        return self._get_observations(), self.extras

    def step(
        self, action: np.ndarray, action_spec: ActionSpec | None = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:  # type: ignore
        """Execute one time-step of the on the robot through the Kortex API.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRlEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRlEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre process the action and store it
        2. Publish the action to the robot
        3. Compute the rewards
        4. Get the observations
        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).
            action_spec: the action specification

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.

        """
        sleep_time = (time.perf_counter() - self._next_step_time) - self.step_dt
        if sleep_time < 0:
            time.sleep(min(-sleep_time, self.step_dt))
        else:
            # print(f"[WARN] full loop overran by {sleep_time * 1000:.1f}ms")
            pass
        self._next_step_time = time.perf_counter()

        # Pre process the robot action
        action = self._pre_process_action(action, action_spec=action_spec)
        # Send the robot action to hardware
        self._publish_action_to_kortex(action, duration=self.step_dt, action_spec=action_spec)

        self._episode_length_buf += 1
        self._common_step_counter += 1

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        #  reset envs that terminated/timed-out and log the episode information
        if self.reset_buf:
            self._reset_idx()

        self._update_robot_info()  # Update all robot information
        # update observations
        obs = self._get_observations()
        self.obs_buf = obs
        # return observations, rewards, resets and extras

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.

        """
        # set seed for torch and other libraries
        return configure_seed(seed)

    def render(self) -> None:
        """Run rendering by visualizing with RViz."""

    def close(self) -> None:
        """Cleanup for the environment."""
        self.kortex.Stop()

    """
    Helper functions.
    """

    def _reset_idx(self) -> None:
        """Reset environments based on specified indices."""
        # reset the episode length buffer
        self._episode_length_buf = 0

    """
    Implementation-specific functions.
    """

    @abstractmethod
    def _pre_process_action(self, actions: np.ndarray, action_spec: ActionSpec | None = None) -> np.ndarray:
        """Pre process the robot action.

        This function is responsible preprocessing the robot action (ie checking joint limits, etc).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, num_joints).
            action_spec: The action specification telling the environment how to process the action

        Returns:
            The actions to publish to the robot

        """
        raise NotImplementedError(f"Please implement the '_pre_process_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _publish_action_to_kortex(
        self, actions: np.ndarray, action_spec: ActionSpec | None = None, duration: float = 1
    ) -> None:
        """Publish action to robot controller via the Kortex API.

        This function is responsible publishing actions to the correct robot controller. These positions are set by
        _pre_process_action()

        Args:
            actions: joint positions to publish to the robot
            action_spec: The actions to publish to the Kortex API
            duration: duration that the action expects, useful for some action specs

        """
        raise NotImplementedError(
            f"Please implement the '_publish_action_to_kortex' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def _get_observations(self) -> dict[str, np.ndarray]:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment in the form {positions: [], velocities: []}.

        """
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).

        """
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self) -> tuple[bool, bool]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Assumed to not be batched

        """
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")

    """
    Helper functions to communicate with Kortex
    """

    def switch_controllers(self, active_controller: int) -> bool:
        """Switch Kortex controllers.

        Args:
            active_controller: Controller (Enum) to activate

        Returns:
            Bool if the controller switch was successful

        """
        print(f"[INFO][KORTEX] Switching to controller: {active_controller}")
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = active_controller
        self.kortex.SetServoingMode(base_servo_mode)
        self.active_controller = active_controller

    def _update_robot_info(self) -> None:
        """Obtain the required robot information from the Kortex API."""
        if not self.cfg.use_udp:  # Refresh the feedback
            self._feedback = self.kortex_c.RefreshFeedback()

        def deg_to_rad_wrapped(deg: np.ndarray) -> np.ndarray:
            """Shift angles > 180 to negative and convert to radians."""
            deg_wrapped = (deg + 180) % 360 - 180  # [0,360] -> [-180, 180]
            return np.deg2rad(deg_wrapped)

        # Compute tool pose
        self._current_robot_tool_pose_b = np.zeros(7)
        self._current_robot_tool_pose_b[0] = self._feedback.base.tool_pose_x
        self._current_robot_tool_pose_b[1] = self._feedback.base.tool_pose_y
        self._current_robot_tool_pose_b[2] = self._feedback.base.tool_pose_z
        self._current_robot_tool_pose_b[3:7] = np_quat_from_euler_xyz(
            deg_to_rad_wrapped(self._feedback.base.tool_pose_theta_x),
            deg_to_rad_wrapped(self._feedback.base.tool_pose_theta_y),
            deg_to_rad_wrapped(self._feedback.base.tool_pose_theta_z),
        )

        # Compute tool velocity
        self._current_robot_tool_vel_b = np.zeros(6)
        self._current_robot_tool_vel_b[0] = self._feedback.base.tool_twist_linear_x
        self._current_robot_tool_vel_b[1] = self._feedback.base.tool_twist_linear_y
        self._current_robot_tool_vel_b[2] = self._feedback.base.tool_twist_linear_z
        self._current_robot_tool_vel_b[3] = self._feedback.base.tool_twist_angular_x
        self._current_robot_tool_vel_b[4] = self._feedback.base.tool_twist_angular_y
        self._current_robot_tool_vel_b[5] = self._feedback.base.tool_twist_angular_z

        # Compute tool wrench
        self._current_robot_tool_wrench_b = np.zeros(6)
        self._current_robot_tool_wrench_b[0] = self._feedback.base.tool_external_wrench_force_x
        self._current_robot_tool_wrench_b[1] = self._feedback.base.tool_external_wrench_force_x
        self._current_robot_tool_wrench_b[2] = self._feedback.base.tool_external_wrench_force_x
        self._current_robot_tool_wrench_b[3] = self._feedback.base.tool_external_wrench_torque_x
        self._current_robot_tool_wrench_b[4] = self._feedback.base.tool_external_wrench_torque_y
        self._current_robot_tool_wrench_b[5] = self._feedback.base.tool_external_wrench_torque_z

        # Compute the current joint positions, velocities, and efforts
        curr_arm_positions = deg_to_rad_wrapped(
            np.asarray([actuator.position for actuator in self._feedback.actuators])
        )
        curr_gripper_positions = np.asarray([self._feedback.interconnect.gripper_feedback.motor[0].position]) / 100
        self._current_joint_positions = np.concatenate((curr_arm_positions, curr_gripper_positions), axis=0)

        curr_arm_velocities = np.deg2rad([actuator.velocity for actuator in self._feedback.actuators])
        curr_gripper_velocities = np.deg2rad([self._feedback.interconnect.gripper_feedback.motor[0].velocity])
        self._current_joint_velocities = np.concatenate((curr_arm_velocities, curr_gripper_velocities), axis=0)

        curr_arm_efforts = np.asarray([actuator.torque for actuator in self._feedback.actuators])
        curr_gripper_efforts = np.asarray([self._feedback.interconnect.gripper_feedback.motor[0].current_motor])
        self._current_joint_efforts = np.concatenate((curr_arm_efforts, curr_gripper_efforts), axis=0)

        if self.cfg.use_wrist_camera:
            self._current_wrist_rgb = self._rs_wrist_camera._latest_rgb_frame
            if self._current_wrist_rgb is not None:
                cv2.imshow("rgb", self._current_wrist_rgb)
                cv2.waitKey(1)
