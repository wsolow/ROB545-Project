"""Define a gymnasium environment interface for compatibility across different environments."""

from abc import abstractmethod
from typing import Any

import numpy as np


class GymEnv:
    """The Gymnasium environment that all Rl environments inherit from."""

    @abstractmethod
    def step():
        raise NotImplementedError

    @abstractmethod
    def reset():
        raise NotImplementedError

    @abstractmethod
    def _find_link_idx(self, link: str) -> int:
        """Find the link index of the robot link."""
        raise NotImplementedError

    @abstractmethod
    def _find_joint_idx(self, joint: str) -> int:
        """Find the joint index robot joint."""
        raise NotImplementedError

    @abstractmethod
    def _get_latest_rgbd() -> np.ndarray:
        """Get the latest RGBD information from the camera in the environment."""
        raise NotImplementedError

    @abstractmethod
    def _get_latest_rgbd_wrist() -> np.ndarray:
        """Get the latest RGBD information from the wrist camera in the environment."""
        raise NotImplementedError

    @property
    def _prev_actions(self) -> np.ndarray:
        """Return the previous action taken in the environment."""
        raise NotImplementedError

    @property
    def robot(self) -> Any:
        """Returns the robot articulation for Isaac and Mj environments. Useful for helper functions."""
        raise NotImplementedError

    @property
    def _robot_tool_pose_b(self) -> np.ndarray:
        """Return current tool frame position."""
        raise NotImplementedError

    @property
    def _robot_tool_vel_b(self) -> np.ndarray:
        """Return current tool frame velocity."""
        raise NotImplementedError

    @property
    def _robot_tool_wrench_b(self) -> np.ndarray:
        """Return current tool frame wrench (force + torque)."""
        raise NotImplementedError

    @property
    def _tcp_pose_b(self) -> np.ndarray:
        """Return current tool frame position."""
        raise NotImplementedError

    @property
    def _ee_vel_b(self) -> np.ndarray:
        """Return current EE velocity."""
        raise NotImplementedError

    @property
    def _joint_positions(self) -> np.ndarray:
        """Return current joint positions."""
        raise NotImplementedError

    @property
    def _joint_velocities(self) -> np.ndarray:
        """Return current joint velocities."""
        raise NotImplementedError

    @property
    def _joint_efforts(self) -> np.ndarray:
        """Return current joint efforts (torques)."""
        raise NotImplementedError

    @property
    def _robot_body_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _robot_root_pose_w(self) -> np.ndarray:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _jacobians(self) -> np.ndarray:
        """Return the jacobian frame transforms of the robot."""
        raise NotImplementedError

    @property
    def _robot_dof_lower_limits(self) -> np.ndarray:
        """Return the lower limits of the robot joints."""
        raise NotImplementedError

    @property
    def _robot_dof_upper_limits(self) -> np.ndarray:
        """Return the upper limits of the robot joints."""
        raise NotImplementedError

    @property
    def _gravity_vector(self) -> np.ndarray:
        """Return the gravity compenstation vector."""
        raise NotImplementedError

    @property
    def _mass_matrices(self) -> np.ndarray:
        """Return the mass matrices."""
        raise NotImplementedError

    @property
    def _robot_body_vel_w(self) -> np.ndarray:
        """Return body velocity in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _joint_centers(self) -> np.ndarray:
        """Return joint centers."""
        raise NotImplementedError
