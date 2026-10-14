"""Define a gymnasium environment interface for compatibility across different environments."""

from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from kinova_lfd.core.specs import ActionSpec


class GymEnv:
    """The Gymnasium environment that all Rl environments inherit from."""

    @abstractmethod
    def step(self, action: torch.Tensor | np.ndarray, action_spec: ActionSpec):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
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
    def _get_latest_rgbd() -> np.ndarray | torch.Tensor:
        """Get the latest RGBD information from the camera in the environment."""
        raise NotImplementedError

    @abstractmethod
    def _get_latest_rgbd_wrist() -> np.ndarray | torch.Tensor:
        """Get the latest RGBD information from the wrist camera in the environment."""
        raise NotImplementedError

    @property
    def _prev_actions(self) -> np.ndarray | torch.Tensor:
        """Return the previous action taken in the environment."""
        raise NotImplementedError

    @property
    def robot(self) -> Any:
        """Returns the robot articulation for Isaac and Mj environments. Useful for helper functions."""
        raise NotImplementedError

    @property
    def _robot_tool_pose_b(self) -> np.ndarray | torch.Tensor:
        """Return current tool frame position."""
        raise NotImplementedError

    @property
    def _robot_tool_vel_b(self) -> np.ndarray | torch.Tensor:
        """Return current tool frame velocity."""
        raise NotImplementedError

    @property
    def _robot_tool_wrench_b(self) -> np.ndarray | torch.Tensor:
        """Return current tool frame wrench (force + torque)."""
        raise NotImplementedError

    @property
    def _tcp_pose_b(self) -> np.ndarray | torch.Tensor:
        """Return current tool frame position."""
        raise NotImplementedError

    @property
    def _ee_vel_b(self) -> np.ndarray | torch.Tensor:
        """Return current EE velocity."""
        raise NotImplementedError

    @property
    def _joint_positions(self) -> np.ndarray | torch.Tensor:
        """Return current joint positions."""
        raise NotImplementedError

    @property
    def _joint_velocities(self) -> np.ndarray | torch.Tensor:
        """Return current joint velocities."""
        raise NotImplementedError

    @property
    def _joint_efforts(self) -> np.ndarray | torch.Tensor:
        """Return current joint efforts (torques)."""
        raise NotImplementedError

    @property
    def _robot_body_pose_w(self) -> np.ndarray | torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _robot_root_pose_w(self) -> np.ndarray | torch.Tensor:
        """Return the body pose information in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _jacobians(self) -> np.ndarray | torch.Tensor:
        """Return the jacobian frame transforms of the robot."""
        raise NotImplementedError

    @property
    def _robot_dof_lower_limits(self) -> np.ndarray | torch.Tensor:
        """Return the lower limits of the robot joints."""
        raise NotImplementedError

    @property
    def _robot_dof_upper_limits(self) -> np.ndarray | torch.Tensor:
        """Return the upper limits of the robot joints."""
        raise NotImplementedError

    @property
    def _gravity_vector(self) -> np.ndarray | torch.Tensor:
        """Return the gravity compenstation vector."""
        raise NotImplementedError

    @property
    def _mass_matrices(self) -> np.ndarray | torch.Tensor:
        """Return the mass matrices."""
        raise NotImplementedError

    @property
    def _robot_body_vel_w(self) -> np.ndarray | torch.Tensor:
        """Return body velocity in XYZ + Quaternion."""
        raise NotImplementedError

    @property
    def _joint_centers(self) -> np.ndarray | torch.Tensor:
        """Return joint centers."""
        raise NotImplementedError
