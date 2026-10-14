"""Base class for teleoperation interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from kinova_lfd.core.np_math import base_to_tcp_twist, euler_xyz_to_rotvec
from kinova_lfd.core.tensor_math import base_to_tcp_twist as torch_base_to_tcp_twist
from kinova_lfd.core.tensor_math import euler_xyz_to_rotvec as torch_euler_xyz_to_rotvec


@dataclass
class DeviceCfg:
    """Configuration for all supported teleoperation devices."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    reference_frame: str = "base"


class DeviceBase(ABC):
    """An interface class for teleoperation devices."""

    def __init__(self, cfg: DeviceCfg):
        """Initialize the teleoperation interface."""
        self.cfg = cfg
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self.frame = cfg.reference_frame
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._gripper_value = 0.0
        self._home_flag = False
        self._quit_flag = False
        self._stop_flag = False
        self._resume_flag = False
        self._save_flag = False
        self._log_flag = False
        self._new_exp_flag = False
        self._admittance_flag = False
        self._close_gripper = False

    def reset(self):
        """Reset the internals."""
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._home_flag = False
        self._quit_flag = False
        self._stop_flag = False
        self._resume_flag = False
        self._save_flag = False
        self._log_flag = False
        self._new_exp_flag = False
        self._admittance_flag = False
        self._close_gripper = False
        self._gripper_value = 0.0

    @property
    def home(self):
        """If to return the robot to home."""
        return self._home_flag

    @property
    def quit(self):
        """If to quit the experiment."""
        return self._quit_flag

    @property
    def stop_exp(self):
        """If to stop the experiment."""
        return self._stop_flag

    @property
    def resume(self):
        """If to resume the experiment."""
        return self._resume_flag

    @property
    def log(self):
        return self._log_flag

    @property
    def save(self):
        return self._save_flag

    @property
    def new_exp(self):
        return self._new_exp_flag

    @property
    def admittance(self):
        return self._admittance_flag

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def start(self):
        """Start a thread."""
        raise NotImplementedError

    def advance(self, tcp_pose_b: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return the current command as a tensor.

        Args:
            tcp_pose_b: TCP pose of the robot in the robot base frame.

        Returns:
            np.ndarray: [x, y, z, rx, ry, rz] or [x, y, z, rx, ry, rz, gripper]

        """
        if isinstance(tcp_pose_b, torch.Tensor):
            delta_pos = torch.as_tensor(self._delta_pos, device=tcp_pose_b.device, dtype=torch.float32).unsqueeze(0)
            delta_rot = torch.as_tensor(self._delta_rot, device=tcp_pose_b.device, dtype=torch.float32).unsqueeze(0)
            if self.frame == "tcp":
                rot_vec = torch_euler_xyz_to_rotvec(delta_rot)
                command = torch.cat((delta_pos, rot_vec), dim=1)
            elif self.frame == "base":
                tcp_lin_vel, tcp_ang_vel = torch_base_to_tcp_twist(delta_pos, delta_rot, tcp_pose_b[:, 3:7])
                command = torch.cat((tcp_lin_vel, tcp_ang_vel), dim=1)

            if self.gripper_term:
                command = torch.cat(
                    (command, torch.as_tensor([self._gripper_value], device=tcp_pose_b.device).unsqueeze(0)), dim=1
                )
        else:
            if self.frame == "tcp":
                rot_vec = euler_xyz_to_rotvec(self._delta_rot)
                command = np.concatenate((self._delta_pos, rot_vec), axis=0)
            elif self.frame == "base":
                tcp_lin_vel, tcp_ang_vel = base_to_tcp_twist(self._delta_pos, self._delta_rot, tcp_pose_b[3:7])
                command = np.concatenate((tcp_lin_vel, tcp_ang_vel), axis=0)

            if self.gripper_term:
                command = np.concatenate((command, np.asarray([self._gripper_value])), axis=0)
        return command

    def __call__(self, observation: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self.advance(observation)
