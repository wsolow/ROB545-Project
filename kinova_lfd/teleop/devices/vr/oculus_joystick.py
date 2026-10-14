"""VR joystick controller for Oculus"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from kinova_lfd.core.np_math import base_to_tcp_twist, euler_xyz_to_rotvec

from ..device_base import DeviceBase, DeviceCfg
from .joystick_listener import VRJoystickListener


class VRJoystick(DeviceBase):
    """A Joystick controller for the Oculus VR headset.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    RC - A
        Move along x-axis              RC - Joystick R   RC - Joystick L
        Move along y-axis              RC - Joystick Up  RC - Joystick Down
        Move along z-axis              RC - Gripper      RC - Trigger
        Rotate along x-axis            LC - Joystick R   LC - Joytick L
        Rotate along y-axis            LC - Joystick Up  LC - Joystick Down
        Rotate along z-axis            LC - Gripper      LC - Trigger
        ============================== ================= =================
    """

    def __init__(self, cfg: VRJoystickCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term

        # Command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)

        self._additional_callbacks: dict[str, Callable] = {}
        self._listener = VRJoystickListener(host=cfg.host, port=cfg.port)

    def __del__(self):
        """Stop the VR joystick listener."""
        if hasattr(self, "_listener") and self._listener.is_running:
            self._listener.close()

    def __str__(self) -> str:
        msg = f"VR Joystick Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t--------------------------------------------------\n"
        msg += "\tToggle gripper (open/close): RC - A\n"
        msg += "\n"
        msg += "\tTranslation Controls (Right Controller)\n"
        msg += "\tMove arm along x-axis: RC - Joystick Right / Left\n"
        msg += "\tMove arm along y-axis: RC - Joystick Up / Down\n"
        msg += "\tMove arm along z-axis: RC - Gripper / Trigger\n"
        msg += "\n"
        msg += "\tRotation Controls (Left Controller)\n"
        msg += "\tRotate arm along x-axis: LC - Joystick Right / Left\n"
        msg += "\tRotate arm along y-axis: LC - Joystick Up / Down\n"
        msg += "\tRotate arm along z-axis: LC - Gripper / Trigger\n"
        return msg

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def reset(self):
        self._close_gripper = False
        self._delta_pos = np.zeros((3,))
        self._delta_rot = np.zeros((3,))

    def advance(self, tcp_pose_b: np.ndarray) -> np.ndarray:
        """Return the current command as a tensor.

        Args:
            tcp_pose_b: TCP pose of robot in robot base frame.

        Returns:
            np.ndarray: [x, y, z, rx, ry, rz] or [x, y, z, rx, ry, rz, gripper]

        """
        self._read_latest()

        rot_vec = euler_xyz_to_rotvec(self._delta_rot)
        if self.frame == "tcp":
            command = np.concatenate((self._delta_pos, rot_vec), axis=0)
        elif self.frame == "base":
            tcp_lin_vel, tcp_ang_vel = base_to_tcp_twist(self._delta_pos, self._delta_rot, tcp_pose_b[3:7])
            command = np.concatenate((tcp_lin_vel, tcp_ang_vel), axis=0)
        if self.gripper_term:
            gripper_value = 1.0 if self._close_gripper else -1.0
            command = np.concatenate((command, np.asarray([gripper_value])), axis=0)
        return command

    def _read_latest(self) -> None:
        """Read the latest input from the VR Joystick."""
        s = self._listener.read_latest()

        if s is not None:
            self._delta_pos[0] = -s["right_controller"]["inputs"]["joystick"][1] * self.pos_sensitivity
            self._delta_pos[1] = -s["right_controller"]["inputs"]["joystick"][0] * self.pos_sensitivity
            self._delta_pos[2] = s["right_controller"]["inputs"]["grip"] * self.pos_sensitivity
            if s["right_controller"]["inputs"]["trigger"] > 0.05:
                self._delta_pos[2] = -s["right_controller"]["inputs"]["trigger"] * self.pos_sensitivity

            self._delta_rot[0] = -s["left_controller"]["inputs"]["joystick"][1] * self.pos_sensitivity
            self._delta_rot[1] = -s["left_controller"]["inputs"]["joystick"][0] * self.pos_sensitivity
            self._delta_rot[2] = s["left_controller"]["inputs"]["grip"] * self.pos_sensitivity
            if s["left_controller"]["inputs"]["trigger"] > 0.05:
                self._delta_rot[2] = -s["left_controller"]["inputs"]["trigger"] * self.pos_sensitivity
            if s["right_controller"]["inputs"]["A_button"]:
                self._close_gripper = not self._close_gripper

    def add_callback(self, key: Any, func: Callable):
        """Callback function."""
        pass


@dataclass
class VRJoystickCfg(DeviceCfg):
    """Configuration for VR Joystick."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    retargeters: None = None
    class_type: type[DeviceBase] = VRJoystick
    host: str = "192.168.2.33"
    port: int = 5555
    reference_frame: str = "base"  # or base
