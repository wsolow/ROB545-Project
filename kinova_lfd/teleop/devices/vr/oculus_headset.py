"""VR headset controller for Oculus"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing_extensions import override

from kinova_lfd.core.math import (
    base_to_tcp_twist,
    euler_xyz_from_quat,
    euler_xyz_to_rotvec,
)

from ..device_base import DeviceBase, DeviceCfg
from .joystick_listener import VRJoystickListener


class VRHeadset(DeviceBase):
    """A Headset controller using the relative positions of the controllers.

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

    def __init__(self, cfg: VRHeadsetCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._headset_offset = cfg.headset_offset
        self.frame = cfg.reference_frame

        self._workspace_lim = np.asarray([[0.35, -0.5, 0.03, -1.57, -1.57, -1.57], [0.6, 0.5, 0.75, 1.57, 1.57, 1.57]])
        self._vr_range = np.asarray([[0.0, -0.4, 1.1, -1.57, -1.57, -1.57], [0.4, 0.4, 2.00, 1.57, 1.57, 1.57]])
        # Command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._tcp_xyz_des_b = None
        self._enable_teleop = False
        self._reference_pose_w = None
        self._a_clicked = False
        self._b_clicked = False

        # PID gains
        self.Kp_pos = 1.0
        self.Ki_pos = 0.0
        self.Kd_pos = 0.1
        self.Kp_rot = 1.0
        self.Ki_rot = 0.0
        self.Kd_rot = 0.1

        # PID integrals
        self.integral_pos = np.zeros(3)
        self.integral_rot = np.zeros(3)

        # Last errors for derivative
        self.last_error_pos = np.zeros(3)
        self.last_error_rot = np.zeros(3)

        self._additional_callbacks: dict[str, Callable] = {}
        self._listener = VRJoystickListener(host=cfg.host, port=cfg.port)

    def __del__(self):
        """Stop the VR joystick listener."""
        if hasattr(self, "_listener") and self._listener.is_running:
            self._listener.close()

    def __str__(self) -> str:
        msg = f"VR Joystick Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t--------------------------------------------------\n"
        msg += "\t Enable/Disable teloperation: RC - A button"
        msg += "\t Set new reference frame: RC - B button"
        msg += "\tToggle gripper (open/close): RC - Trigger\n"
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
        self._reference_pose_w = None
        self._enable_teleop = False
        self._tcp_xyz_des_b = None
        self._a_clicked = False
        self._b_clicked = False

        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        # PID integrals
        self.integral_pos = np.zeros(3)
        self.integral_rot = np.zeros(3)

        # Last errors for derivative
        self.last_error_pos = np.zeros(3)
        self.last_error_rot = np.zeros(3)

    @override
    def advance(self, tcp_pose_b: np.ndarray, dt: float = 1 / 60) -> np.ndarray:
        """Return the current command as a tensor.

        Returns:
            np.ndarray: [x, y, z, rx, ry, rz] or [x, y, z, rx, ry, rz, gripper]

        """
        r, p, y = euler_xyz_from_quat(tcp_pose_b[3:7])
        self._read_latest()

        if self._tcp_xyz_des_b is None or not self._enable_teleop:
            command = np.zeros(6)
            if self.gripper_term:
                gripper_value = 1.0
                command = np.concatenate((command, np.asarray([gripper_value])), axis=0)

            return command
        r, p, y = euler_xyz_from_quat(tcp_pose_b[3:7])
        robot_xyz_b = np.concatenate((tcp_pose_b[0:3], r, p, y), axis=-1)

        # Compute errors
        error_pos = robot_xyz_b[:3] - self._tcp_xyz_des_b[:3]
        error_rot = robot_xyz_b[3:6] - self._tcp_xyz_des_b[3:6]

        # Update integral terms
        self.integral_pos += error_pos * dt
        self.integral_rot += error_rot * dt

        # Compute derivative terms
        derivative_pos = (error_pos - self.last_error_pos) * dt
        derivative_rot = (error_rot - self.last_error_rot) * dt

        # PID control for translation
        delta_pos = self.Kp_pos * error_pos + self.Ki_pos * self.integral_pos + self.Kd_pos * derivative_pos
        self._delta_pos = np.clip(delta_pos, -self.pos_sensitivity, self.pos_sensitivity)
        self._delta_pos[1] = -self._delta_pos[1]

        # PID control for rotation (Euler -> rotation vector)
        delta_rot = self.Kp_rot * error_rot + self.Ki_rot * self.integral_rot + self.Kd_rot * derivative_rot
        self._delta_rot = np.clip(delta_rot, -self.rot_sensitivity, self.rot_sensitivity)
        rot_vec = euler_xyz_to_rotvec(self._delta_rot)
        # Combine translation + rotation for twist command
        if self.frame == "tcp":
            command = np.concatenate((self._delta_pos, rot_vec), axis=0)
        elif self.frame == "base":
            tcp_lin_vel, tcp_ang_vel = base_to_tcp_twist(self._delta_pos, self._delta_rot, tcp_pose_b[3:7])
            command = np.concatenate((tcp_lin_vel, tcp_ang_vel), axis=0)
        command = np.concatenate((self._delta_pos, rot_vec), axis=0)
        command[3:] = 0
        # Append gripper command if needed
        if self.gripper_term:
            gripper_value = 1.0 if self._close_gripper else -1.0
            command = np.concatenate((command, np.asarray([gripper_value])), axis=0)

        # Save last errors
        self.last_error_pos = error_pos
        self.last_error_rot = error_rot

        return command

    def _read_latest(self) -> None:
        """Read the latest input from the VR Joystick."""
        s = self._listener.read_latest()
        print(s)
        if s is not None:
            if self._reference_pose_w == None:
                self._reference_pose_w = np.asarray(s["headset"]["pose"]).squeeze(0)

            left_elbow_pos, right_elbow_pos = get_tracker_data_fixed_arm(
                s, h_pose_raw=self._reference_pose_w.cpu().numpy()
            )

            # Enable/Disable teleoperation
            if s["right_controller"]["inputs"]["A_button"]:
                if not self._a_clicked:
                    old_teleop = self._enable_teleop
                    self._enable_teleop = not self._enable_teleop
                    if old_teleop != self._enable_teleop:
                        if self._enable_teleop:
                            print("[INFO] Enabling VR Teleoperation")
                        else:
                            print("[INFO] Disabling VR Teleoperation")
                    self._a_clicked = True
            else:
                self._a_clicked = False

            # Set new reference frame
            if s["right_controller"]["inputs"]["B_button"]:
                if not self._b_clicked:
                    self._enable_teleop = False
                    self._reference_pose_w = np.asarray(s["headset"]["pose"]).squeeze(0)
                    print("[INFO] Set new reference frame. Disabling VR Teleoperation.")
                    self._b_clicked = True
            else:
                self._b_clicked = False

            right_elbow_pos_b = np.asarray(right_elbow_pos)
            right_elbow_pos_b[1] = right_elbow_pos_b[1] - 0.2
            right_elbow_pos_b = (right_elbow_pos_b - self._vr_range[0, :3]) / (
                self._vr_range[1, :3] - self._vr_range[0, :3]
            )
            r, p, y = (
                np.array([0]),
                np.array([0]),
                np.array([0]),
            )
            rc_xyz_b = np.concatenate((right_elbow_pos_b.squeeze(), r, p, y), axis=-1)
            self._tcp_xyz_des_b = np.clip(rc_xyz_b, self._workspace_lim[0], self._workspace_lim[1])

            self._close_gripper = s["right_controller"]["inputs"]["trigger"] > 0.9

    def add_callback(self, key: Any, func: Callable) -> None:
        """Handle callback."""
        ...


@dataclass
class VRHeadsetCfg(DeviceCfg):
    """Configuration for VR Joystick."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    headset_offset: float = 0.5
    retargeters: None = None
    class_type: type[DeviceBase] = VRHeadset
    host: str = "192.168.2.33"
    port: int = 5555
    reference_frame: str = "tcp"  # or base


def convert_unity_to_world_coordinates(pose):
    """Convert pose from Unity (z-forward, x-right, y-up, left-handed) to world coordinates (x-forward, y-left, z-up).
    Handles both position (xyz) and quaternion (xyzw) formats.
    Unity coordinate system: Forward=+Z, Right=+X, Up=+Y (left-handed)
    MuJoCo coordinate system: Forward=+X, Right=-Y, Up=+Z (right-handed)
    """
    # Rotation matrix to convert from Unity to MuJoCo world coordinates
    # X_mujoco = Z_unity (forward)
    # Y_mujoco = -X_unity (left, which is negative right)
    # Z_mujoco = Y_unity (up)
    rotation_matrix_3d = np.array(
        [[0, 0, 1], [-1, 0, 0], [0, 1, 0]]  # X_mujoco = Z_unity  # Y_mujoco = -X_unity  # Z_mujoco = Y_unity
    )

    # Create 4x4 homogeneous transformation matrix
    rotation_matrix_4d = np.eye(4)
    rotation_matrix_4d[:3, :3] = rotation_matrix_3d

    if type(pose) == list:
        pose = np.array(pose)
    elif type(pose) == np.ndarray:
        pass
    elif type(pose) == tuple:
        pose = np.array(pose)
    else:
        raise ValueError("Pose must be list or numpy array")

    pose = np.asarray(pose)
    if pose.shape[0] == 3:
        # Only position - use 3D rotation matrix
        return rotation_matrix_3d @ pose
    if pose.shape[0] == 7:
        # Position + quaternion (xyz, xyzw)
        pos = pose[:3]
        quat = pose[3:]  # [x, y, z, w]

        # Convert quaternion to rotation matrix
        r = R.from_quat(quat)
        r_matrix = r.as_matrix()

        # Create 4x4 transformation matrix from the quaternion rotation
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r_matrix
        transform_matrix[:3, 3] = pos

        # Apply the coordinate transformation
        transformed_matrix = rotation_matrix_4d @ transform_matrix @ np.linalg.inv(rotation_matrix_4d)

        # Extract new position and rotation
        new_pos = transformed_matrix[:3, 3]
        new_rotation_matrix = transformed_matrix[:3, :3]

        # Convert back to quaternion
        r_new = R.from_matrix(new_rotation_matrix)
        quat_new = r_new.as_quat()

        return np.concatenate([new_pos, quat_new])
    raise ValueError(f"Pose `{pose}` must be length 3 (xyz) or 7 (xyzxyzw)")


def convert_to_world_coordinates(pose: np.ndarray) -> np.ndarray:
    """Convert pose from OpenVR (z-up, x-right, y-back) to world coordinates (x-forward, y-left, z-up).
    Handles both position (xyz) and quaternion (xyzw) formats.
    """
    # Rotation matrix to convert from OpenVR to mujoco  world coordinates
    rotation_matrix_3d = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    # Create 4x4 homogeneous transformation matrix
    rotation_matrix_4d = np.eye(4)
    rotation_matrix_4d[:3, :3] = rotation_matrix_3d

    if type(pose) == list:
        pose = np.array(pose)
    elif type(pose) == np.ndarray:
        pass
    elif type(pose) == tuple:
        pose = np.array(pose)
    else:
        raise ValueError("Pose must be list or numpy array")

    pose = np.asarray(pose)
    if pose.shape[0] == 3:
        # Only position - use 3D rotation matrix
        return rotation_matrix_3d @ pose
    if pose.shape[0] == 7:
        # Position + quaternion (xyz, xyzw)
        pos = pose[:3]
        quat = pose[3:]  # [x, y, z, w]

        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R

        r = R.from_quat(quat)
        r_matrix = r.as_matrix()

        # Create 4x4 transformation matrix from the quaternion rotation
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r_matrix
        transform_matrix[:3, 3] = pos

        # Apply the coordinate transformation
        transformed_matrix = rotation_matrix_4d @ transform_matrix @ np.linalg.inv(rotation_matrix_4d)

        # Extract new position and rotation
        new_pos = transformed_matrix[:3, 3]
        new_rotation_matrix = transformed_matrix[:3, :3]

        # Convert back to quaternion
        r_new = R.from_matrix(new_rotation_matrix)
        quat_new = r_new.as_quat()

        return np.concatenate([new_pos, quat_new])
    raise ValueError("Pose must be length 3 (xyz) or 7 (xyzxyzw)")


def get_quat_for_xy_plane(quat: np.ndarray) -> np.ndarray:
    """Get the quaternion for the xy plane - extract only yaw rotation

    Args:
        quat: Quaternion rotation in xyzw

    Returns:
        Quaternion for XY plane

    """
    qx, qy, qz, qw = quat

    # Extract yaw directly from quaternion components
    # For a quaternion representing ZYX euler angles:
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    # Create a pure yaw rotation
    return R.from_euler("z", yaw).as_quat()


def get_tracker_data_fixed_arm(data, h_pose_raw: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    # Integrate ZMQ position packets at the start - receive latest network data
    # Check if we're using ZMQ objects (they have a 'receiver' attribute)

    if data is None:
        return None

    # IMPORTANT: Always extract poses when available, regardless of grip state
    # The main loop will decide whether to use them for arm control based on grip state
    # This ensures arms work properly when grips are pressed

    # Extract headset pose
    if h_pose_raw is None:
        h_pose_raw = np.array(data["headset"]["pose"], dtype=np.float64)
    h_pose = convert_unity_to_world_coordinates(h_pose_raw)

    lc_pose_raw = np.array(data["left_controller"]["pose"], dtype=np.float64)
    lc_pose = convert_unity_to_world_coordinates(lc_pose_raw)

    rc_pose_raw = np.array(data["right_controller"]["pose"], dtype=np.float64)
    rc_pose = convert_unity_to_world_coordinates(rc_pose_raw)

    # Get Elbow Positions
    # Raw elbow positions are relative to headset position.
    # Make elbow X/Y relative to head X/Y
    left_elbow_pos = lc_pose[:3] - np.array([h_pose[0], h_pose[1], 0])
    right_elbow_pos = rc_pose[:3] - np.array([h_pose[0], h_pose[1], 0])

    # Make elbow Z relative to headset Z Headset Z is always at nominal height.
    # Now you can sit down as long as your upper body is still at same configuration
    nominal_headset_height = 1.65
    left_elbow_pos[2] = nominal_headset_height - (h_pose[2] - lc_pose[2])
    right_elbow_pos[2] = nominal_headset_height - (h_pose[2] - rc_pose[2])

    # Rotate elbow positions to be relative to the headset Yaw orientation.
    left_elbow_pos = R.from_quat(get_quat_for_xy_plane(h_pose[3:])).inv().apply(left_elbow_pos)
    right_elbow_pos = R.from_quat(get_quat_for_xy_plane(h_pose[3:])).inv().apply(right_elbow_pos)

    return left_elbow_pos, right_elbow_pos
