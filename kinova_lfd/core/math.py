import numpy as np


def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).

    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions (xyzw format)
    xyz = quat[:, :3]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec + quat[:, 3:4] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (x, y, z, w). Shape is (..., 4).

    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((-q[..., :3], q[..., 3:4]), axis=-1).reshape(shape)


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (x, y, z, w). Shape is (N, 4).

    """
    return quat_conjugate(q) / np.pow(q, 2).sum(axis=-1).clip(min=eps)


def euler_xyz_from_quat(quat: np.ndarray, wrap_to_2pi: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    """
    q_x, q_y, q_z, q_w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sin_pitch) >= 1, copysign(np.pi / 2.0, sin_pitch), np.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    return roll, pitch, yaw


def copysign(mag: float, other: np.ndarray) -> np.ndarray:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.

    """
    mag_torch = abs(mag) * np.ones_like(other)
    return np.copysign(mag_torch, other)


def euler_xyz_to_rotvec(euler: np.ndarray) -> np.ndarray:
    """Convert XYZ extrinsic Euler angles to rotation vector.

    Args:
        euler: tensor of shape (N, 3) in radians [rx, ry, rz].

    Returns:
        Torch tensor rotation vectors of shape (N, 3).

    """
    rx = euler[0]
    ry = euler[1]
    rz = euler[2]

    zeros = np.zeros_like(rx)
    ones = np.ones_like(rx)

    # Each rotation matrix: stack into (N, 3, 3)
    # Rotation about X
    Rx = np.stack(
        [
            np.asarray([ones, zeros, zeros]),
            np.asarray([zeros, np.cos(rx), -np.sin(rx)]),
            np.asarray([zeros, np.sin(rx), np.cos(rx)]),
        ],
        axis=0,
    )  # (N, 3, 3)

    # Rotation about Y
    Ry = np.stack(
        [
            np.asarray([np.cos(ry), zeros, np.sin(ry)]),
            np.asarray([zeros, ones, zeros]),
            np.asarray([-np.sin(ry), zeros, np.cos(ry)]),
        ],
        axis=0,
    )

    # Rotation about Z
    Rz = np.stack(
        [
            np.asarray([np.cos(rz), -np.sin(rz), zeros]),
            np.asarray([np.sin(rz), np.cos(rz), zeros]),
            np.asarray([zeros, zeros, ones]),
        ],
        axis=0,
    )

    # Combined: XYZ extrinsic = Z @ Y @ X, shape (N, 3, 3)
    R = np.matmul(Rz, np.matmul(Ry, Rx))

    # Trace of each matrix: (N,)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = np.acos(((trace - 1) / 2).clip(-1, 1))

    # Axis from skew-symmetric part: (N, 3)
    axis = np.stack(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        axis=0,
    )

    # Normalize axis, guarding near-zero angles
    sin_angle = np.sin(angle)
    axis = axis / (2 * sin_angle + 1e-8)

    rotvec = axis * angle

    # Zero out near-singularity cases (angle ~ 0)
    near_zero = np.abs(angle) < 1e-6
    rotvec[near_zero] = 0.0

    return rotvec


def base_to_tcp_twist(
    linear_vel: np.ndarray,
    angular_vel: np.ndarray,
    tcp_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a twist from base frame to TCP frame.

    Quaternions are [w, x, y, z] following IsaacLab convention.

    Args:
        linear_vel: linear velocity in XYZ (m/s)
        angular_vel: angular velocity in RPY (rad/s)
        tcp_quat: The current orientation of the TCP frame

    Returns:
        Linear and angular velocity tensors in base frame.

    """
    tcp_quat_inv = quat_inv(tcp_quat)

    # Rotate linear and angular velocities into TCP frame
    linear_vel_tcp = quat_apply(tcp_quat_inv, linear_vel)
    angular_vel_tcp = quat_apply(tcp_quat_inv, angular_vel)

    return linear_vel_tcp, angular_vel_tcp


def np_euler_xyz_degrees_from_quat(
    quat: np.ndarray, wrap_to_2pi: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert rotations given as quaternions to Euler angles in degrees.

    The euler angles are assumed in XYZ extrinsic convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        wrap_to_2pi (bool): Whether to wrap output Euler angles into [0, 2π). If
            False, angles are returned in the default range (−π, π]. Defaults to
            False.

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (1,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    """
    q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sin_pitch) >= 1, np.copysign(np.array(np.pi / 2.0), sin_pitch), np.arcsin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    if wrap_to_2pi:
        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def np_quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Convert rotations given as Euler angles in radians to Quaternions.

    The euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (1,).
        pitch: Rotation around y-axis (in radians). Shape is (1,).
        yaw: Rotation around z-axis (in radians). Shape is (1,).

    Returns:
        The quaternion in (x, y, z, w). Shape is (N, 4).

    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=0)


def np_convert_quat(quat: np.ndarray, to: str = "xyzw") -> np.ndarray:
    """Convert quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (4,).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).

    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # convert to specified quaternion type
    if to == "xyzw":
        # wxyz -> xyzw
        return np.roll(quat, -1, axis=0)
    # xyzw -> wxyz
    return np.roll(quat, 1, axis=0)
