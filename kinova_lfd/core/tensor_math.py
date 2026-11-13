"""Torch math utilities using (x, y, z, w) quaternions.

Torch counterpart of :mod:`kinova_lfd.core.np_math`. Every quaternion here is
(x, y, z, w), the convention used at the boundary of every environment (Kortex and
MuJoCo). MuJoCo-internal code uses :mod:`kinova_lfd.core.mj_tensor_math` (w, x, y, z).

Written by Will Solow, 2026.
"""

import torch


def convert_quat(quat: torch.Tensor, to: str = "xyzw") -> torch.Tensor:
    """Convert quaternion from one convention to another.

    If ``to == "xyzw"`` the input is (w, x, y, z); if ``to == "wxyz"`` the input is (x, y, z, w).

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert the quaternion to. Defaults to "xyzw".

    Returns:
        The converted quaternion, shape (..., 4).

    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4).")
    if to == "xyzw":
        return quat.roll(-1, dims=-1)
    if to == "wxyz":
        return quat.roll(1, dims=-1)
    raise ValueError(f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}.")


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (x, y, z, w). Shape is (..., 4).

    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1).view(shape)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (x, y, z, w). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (x, y, z, w). Shape is (N, 4).

    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (x, y, z, w). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).

    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 3:4] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.compile
def euler_xyz_to_rotvec(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ extrinsic Euler angles to rotation vector.

    Args:
        euler: tensor of shape (N, 3) in radians [rx, ry, rz].

    Returns:
        Torch tensor rotation vectors of shape (N, 3).

    """
    rx = euler[:, 0]
    ry = euler[:, 1]
    rz = euler[:, 2]

    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    # Each rotation matrix: stack into (N, 3, 3)
    # Rotation about X
    Rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
            torch.stack([zeros, torch.sin(rx), torch.cos(rx)], dim=1),
        ],
        dim=1,
    )  # (N, 3, 3)

    # Rotation about Y
    Ry = torch.stack(
        [
            torch.stack([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-torch.sin(ry), zeros, torch.cos(ry)], dim=1),
        ],
        dim=1,
    )

    # Rotation about Z
    Rz = torch.stack(
        [
            torch.stack([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
            torch.stack([torch.sin(rz), torch.cos(rz), zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )

    # Combined: XYZ extrinsic = Z @ Y @ X, shape (N, 3, 3)
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # Trace of each matrix: (N,)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = torch.acos(((trace - 1) / 2).clamp(-1, 1))

    # Axis from skew-symmetric part: (N, 3)
    axis = torch.stack(
        [
            R[:, 2, 1] - R[:, 1, 2],
            R[:, 0, 2] - R[:, 2, 0],
            R[:, 1, 0] - R[:, 0, 1],
        ],
        dim=1,
    )

    # Normalize axis, guarding near-zero angles
    sin_angle = torch.sin(angle).unsqueeze(1)
    axis = axis / (2 * sin_angle + 1e-8)

    rotvec = axis * angle.unsqueeze(1)

    # Zero out near-singularity cases (angle ~ 0)
    near_zero = angle.abs() < 1e-6
    rotvec[near_zero] = 0.0

    return rotvec


def base_to_tcp_twist(
    linear_vel: torch.Tensor,
    angular_vel: torch.Tensor,
    tcp_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform a twist from base frame to TCP frame.

    Args:
        linear_vel: Linear velocity in XYZ (m/s). Shape is (N, 3).
        angular_vel: Angular velocity in RPY (rad/s). Shape is (N, 3).
        tcp_quat: Current TCP orientation in (x, y, z, w). Shape is (N, 4).

    Returns:
        Linear and angular velocity tensors in the TCP frame.

    """
    tcp_quat_inv = quat_inv(tcp_quat)
    linear_vel_tcp = quat_apply(tcp_quat_inv, linear_vel)
    angular_vel_tcp = quat_apply(tcp_quat_inv, angular_vel)
    return linear_vel_tcp, angular_vel_tcp
