from typing import Literal

import torch

from kinova_lfd.core.tensor_math import euler_xyz_from_quat, quat_error_magnitude, quat_from_euler_xyz
from kinova_lfd.envs.base.mujoco.configclass import configclass


@configclass
class WaypointCfg:
    """Configuration class for trajectory waypoints."""

    max_points: int | None = 20
    """Maximum number of waypoints in a trajectory."""
    max_dist: float | None = 0.10
    """Maximum distance between waypoints in a trajectory."""
    max_rad: float | None = 0.75
    """Maximum radian error between waypoitns in a trajectory."""
    device: str = "cuda"
    """Torch device to use."""
    return_type: Literal["euler", "quat"] = "euler"

    def build(self):
        return TrajectoryWaypoints(self)


class TrajectoryWaypoints:
    """Controller for setting intermediate waypoints between current and goal poses.

    Goal of obtaining better performing IK.
    """

    def __init__(self, cfg: WaypointCfg) -> None:
        self.cfg = cfg

    def get_next_point(self, start_pose: torch.Tensor, goal_pose: torch.Tensor) -> torch.Tensor:
        """Get the next waypoint in the trajectory."""
        # Handle either Quaternion (ndim=7) or euler angles (ndim=6)
        if start_pose.shape[1] == 6:
            start_quat = quat_from_euler_xyz(start_pose[:, 3], start_pose[:, 4], start_pose[:, 5])
            start_pose = torch.cat((start_pose[:, :3], start_quat), dim=1)
        if goal_pose.shape[1] == 6:
            goal_quat = quat_from_euler_xyz(goal_pose[:, 3], goal_pose[:, 4], goal_pose[:, 5])
            goal_pose = torch.cat((goal_pose[:, :3], goal_quat), dim=1)

        way_pose = self._interpolate_pose(start_pose[:, :3], start_pose[:, 3:7], goal_pose[:, :3], goal_pose[:, 3:7])
        if self.cfg.return_type == "euler":
            r, p, y = euler_xyz_from_quat(way_pose[:, 1, 3:7])
            return torch.cat((way_pose[:, 1, :3], torch.stack((r, p, y), dim=-1)), dim=-1)
        return way_pose[:, 1]

    def _interpolate_pose(
        self,
        start_pos: torch.Tensor,
        start_quat: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Return (positions, quaternions) each of shape (..., n_waypoints,7)."""
        rot_err = quat_error_magnitude(start_quat, goal_quat)
        pos_err = torch.linalg.vector_norm(start_pos - goal_pos, dim=-1)

        max_waypoints = torch.max(torch.ceil(pos_err / self.cfg.max_dist), torch.ceil(rot_err / self.cfg.max_rad)) + 1
        num_waypoints = torch.max(
            torch.minimum(max_waypoints, torch.as_tensor(self.cfg.max_points, device=self.cfg.device))
        ).to(torch.int32)
        t = torch.linspace(
            0,
            1,
            num_waypoints,
            device=self.cfg.device,
        )

        t_shape = (1,) * (start_pos.dim() - 1) + (num_waypoints,)
        t = t.reshape(t_shape).repeat(start_pos.shape[0], 1)

        p0 = start_pos.unsqueeze(-2)
        p1 = goal_pos.unsqueeze(-2)
        q0 = start_quat.unsqueeze(-2)
        q1 = goal_quat.unsqueeze(-2)

        positions = self._lerp_poses(p0, p1, t.unsqueeze(-1))
        quaternions = self._slerp_quats(q0, q1, t.unsqueeze(-1))

        return torch.cat((positions, quaternions), dim=-1).squeeze(-2)

    def _lerp_poses(self, p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two positiontorch.tensors."""
        t = t.unsqueeze(-1) if t.shape != p0.shape else t
        return (1 - t) * p0 + t * p1

    def _slerp_quats(self, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions.

        Handles quaternion double-cover (q == -q) and the sin(theta) -> 0 fallback.

        q0, q1: (..., 4) in [x, y, z, w]
        t:      (..., 1) or (...,) scalar weights in [0, 1]
        returns: (..., 4) normalized quaternion
        """
        t = t.unsqueeze(-1) if t.shape != q0.shape else t

        q0 = q0 / q0.norm(dim=-1, keepdim=True)
        q1 = q1 / q1.norm(dim=-1, keepdim=True)

        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()

        dot = dot.clamp(-1.0, 1.0)
        theta = torch.acos(dot)

        sin_theta = torch.sin(theta)
        safe = sin_theta.abs() > 1e-6

        coeff0 = torch.where(safe, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
        coeff1 = torch.where(safe, torch.sin(t * theta) / sin_theta, t)

        q = coeff0 * q0 + coeff1 * q1
        return q / q.norm(dim=-1, keepdim=True)


def main():
    waypoint_cfg = WaypointCfg(max_points=10, max_dist=0.05, max_rad=1)
    waypoint_maker = waypoint_cfg.build()

    start = torch.as_tensor([[0.0, 0, 0, 1, 0, 0, 0]], device="cuda", dtype=torch.float32)
    end = torch.as_tensor([[0.0, 0, 1, 1, 0, 0, 0]], device="cuda", dtype=torch.float32)

    waypoint_maker.get_next_point(start, end)


if __name__ == "__main__":
    main()
