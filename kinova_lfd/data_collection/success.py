"""Episode success predicates for policy demonstrations."""

import torch


def cube_goal_success(cube_pose: torch.Tensor, goal_pose: torch.Tensor, delta: float) -> bool:
    """Return whether the first cube position is within ``delta`` of its goal."""
    if cube_pose.ndim != 2 or goal_pose.ndim != 2 or cube_pose.shape[0] == 0 or goal_pose.shape[0] == 0:
        raise ValueError("Cube and goal poses must be non-empty batched tensors.")
    distance = torch.linalg.vector_norm(cube_pose[0, :3] - goal_pose[0, :3]).item()
    return distance <= delta
