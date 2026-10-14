"""Manager for observations for S2R transfer."""

from pathlib import Path

import torch
import yaml
from tensordict import TensorDict

from kinova_lfd.envs.base.gym import GymEnv


class ObservationManager:
    """The observation manager class for facilitating sim to real transfer."""

    def __init__(self, terms: list[str], env: GymEnv | None = None) -> None:
        self.terms = terms
        self.env = env

    @property
    def obs(self) -> dict[str, torch.Tensor]:
        """Return the observations based on the config."""
        raw_obs = {
            "joint_pos": self.env._joint_positions[:, self.env.cfg.joint_ids],
            "joint_vel": self.env._joint_velocities[:, self.env.cfg.joint_ids],
            "tcp_pose_b": self.env._tcp_pose_b,
            "ee_vel_b": self.env._ee_vel_b,
            "prev_actions": self.env._prev_actions,
            "cube_pos": self.env.cube_pose_b[:, 0:3] if hasattr(self.env, "cube_pos") else None,
            "cube_goal": self.env.cube_goal_xyz_b[:, 0:3] if hasattr(self.env, "cube_goal") else None,
            "reach_goal": self.env.goal_ee_xyz_b if hasattr(self.env, "goal_ee_xyz_b") else None,
        }
        # Only return keys that are in the config
        return {k: v for k, v in raw_obs.items() if k in self.terms}

    def obs_vec(self) -> torch.Tensor:
        """Observations as a single vector."""
        return torch.cat(list(self.obs.values()), dim=-1)

    def obs_from_dict(self, obs: TensorDict) -> torch.Tensor:
        """Observations from a SkilletEnv tensordict obs spec."""
        return torch.cat([obs[v] for v in self.terms if v in obs], dim=1)

    def save(self, path: str | Path):
        """Save the observation manager."""
        with Path(path).open("w") as f:
            yaml.dump({"obs": self.terms}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ObservationManager":
        """Load the action manager from path."""
        with Path(path).open() as f:
            terms = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(terms["obs"])
