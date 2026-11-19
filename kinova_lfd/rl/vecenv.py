"""RSL-RL vectorized environment wrapper for :class:`MjDirectRlEnv`.

mjlab ships an equivalent wrapper (``mjlab.rl.RslRlVecEnvWrapper``), but it requires a ``ManagerBasedRlEnv``
(action/observation managers), so direct-style environments need this thin adapter instead.
"""

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from kinova_lfd.envs.base.mujoco.mj_env import MjDirectRlEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wrap a :class:`MjDirectRlEnv` for the RSL-RL library.

    This must be the last wrapper in the chain since it does not follow the :class:`gym.Wrapper` interface.
    """

    def __init__(self, env: MjDirectRlEnv, clip_actions: float | None = None) -> None:
        """Initialize the wrapper and reset the environment (the RSL-RL runner never calls reset).

        Args:
            env: The environment to wrap.
            clip_actions: Symmetric clipping value for actions. If ``None``, no clipping is done.

        """
        if not isinstance(env, MjDirectRlEnv):
            raise TypeError(f"The environment must inherit from MjDirectRlEnv. Environment type: {type(env)}")
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.unwrapped.max_episode_length
        self.num_actions = gym.spaces.flatdim(env.unwrapped.single_action_space)

        if clip_actions is not None:
            self.single_action_space = gym.spaces.Box(low=-clip_actions, high=clip_actions, shape=(self.num_actions,))
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        self.env.reset()

    @property
    def cfg(self) -> object:
        """Environment configuration object."""
        return self.env.unwrapped.cfg

    @property
    def unwrapped(self) -> MjDirectRlEnv:
        """Return the base environment."""
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer for current episode lengths."""
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        """Seed the environment."""
        return self.env.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments."""
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        """Return the current observations of the environment."""
        return TensorDict(self.env.unwrapped._get_observations(), batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and convert the result to the RSL-RL convention."""
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # Time-out information is only needed for infinite-horizon tasks (value bootstrapping).
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()

    def __repr__(self) -> str:
        """Return the wrapper name and the wrapped environment representation."""
        return f"<{type(self).__name__}{self.env}>"
