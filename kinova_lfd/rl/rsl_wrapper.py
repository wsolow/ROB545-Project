"""Environment definitions for compatibility with the RSL-RL library."""

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast, override

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.envs.base.mujoco.mj_env import MjDirectRlEnv
from kinova_lfd.envs.base.seed import configure_seed


class RslRlVecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that the same type of
    action is applied to all environments and the same type of observation is returned from all environments.
    """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """Number of environments."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of actions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_episode_length(self) -> int | torch.Tensor:
        """Maximum episode length."""
        raise NotImplementedError

    @property
    @abstractmethod
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer for current episode lengths."""
        raise NotImplementedError

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the episode length buffer."""
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device | str:
        """Device to use."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cfg(self) -> dict | object:
        """Configuration object."""
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            The observations from the environment.

        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            observations: Observations from the environment.
            rewards: Rewards from the environment. Shape: (num_envs,)
            dones: Done flags from the environment. Shape: (num_envs,)
            extras: Extra information from the environment.

        Observations:
            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps from required observation sets (e.g. actor) to lists of observation groups.
            The observation sets (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "actor": Specified observation groups are used as input to the actor network.
            - "critic": Specified observation groups are used as input to the critic network.
            - "student": Specified observation groups are used as input to the student network.
            - "teacher": Specified observation groups are used as input to the teacher network.
            - "rnd_state": Specified observation groups are used as input to the RND network.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`, which provides detailed information on the expected configuration.

        Extras:
            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that
               happen due to time limits and not due to the environment reaching a terminal state. This is useful
               for environments that have a fixed episode length.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.

        """
        raise NotImplementedError


class RslRlVecEnvWrapper(RslRlVecEnv):
    """Wraps around Isaac Lab environment for the RSL-RL library.

    .. caution::
        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: MjDirectRlEnv, clip_actions: float | None = None) -> None:
        """Initialize the wrapper.

        The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRlEnv` or :class:`DirectRlEnv`.

        """
        # check that input is valid
        if not isinstance(env, MjDirectRlEnv):
            raise TypeError(f"The environment must be inherited from MjDirectRlEnv. Environment type: {type(env)}")
        super().__init__()
        self.env: MjDirectRlEnv = env

        self._clip_actions = clip_actions

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self._num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            try:
                self._num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
            except AttributeError:
                print(
                    f"[WARN] Manually flattening `{self.unwrapped.single_action_space}`, assuming type mjlab.utils.spaces.Box"
                )
                self._num_actions = int(np.prod(self.unwrapped.single_action_space.shape))

        # modify the action space to the clip range
        if self._clip_actions is not None:
            self.single_action_space = gym.spaces.Box(
                low=-self._clip_actions, high=self._clip_actions, shape=(self._num_actions,)
            )
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    # ==================== RSL RL Properties ====================

    @property
    @override
    def num_envs(self) -> int:
        """Number of environments."""
        return self.env.num_envs

    @property
    @override
    def num_actions(self) -> int:
        return self._num_actions

    @property
    @override
    def max_episode_length(self) -> int | torch.Tensor:
        """Maximum episode length."""
        return self.env.unwrapped.max_episode_length

    @property
    @override
    def episode_length_buf(self) -> torch.Tensor:
        """Buffer for current episode lengths."""
        if not hasattr(self.env.unwrapped, "episode_length_buf"):
            raise TypeError("The environment is not a DirectRlInterface.")
        drl_env = cast("GymEnv", self.env.unwrapped)
        return drl_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the episode length buffer."""
        if not hasattr(self.env.unwrapped, "episode_length_buf"):
            raise TypeError("The environment is not a DirectRlInterface.")
        drl_env = cast("GymEnv", self.env.unwrapped)
        drl_env.episode_length_buf = value

    @property
    @override
    def device(self) -> torch.device | str:
        """Device to use."""
        return self.env.device

    @property
    @override
    def cfg(self) -> dict | object:
        """Configuration object."""
        return self.env.unwrapped.cfg

    @property
    @override
    def unwrapped(self) -> GymEnv:
        """Return the base environment, which is a DirectRlInterface."""
        return self.env.unwrapped

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        """Seed the environment."""
        return configure_seed(seed)

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    @override
    def get_observations(self) -> TensorDict:
        """Return the current observations of the environment."""
        if isinstance(self.env, MjDirectRlEnv):  # Is a MjDirectRlEnv
            obs_dict = self.unwrapped._get_observations()
        elif hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()

        return TensorDict(obs_dict, batch_size=[self.num_envs])

    @override
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self._clip_actions is not None:
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    # ==================== Misc ====================

    @override
    def __str__(self) -> str:
        """Return the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    @override
    def __repr__(self) -> str:
        """Return the string representation of the wrapper."""
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        """Return the class name of the wrapper."""
        return cls.__name__
