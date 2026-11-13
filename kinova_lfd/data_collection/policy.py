"""RSL-RL policy action source and lift-cube success checks for MuJoCo Gen3 recording.

The RSL-RL stack is imported lazily in :meth:`RslRlPolicySource.connect`, so the success
helpers can be imported cheaply (e.g. by diffusion playback).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from kinova_lfd.envs.le_robot.base import JOINT_ACTION_KEYS

from .configs import RslRlPolicyConfig

if TYPE_CHECKING:
    from kinova_lfd.rl.rsl_rl.runners import OnPolicyRunner
    from kinova_lfd.rl.rsl_wrapper import RslRlVecEnvWrapper


def cube_goal_success(cube_pose: torch.Tensor, goal_pose: torch.Tensor, delta: float) -> bool:
    """Return whether the first cube position is within ``delta`` of its goal."""
    if cube_pose.ndim != 2 or goal_pose.ndim != 2 or cube_pose.shape[0] == 0 or goal_pose.shape[0] == 0:
        raise ValueError("Cube and goal poses must be non-empty batched tensors.")
    distance = torch.linalg.vector_norm(cube_pose[0, :3] - goal_pose[0, :3]).item()
    return distance <= delta


def policy_episode_succeeded(robot: Any, success_delta: float) -> bool:
    """Return whether the robot env's final lift-cube position is within ``success_delta`` of the goal."""
    env = getattr(robot, "env", None)
    cube_pose = getattr(env, "last_cube_pose_w", getattr(env, "cube_pose_w", None))
    goal_pose = getattr(env, "last_cube_goal_pose_w", getattr(env, "cube_goal_pose_w", None))
    if cube_pose is None or goal_pose is None:
        raise TypeError("Automatic policy success is only supported by MjGen3LiftCubeEnv.")
    return cube_goal_success(cube_pose, goal_pose, success_delta)


class RslRlPolicySource:
    """Teleoperator-compatible wrapper around a deterministic RSL-RL policy."""

    def __init__(self, config: RslRlPolicyConfig, robot: Any):
        self.config = config
        self.robot = robot
        self.runner: OnPolicyRunner | None = None
        self.policy_env: RslRlVecEnvWrapper | None = None
        self.policy: Any = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        """Load the checkpoint against the already-connected robot environment."""
        if self._is_connected:
            return
        if self.robot.env is None:
            raise RuntimeError("Connect the robot before connecting the policy source.")
        if self.robot.task_name != self.config.task:
            raise ValueError(f"Policy task {self.config.task!r} does not match robot task {self.robot.task_name!r}.")

        from kinova_lfd.rl.cfg.default_rsl_rl import make_default_rsl_rl_cfg
        from kinova_lfd.rl.rsl_rl.runners import OnPolicyRunner
        from kinova_lfd.rl.rsl_wrapper import RslRlVecEnvWrapper
        from kinova_lfd.rl.util.hydra import _load_agent_cfg

        self.policy_env = RslRlVecEnvWrapper(self.robot.env, clip_actions=self.config.clip_actions)
        agent_cfg = _load_agent_cfg(self.config.agent)
        if agent_cfg is None:
            agent_cfg = make_default_rsl_rl_cfg()
        if isinstance(agent_cfg, dict):
            agent_cfg["device"] = self.config.device
            agent_cfg_dict = agent_cfg
        else:
            agent_cfg.device = self.config.device
            agent_cfg_dict = agent_cfg.to_dict()
        self.runner = OnPolicyRunner(
            self.policy_env,
            agent_cfg_dict,
            log_dir=None,
            device=self.config.device,
        )
        self.runner.load(str(Path(self.config.checkpoint).expanduser().resolve()))
        self.policy = self.runner.get_inference_policy(device=self.policy_env.unwrapped.device)
        self._is_connected = True
        self.reset()

    def disconnect(self) -> None:
        """Release policy references without closing the robot environment."""
        self.policy = None
        self.runner = None
        self.policy_env = None
        self._is_connected = False

    def reset(self) -> None:
        """Reset recurrent inference state when the policy provides it."""
        if self.runner is not None and hasattr(self.runner.alg.policy, "reset"):
            self.runner.alg.policy.reset()

    def get_action(self) -> dict[str, float]:
        """Infer one joint-position action from the current native task observation."""
        if not self._is_connected or self.policy_env is None or self.policy is None:
            raise RuntimeError("Policy source is not connected.")
        observations = self.policy_env.get_observations()
        action = self.policy(observations)
        values = action.detach().to(device="cpu", dtype=torch.float32).reshape(-1).tolist()
        if len(values) != len(JOINT_ACTION_KEYS):
            raise ValueError(f"Expected {len(JOINT_ACTION_KEYS)} policy actions, got {len(values)}.")
        return dict(zip(JOINT_ACTION_KEYS, values, strict=True))
