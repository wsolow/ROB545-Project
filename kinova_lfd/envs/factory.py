"""Construct a manipulation environment from a task name."""

import dataclasses
from typing import Any

from kinova_lfd.envs.base.gym import GymEnv

MJ_TASKS = ("Mj-Gen3-v0", "Mj-Gen3-LiftCube-v0")
"""Names of the MuJoCo tasks that ``get_env_cfg`` can build a dataclass config for."""


def _apply_cfg(cfg: object, overrides: dict[str, Any] | object | None) -> object:
    """Apply top-level field overrides (dict) to a config dataclass, or return a full config as-is."""
    if overrides is None:
        return cfg
    if isinstance(overrides, dict):
        return dataclasses.replace(cfg, **overrides)
    return overrides


def get_env_cfg(task_name: str, overrides: dict[str, Any] | object | None = None) -> object:
    """Construct the local configuration for a task without using Gym registration."""
    task_name = task_name.split(":")[-1]
    if task_name == "Mj-Gen3-v0":
        from kinova_lfd.envs.tasks.mj_gen3 import MjGen3EnvCfg

        return _apply_cfg(MjGen3EnvCfg(), overrides)
    if task_name == "Mj-Gen3-LiftCube-v0":
        from kinova_lfd.envs.tasks.mj_gen3_lift_cube import MjGen3LiftCubeEnvCfg

        return _apply_cfg(MjGen3LiftCubeEnvCfg(), overrides)
    raise ValueError(f"Invalid task name: {task_name}")


def create_env(task_name: str, cfg: dict[str, Any] | object | None = None) -> GymEnv:
    """Create an environment from the local task catalog and configuration."""
    task_name = task_name.split(":")[-1]
    cfg = cfg or {}
    if task_name == "Kortex-TCP-Gen3-v0":
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from kinova_lfd.envs.tasks.kortex_tcp_gen3 import Gen3KortexTcpEnv, Gen3KortexTcpEnvCfg

        env_cfg = Gen3KortexTcpEnvCfg(**cfg) if isinstance(cfg, dict) else cfg
        return Gen3KortexTcpEnv(cfg=env_cfg)
    if task_name == "Kortex-UDP-Gen3-v0":
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from kinova_lfd.envs.tasks.kortex_udp_gen3 import Gen3KortexUdpEnv, Gen3KortexUdpEnvCfg

        env_cfg = Gen3KortexUdpEnvCfg(**cfg) if isinstance(cfg, dict) else cfg
        return Gen3KortexUdpEnv(cfg=env_cfg)
    if task_name == "Mj-Gen3-v0":
        from kinova_lfd.envs.tasks.mj_gen3 import MjGen3Env

        env_cfg = get_env_cfg(task_name, cfg)
        return MjGen3Env(cfg=env_cfg)
    if task_name == "Mj-Gen3-LiftCube-v0":
        from kinova_lfd.envs.tasks.mj_gen3_lift_cube import MjGen3LiftCubeEnv

        env_cfg = get_env_cfg(task_name, cfg)
        return MjGen3LiftCubeEnv(cfg=env_cfg)

    raise ValueError(f"Invalid task name: {task_name}")
