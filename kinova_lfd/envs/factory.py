"""Construct a manipulation environment from a task name."""

from typing import Any

from kinova_lfd.envs.base.gym import GymEnv


def create_env(task_name: str, cfg: dict[str, Any]) -> GymEnv:
    """Create a Kortex environment for the given task name and configuration."""
    if task_name == "Kortex-TCP-Gen3-v0":
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from kinova_lfd.envs.tasks.kortex_tcp_gen3 import Gen3KortexTcpEnv, Gen3KortexTcpEnvCfg

        env_cfg = Gen3KortexTcpEnvCfg(**cfg)
        return Gen3KortexTcpEnv(cfg=env_cfg)
    if task_name == "Kortex-UDP-Gen3-v0":
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from kinova_lfd.envs.tasks.kortex_udp_gen3 import Gen3KortexUdpEnv, Gen3KortexUdpEnvCfg

        env_cfg = Gen3KortexUdpEnvCfg(**cfg)
        return Gen3KortexUdpEnv(cfg=env_cfg)
    if task_name == "Mj-Gen3-v0":
        from kinova_lfd.envs.tasks.mj_gen3 import MjGen3Env, MjGen3EnvCfg

        env_cfg = MjGen3EnvCfg()
        env_cfg.sim.device = "cuda"
        return MjGen3Env(cfg=env_cfg)

    raise ValueError(f"Invalid task name: {task_name}")
