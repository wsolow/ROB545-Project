from typing import Any

from kinova_lfd.envs.base.gym import GymEnv


def create_env(task_name: str, cfg: dict[str, Any]) -> GymEnv:
    """Create a Kortex environment for the given task name and configuration."""
    if task_name == "Kortex-Gen3-v0":
        import os

        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        from kinova_lfd.envs.base.kortex.kortex_bridge import setup_kortex_tcp_udp
        from kinova_lfd.envs.tasks.kortex_gen3 import Gen3KortexEnv, Gen3KortexEnvCfg

        tcp_udp = setup_kortex_tcp_udp(ip=cfg["robot_ip"])
        env_cfg = Gen3KortexEnvCfg(**cfg)
        return Gen3KortexEnv(cfg=env_cfg, kortex_tcp_udp=tcp_udp)
    if task_name == "Mj-Gen3-v0":
        from kinova_lfd.envs.tasks.mj_gen3 import MjGen3Env, MjGen3EnvCfg

        env_cfg = MjGen3EnvCfg()
        env_cfg.sim.device = "cuda"
        return MjGen3Env(cfg=env_cfg)

    raise ValueError(f"Invalid task name: {task_name}")
