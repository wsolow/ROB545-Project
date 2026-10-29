"""LeRobot Robot wrapping KinovaLfD ``Mj-Gen3-v0``."""

from __future__ import annotations

from typing import Any

from kinova_lfd.envs.base.gym import GymEnv

from .base import Gen3LfDRobot
from .config_mj_gen3 import MjGen3RobotConfig


class MjGen3Robot(Gen3LfDRobot):
    """Simulated Gen3 arm in MuJoCo, exposed through the LeRobot ``Robot`` API."""

    config_class = MjGen3RobotConfig
    name = "mj_gen3"
    task_name = "Mj-Gen3-v0"

    def __init__(self, config: MjGen3RobotConfig):
        super().__init__(config)
        self.config: MjGen3RobotConfig = config

    def _env_cfg_dict(self) -> dict[str, Any]:
        # Factory currently ignores the cfg dict for Mj and hardcodes cuda; keep
        # the device here so ``_post_create_env`` can still honor the config.
        return {"device": self.config.device}

    def _post_create_env(self, env: GymEnv) -> None:
        cfg = getattr(env, "cfg", None)
        sim = getattr(cfg, "sim", None) if cfg is not None else None
        if sim is not None and hasattr(sim, "device"):
            sim.device = self.config.device
