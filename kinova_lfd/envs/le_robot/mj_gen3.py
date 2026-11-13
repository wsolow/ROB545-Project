"""LeRobot Robot wrapping KinovaLfD ``Mj-Gen3-v0``."""

import logging
from typing import Any

from kinova_lfd.envs.factory import get_env_cfg

from .base import Gen3LfDRobot
from .config_mj_gen3 import MjGen3RobotConfig

logger = logging.getLogger(__name__)


class MjGen3Robot(Gen3LfDRobot):
    """Simulated Gen3 arm in MuJoCo, exposed through the LeRobot ``Robot`` API."""

    config_class = MjGen3RobotConfig
    name = "mj_gen3"

    def __init__(self, config: MjGen3RobotConfig):
        super().__init__(config)
        self.config: MjGen3RobotConfig = config
        self.task_name = config.task_name

    def _env_cfg_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "enable_cameras": self.config.enable_cameras,
            "interpolate_joint_targets": self.config.interpolate_joint_targets,
        }
        if self.config.dt is not None:
            # One env.step per control tick: physics substeps at the task's timestep, while IK,
            # forward kinematics, and camera rendering run once per tick.
            physics_dt = get_env_cfg(self.task_name).sim.mujoco.timestep
            decimation = max(1, round(self.config.dt / physics_dt))
            if abs(decimation * physics_dt - self.config.dt) > 1e-9:
                logger.warning(
                    "dt=%.4f s is not a multiple of the physics timestep %.4f s; env.step will span %.4f s.",
                    self.config.dt,
                    physics_dt,
                    decimation * physics_dt,
                )
            cfg["decimation"] = decimation
        if self.config.episode_length_s is not None:
            cfg["episode_length_s"] = self.config.episode_length_s
        return cfg
