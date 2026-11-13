"""LeRobot Robot wrapping KinovaLfD ``Mj-Gen3-v0``."""

from typing import Any

from .base import Gen3LfDRobot
from .config_mj_gen3 import MjGen3RobotConfig


class MjGen3Robot(Gen3LfDRobot):
    """Simulated Gen3 arm in MuJoCo, exposed through the LeRobot ``Robot`` API."""

    config_class = MjGen3RobotConfig
    name = "mj_gen3"

    def __init__(self, config: MjGen3RobotConfig):
        super().__init__(config)
        self.config: MjGen3RobotConfig = config
        self.task_name = config.task_name

    def _env_cfg_dict(self) -> dict[str, Any]:
        return {
            "enable_cameras": self.config.enable_cameras,
        }
