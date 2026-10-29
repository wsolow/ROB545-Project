"""LeRobot Robot wrapping KinovaLfD ``Kortex-TCP-Gen3-v0``."""

from __future__ import annotations

from typing import Any

from .base import Gen3LfDRobot
from .config_kortex_tcp_gen3 import KortexTcpGen3RobotConfig


class KortexTcpGen3Robot(Gen3LfDRobot):
    """Real Gen3 arm via Kortex TCP, exposed through the LeRobot ``Robot`` API."""

    config_class = KortexTcpGen3RobotConfig
    name = "kortex_tcp_gen3"
    task_name = "Kortex-TCP-Gen3-v0"

    def __init__(self, config: KortexTcpGen3RobotConfig):
        super().__init__(config)
        self.config: KortexTcpGen3RobotConfig = config

    def _env_cfg_dict(self) -> dict[str, Any]:
        return {
            "robot_ip": self.config.robot_ip,
            "use_tabletop_camera": self.config.use_tabletop_camera,
            "use_wrist_camera": self.config.use_wrist_camera,
            "dt": self.config.dt,
        }
