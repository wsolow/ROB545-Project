"""LeRobot Robot wrapping KinovaLfD ``Kortex-TCP-Gen3-v0``."""

from typing import Any

import numpy as np
from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from kinova_lfd.core.np_math import base_to_tcp_twist
from kinova_lfd.core.specs import ActionSpec

from .base import TWIST_TCP_KEYS, Gen3LfDRobot, action_dict_to_array
from .config_kortex_tcp_gen3 import KortexTcpGen3RobotConfig


class KortexTcpGen3Robot(Gen3LfDRobot):
    """Real Gen3 arm via Kortex TCP, exposed through the LeRobot ``Robot`` API."""

    config_class = KortexTcpGen3RobotConfig
    name = "kortex_tcp_gen3"
    task_name = "Kortex-TCP-Gen3-v0"

    def __init__(self, config: KortexTcpGen3RobotConfig):
        super().__init__(config)
        self.config: KortexTcpGen3RobotConfig = config
        self.admittance = False

    def _env_cfg_dict(self) -> dict[str, Any]:
        return {
            "robot_ip": self.config.robot_ip,
            "use_tabletop_camera": self.config.use_tabletop_camera,
            "use_wrist_camera": self.config.use_wrist_camera,
            "dt": self.config.dt,
        }

    def set_admittance(self, enabled: bool) -> None:
        """Enable/disable Cartesian admittance; applied on the next ``send_action``."""
        if self._action_spec != ActionSpec.TWIST_TCP:
            raise RuntimeError("Admittance recording requires action_spec='TWIST_TCP'.")
        self.admittance = enabled
        print(f"[KortexTcpGen3Robot] Cartesian admittance {'ON' if enabled else 'OFF'}.")

    @check_if_not_connected
    def home(self, timeout: float = 30.0) -> bool:
        """Leave admittance and move (blocking) to ``HOME_TCP_CART`` with the gripper open."""
        from kinova_lfd.agents.util import HOME_TCP_CART

        self.admittance = False
        print("[KortexTcpGen3Robot] Moving to home...")
        reached = self.env.move_to_tcp_pose(HOME_TCP_CART, timeout=timeout)
        print("[KortexTcpGen3Robot] Home reached." if reached else "[KortexTcpGen3Robot] Home move timed out.")
        return reached

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """In admittance, returns the measured TCP twist (tool frame) instead of the commanded one."""
        if not self.admittance:
            return super().send_action(action)
        vector = action_dict_to_array(action, self._action_spec)
        self.env.step(vector, action_spec=ActionSpec.CARTESIAN_ADMITTANCE)
        # Kortex reports the tool twist in the base frame; rotate it into the TCP frame.
        vel_b = self.env._robot_tool_vel_b
        lin, ang = base_to_tcp_twist(vel_b[:3], vel_b[3:6], self.env._robot_tool_pose_b[3:7])
        measured = np.concatenate((lin, ang, vector[-1:]))
        return {key: float(value) for key, value in zip(TWIST_TCP_KEYS, measured, strict=True)}
