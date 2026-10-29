"""Configuration for the Kortex TCP Gen3 LeRobot wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("kortex_tcp_gen3")
@dataclass(kw_only=True)
class KortexTcpGen3RobotConfig(RobotConfig):
    """Config for wrapping ``Kortex-TCP-Gen3-v0`` as a LeRobot ``Robot``."""

    robot_ip: str = "192.168.1.10"
    dt: float = 1.0 / 20.0

    # Observation modalities
    use_wrist_camera: bool = True
    use_tabletop_camera: bool = False
    include_joint_positions: bool = True
    include_ee_pose: bool = True
    camera_height: int = 480
    camera_width: int = 640

    # Action mode forwarded to Kinova ``ActionSpec``
    action_spec: str = "TWIST_TCP"
