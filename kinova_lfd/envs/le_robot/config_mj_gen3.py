"""Configuration for the MuJoCo Gen3 LeRobot wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mj_gen3")
@dataclass(kw_only=True)
class MjGen3RobotConfig(RobotConfig):
    """Config for wrapping ``Mj-Gen3-v0`` as a LeRobot ``Robot``."""

    device: str = "cuda"

    # Observation modalities (Mj scene always has both cameras; flags gate obs)
    use_wrist_camera: bool = True
    use_tabletop_camera: bool = True
    include_joint_positions: bool = True
    include_ee_pose: bool = True
    camera_height: int = 480
    camera_width: int = 640

    # Action mode forwarded to Kinova ``ActionSpec``
    action_spec: str = "TWIST_TCP"

    # MuJoCo env.dt is ~4 ms; recording often runs at 20 Hz wall clock. Repeat
    # env.step this many times per send_action so sim time tracks wall time.
    # record_gen3 sets max(1, round((1/fps) / 0.004)). Direct teleop leaves 1.
    physics_steps_per_action: int = 1
