"""Configuration for the MuJoCo Gen3 LeRobot wrapper."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mj_gen3")
@dataclass(kw_only=True)
class MjGen3RobotConfig(RobotConfig):
    """Config for wrapping ``Mj-Gen3-v0`` as a LeRobot ``Robot``."""

    device: str = "cuda"
    task_name: str = "Mj-Gen3-v0"
    enable_cameras: bool = True

    # Observation modalities (Mj scene always has both cameras; flags gate obs)
    use_wrist_camera: bool = True
    use_tabletop_camera: bool = True
    include_joint_positions: bool = True
    include_ee_pose: bool = True
    camera_height: int = 480
    camera_width: int = 640

    # Action mode forwarded to Kinova ``ActionSpec``
    action_spec: str = "TWIST_TCP"

    # Wall-clock control period (s), matching ``KortexTcpGen3RobotConfig.dt``. On connect,
    # send_action repeats env.step round(dt / env.step_dt) times so sim time tracks wall
    # time. None -> one env.step per action (e.g. policies trained at env.step_dt).
    dt: float | None = None
