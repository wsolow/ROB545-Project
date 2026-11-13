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

    # Control period (s), matching ``KortexTcpGen3RobotConfig.dt``. Sets the env decimation to
    # round(dt / physics_dt) so one env.step (one IK solve, forward, and render) spans one control
    # tick. None -> the task's own decimation (e.g. policies trained at its env.step_dt).
    dt: float | None = None

    # Override the task's episode_length_s (env auto-resets once this much sim time passes
    # since the last reset). None -> the task default.
    episode_length_s: float | None = None

    # Linearly interpolate arm joint targets across the physics substeps of each env.step
    # instead of holding them constant (Mj-Gen3-v0 only).
    interpolate_joint_targets: bool = False
