"""LeRobot Robot wrapping KinovaLfD ``Mj-Gen3-v0``."""

from dataclasses import dataclass
from typing import Any

from lerobot.robots.config import RobotConfig

from kinova_lfd.envs.factory import get_env_cfg

from .base import Gen3LfDRobot

"""Configuration for the MuJoCo Gen3 LeRobot wrapper."""


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

    action_spec: str = "TWIST_TCP"

    dt: float | None = None

    episode_length_s: float | None = None

    interpolate_joint_targets: bool = False


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

            cfg["decimation"] = decimation
        if self.config.episode_length_s is not None:
            cfg["episode_length_s"] = self.config.episode_length_s
        return cfg
