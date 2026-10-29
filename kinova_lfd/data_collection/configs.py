"""Configuration for Gen3 LeRobot demonstration recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots import RobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig

from kinova_lfd.envs.lerobot import KortexTcpGen3RobotConfig, MjGen3RobotConfig


@dataclass
class Gen3DatasetConfig:
    """Local-dataset settings; defaults to saving under ``_data/``."""

    # Short name; stored as ``_data/<dataset_name>`` unless ``root`` is set.
    dataset_name: str = "gen3_demo"
    repo_id: str = "local/gen3_demo"
    single_task: str = "Gen3 demonstration"
    root: Path | None = None
    fps: int = 20
    episode_time_s: float = 60.0
    reset_time_s: float = 10.0
    num_episodes: int = 10
    video: bool = True
    push_to_hub: bool = False
    resume: bool = False
    streaming_encoding: bool = False
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1

    def resolved_root(self) -> Path:
        if self.root is not None:
            return Path(self.root)
        return Path("_data") / self.dataset_name

    def to_lerobot_dataset_config(self) -> DatasetRecordConfig:
        return DatasetRecordConfig(
            repo_id=self.repo_id,
            single_task=self.single_task,
            root=self.resolved_root(),
            fps=self.fps,
            episode_time_s=self.episode_time_s,
            reset_time_s=self.reset_time_s,
            num_episodes=self.num_episodes,
            video=self.video,
            push_to_hub=self.push_to_hub,
            streaming_encoding=self.streaming_encoding,
            num_image_writer_processes=self.num_image_writer_processes,
            num_image_writer_threads_per_camera=self.num_image_writer_threads_per_camera,
            video_encoding_batch_size=self.video_encoding_batch_size,
            no_stamp=True,
        )


@dataclass
class Gen3RecordConfig:
    """Top-level recording config for Mj / Kortex Gen3 + Gen3/stock teleops."""

    robot: RobotConfig = field(default_factory=MjGen3RobotConfig)
    teleop: TeleoperatorConfig = field(default_factory=lambda: make_teleop_config("gen3_keyboard"))
    dataset: Gen3DatasetConfig = field(default_factory=Gen3DatasetConfig)
    # Scales teleop delta_x/y/z into TCP linear twist (m/s-ish).
    linear_scale: float = 0.15
    display_data: bool = False
    play_sounds: bool = False
    resume: bool = False
    # MuJoCo passive viewer (only used when robot is mj_gen3).
    view_mujoco: bool = True
    view_wrist: bool = True
    view_tabletop: bool = True


def make_robot_config(robot_type: str, **overrides) -> RobotConfig:
    """Build a Gen3 robot config from a short type name."""
    if robot_type in ("mj_gen3", "mj"):
        return MjGen3RobotConfig(**overrides)
    if robot_type in ("kortex_tcp_gen3", "kortex", "kortex_tcp"):
        return KortexTcpGen3RobotConfig(**overrides)
    raise ValueError(f"Unknown robot type {robot_type!r}. Use 'mj_gen3' or 'kortex_tcp_gen3'.")


def make_teleop_config(teleop_type: str, *, use_gripper: bool = True, **overrides) -> TeleoperatorConfig:
    """Build a teleop config from a short type name."""
    if teleop_type in ("gen3_keyboard",):
        from kinova_lfd.data_collection.teleop import Gen3KeyboardTeleopConfig

        # Gen3 keyboard owns gripper via P; ignore use_gripper / LeRobot-only kwargs.
        overrides.pop("use_gripper", None)
        return Gen3KeyboardTeleopConfig(**overrides)
    if teleop_type in ("keyboard_ee", "keyboard"):
        return KeyboardEndEffectorTeleopConfig(use_gripper=use_gripper, **overrides)
    if teleop_type in ("gamepad", "xbox"):
        return GamepadTeleopConfig(use_gripper=use_gripper, **overrides)
    raise ValueError(f"Unknown teleop type {teleop_type!r}. Use 'gen3_keyboard', 'keyboard_ee', or 'gamepad'.")


def teleop_emits_twist_tcp(teleop: TeleoperatorConfig) -> bool:
    """True if the teleop already produces Gen3 TWIST_TCP action keys."""
    return getattr(teleop, "type", None) == "gen3_keyboard"
