"""Factories for Gen3 robots, teleoperators, and recording viewers."""

from typing import Any

from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config

from .lerobot_device import LeRobotDeviceTeleop, LeRobotDeviceTeleopConfig


def make_robot(config: RobotConfig) -> Robot:
    """Construct a Gen3 (or other) robot via LeRobot's factory.

    The Gen3 robot configs register with LeRobot when this package imports
    ``kinova_lfd.envs.le_robot`` (via ``configs``), so no explicit registration is needed.
    """
    return make_robot_from_config(config)


def make_teleop(config: TeleoperatorConfig) -> Teleoperator:
    """Construct a repo-device teleop, or fall back to LeRobot's factory for other configs."""
    if isinstance(config, LeRobotDeviceTeleopConfig):
        return LeRobotDeviceTeleop(config)
    return make_teleoperator_from_config(config)


def make_viewer(robot: Robot, *, view_wrist: bool, view_tabletop: bool) -> Any | None:
    """Build the live viewer for a connected Gen3 robot, or ``None`` for unsupported robots.

    ``mj_gen3`` gets an ``MjViewer`` (MuJoCo window + camera preview) and ``kortex_tcp_gen3``
    a ``KortexViewer`` (camera preview). Both run in push mode: the record loop hands them
    the frames already in the robot observation via ``sync_display(wrist=..., tabletop=...)``,
    so the viewer never re-renders (MuJoCo) or competes for the RealSense pipeline (Kortex).
    Call ``open_display()`` / ``sync_display(...)`` / ``close_display()`` on the result.
    """
    from kinova_lfd.envs.le_robot import KortexTcpGen3Robot, MjGen3Robot

    env = getattr(robot, "env", None)
    if env is None:
        raise RuntimeError("Connect the robot before creating its viewer.")
    if isinstance(robot, MjGen3Robot):
        from kinova_lfd.envs.base.viewers.mujoco_viewer import MjViewer

        return MjViewer(env, policy=None, view_wrist=view_wrist, view_tabletop=view_tabletop, ignore_mjlab_keys=True)
    if isinstance(robot, KortexTcpGen3Robot):
        from kinova_lfd.envs.base.viewers.kortex_viewer import KortexViewer

        return KortexViewer(env, policy=None, view_wrist=view_wrist, view_tabletop=view_tabletop, pull_frames=False)
    return None
