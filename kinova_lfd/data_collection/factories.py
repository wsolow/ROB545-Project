"""Factories for Gen3 robots and stock LeRobot teleoperators."""

from __future__ import annotations

import logging

from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_gen3_robots() -> None:
    """Import Gen3 configs so ``@RobotConfig.register_subclass`` side-effects run.

    Safe to call multiple times. Also pulls in the LeRobot ``kinova_gen3`` shim when
    available so ``make_robot_from_config`` can resolve ``mj_gen3`` / ``kortex_tcp_gen3``.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Direct import registers ChoiceRegistry subclasses.
    from kinova_lfd.envs.le_robot import (  # noqa: F401
        KortexTcpGen3RobotConfig,
        MjGen3RobotConfig,
    )

    try:
        from lerobot.robots import kinova_gen3  # noqa: F401
    except ImportError:
        logger.debug("lerobot.robots.kinova_gen3 shim unavailable; using direct kinova_lfd imports.")

    _REGISTERED = True


def make_robot(config: RobotConfig) -> Robot:
    """Construct a Gen3 (or other) robot via LeRobot's factory after registration."""
    register_gen3_robots()
    return make_robot_from_config(config)


def make_teleop(config: TeleoperatorConfig) -> Teleoperator:
    """Construct a Gen3 or stock LeRobot teleoperator."""
    if getattr(config, "type", None) == "gen3_keyboard":
        from kinova_lfd.data_collection.teleop import Gen3KeyboardTeleop

        return Gen3KeyboardTeleop(config)
    return make_teleoperator_from_config(config)
