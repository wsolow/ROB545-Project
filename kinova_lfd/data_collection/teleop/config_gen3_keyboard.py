"""Configuration for Gen3 SE(3) keyboard teleoperator."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gen3_keyboard")
@dataclass
class Gen3KeyboardTeleopConfig(TeleoperatorConfig):
    """Keyboard teleop that emits Gen3 ``TWIST_TCP`` actions (6-DoF + gripper)."""

    pos_sensitivity: float = 0.15
    rot_sensitivity: float = 0.2
