"""Processors bridging stock LeRobot EE teleops to Gen3 TWIST_TCP actions."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    IdentityProcessorStep,
    ProcessorStepRegistry,
    RobotActionProcessorStep,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

# GripperAction from GamepadTeleop / KeyboardEndEffectorTeleop
_GRIPPER_CLOSE = 0
_GRIPPER_STAY = 1
_GRIPPER_OPEN = 2


@ProcessorStepRegistry.register("delta_ee_to_twist_tcp")
@dataclass
class DeltaEEToTwistTcpProcessorStep(RobotActionProcessorStep):
    """Map ``delta_x/y/z`` (+ discrete gripper) to Gen3 ``TWIST_TCP`` keys.

    Stock LeRobot ``keyboard_ee`` / ``gamepad`` teleops emit translation deltas and a
    discrete gripper command ``{0=close, 1=stay, 2=open}``. Gen3 robots expect
    ``tcp.vx..wz`` and continuous ``gripper.pos`` in ``[0, 1]``. Angular rates are
    zeroed (stock EE teleops have no rotation).
    """

    linear_scale: float = 0.15
    open_pos: float = 0.0
    close_pos: float = 1.0
    _last_gripper_pos: float = field(default=0.0, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        dx = float(action.pop("delta_x", 0.0)) * self.linear_scale
        dy = float(action.pop("delta_y", 0.0)) * self.linear_scale
        dz = float(action.pop("delta_z", 0.0)) * self.linear_scale

        gripper_cmd = action.pop("gripper", _GRIPPER_STAY)
        try:
            gripper_cmd = int(gripper_cmd)
        except (TypeError, ValueError):
            gripper_cmd = _GRIPPER_STAY

        if gripper_cmd == _GRIPPER_CLOSE:
            self._last_gripper_pos = self.close_pos
        elif gripper_cmd == _GRIPPER_OPEN:
            self._last_gripper_pos = self.open_pos
        # stay: keep previous command

        # Drop any leftover teleop-only keys.
        action.clear()
        action.update(
            {
                "tcp.vx": dx,
                "tcp.vy": dy,
                "tcp.vz": dz,
                "tcp.wx": 0.0,
                "tcp.wy": 0.0,
                "tcp.wz": 0.0,
                "gripper.pos": float(self._last_gripper_pos),
            }
        )
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_feats = features[PipelineFeatureType.ACTION]
        for key in ("delta_x", "delta_y", "delta_z", "gripper"):
            action_feats.pop(key, None)

        for key in ("tcp.vx", "tcp.vy", "tcp.vz", "tcp.wx", "tcp.wy", "tcp.wz", "gripper.pos"):
            action_feats[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


def make_gen3_teleop_action_processor(
    linear_scale: float = 0.15,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Teleop pipeline that converts EE deltas into TWIST_TCP before recording/send."""
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[DeltaEEToTwistTcpProcessorStep(linear_scale=linear_scale)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def make_identity_robot_action_processor() -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
