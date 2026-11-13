"""LeRobot ``Teleoperator`` adapter for KinovaLfD ``DeviceBase`` teleop devices.

Any ``DeviceBase`` (``Se3Keyboard``, ``XboxController``, ...) can drive LeRobot recording
through this class. Devices produce a 6-DoF twist plus gripper via ``DeviceBase.advance``;
base-frame devices need the current TCP pose, which the record loop hands over each tick
through :meth:`LeRobotDeviceTeleop.update_observation` (read from the ``ee.*`` keys).

Written by Will Solow, 2026.
"""

from typing import TYPE_CHECKING

import numpy as np
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from kinova_lfd.envs.le_robot.base import EE_POSE_KEYS, TWIST_TCP_KEYS

if TYPE_CHECKING:
    from kinova_lfd.teleop.devices.device_base import DeviceBase
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

TELEOP_DEVICES = ("keyboard", "xbox")


@TeleoperatorConfig.register_subclass("lerobot_device")
@dataclass
class LeRobotDeviceTeleopConfig(TeleoperatorConfig):
    """Wrap a KinovaLfD ``DeviceBase`` as a LeRobot teleop emitting Gen3 ``TWIST_TCP`` actions.

    ``device`` selects the repo device: ``"keyboard"`` -> ``Se3Keyboard``, ``"xbox"`` ->
    ``XboxController``. Sensitivities are the linear (m/s) / angular (rad/s) rates at
    full key press or stick deflection. ``reference_frame`` is the frame the operator
    commands in (``"base"`` or ``"tcp"``); the emitted action is always a TCP twist.
    """

    device: str = "keyboard"
    pos_sensitivity: float = 0.15
    rot_sensitivity: float = 0.15
    reference_frame: str = "base"


def build_device(config: LeRobotDeviceTeleopConfig) -> "DeviceBase":
    """Construct the repo teleop device named by ``config.device``."""
    common = {
        "pos_sensitivity": config.pos_sensitivity,
        "rot_sensitivity": config.rot_sensitivity,
        "reference_frame": config.reference_frame,
        "gripper_term": True,
    }
    if config.device == "keyboard":
        from kinova_lfd.teleop.devices.keyboard import Se3Keyboard, Se3KeyboardCfg

        return Se3Keyboard(Se3KeyboardCfg(**common))
    if config.device == "xbox":
        from kinova_lfd.teleop.devices.xbox import XboxController, XboxControllerCfg

        return XboxController(XboxControllerCfg(**common))
    raise ValueError(f"Unknown teleop device {config.device!r}. Use one of {TELEOP_DEVICES}.")


class LeRobotDeviceTeleop(Teleoperator):
    """Expose a KinovaLfD ``DeviceBase`` through the LeRobot ``Teleoperator`` API.

    Emits ``TWIST_TCP`` keys (``tcp.vx..wz``, ``gripper.pos``). Devices that count button
    presses (``press_count``, e.g. ``XboxController``) also drive episode control through
    :meth:`get_recording_controls`, using :attr:`recording_buttons`.
    """

    config_class = LeRobotDeviceTeleopConfig
    name = "lerobot_device"

    # Button -> recording control (see ``controls.apply_recording_control``).
    recording_buttons: dict[str, str] = {"x": "end", "b": "rerecord", "a": "reset", "back": "stop"}
    # Recording control -> description shown in the device's startup banner.
    control_labels: dict[str, str] = {
        "end": "End episode & save",
        "rerecord": "Discard & re-record episode",
        "reset": "Reset environment (discard episode)",
        "stop": "Stop recording session",
    }

    def __init__(self, config: LeRobotDeviceTeleopConfig, device: "DeviceBase | None" = None):
        """Wrap ``device``, or build the one named by ``config.device`` on :meth:`connect`."""
        super().__init__(config)
        self.config = config
        self._device = device
        self._connected = False
        self._tcp_pose: np.ndarray | None = None
        self._seen_presses: dict[str, int] = {}

    @property
    def device(self) -> "DeviceBase | None":
        """The wrapped teleop device (``None`` until connected unless one was passed in)."""
        return self._device

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(TWIST_TCP_KEYS, float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._device is None:
            self._device = build_device(self.config)
        if hasattr(self._device, "button_bindings") and hasattr(self._device, "press_count"):
            # Buttons mean recording controls here; show those in the banner instead.
            self._device.button_bindings = {
                button: self.control_labels[control] for button, control in self.recording_buttons.items()
            }
        self._device.start()
        self._snapshot_presses()
        self._connected = True

    def update_observation(self, observation: RobotObservation) -> None:
        """Cache the TCP pose (``ee.x..ee.qw``, xyzw quaternion) used by base-frame devices."""
        if all(key in observation for key in EE_POSE_KEYS):
            self._tcp_pose = np.asarray([observation[key] for key in EE_POSE_KEYS], dtype=np.float64)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        assert self._device is not None
        if self._tcp_pose is None:
            if self._device.frame == "base":
                raise RuntimeError(
                    "Base-frame teleop needs the TCP pose: call update_observation(obs) with the robot's "
                    "ee.* keys (include_ee_pose=True) before get_action()."
                )
            pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # unused in the tcp frame
        else:
            pose = self._tcp_pose
        command = np.asarray(self._device.advance(pose), dtype=np.float64).reshape(-1)
        if command.size != len(TWIST_TCP_KEYS):
            raise ValueError(
                f"Device returned {command.size} values; expected {len(TWIST_TCP_KEYS)} "
                "(6-DoF twist + gripper, gripper_term=True)."
            )
        return {key: float(value) for key, value in zip(TWIST_TCP_KEYS, command, strict=True)}

    def get_recording_controls(self) -> list[str]:
        """Control Recording (``"end"``, ``"rerecord"``, ``"reset"``, ``"stop"``) pressed since last call."""
        press_count = getattr(self._device, "press_count", None)
        if press_count is None:
            return []
        controls = []
        for button, control in self.recording_buttons.items():
            count = press_count(button)
            if count > self._seen_presses.get(button, 0):
                controls.append(control)
            self._seen_presses[button] = count
        return controls

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Get the current teleop events (intervention, success, failure, terminate episode)."""
        # Episode control goes through get_recording_controls / the keyboard listener.
        device = self._device
        moving = device is not None and bool(np.any(device._delta_pos) or np.any(device._delta_rot))
        return {
            TeleopEvents.IS_INTERVENTION: moving,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def reset(self) -> None:
        """Clear motion / gripper state and drop button presses not yet consumed."""
        if self._device is not None:
            self._device.reset()
        self._snapshot_presses()

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._device is not None:
            self._device.stop()
        self._connected = False
        self._tcp_pose = None

    def _snapshot_presses(self) -> None:
        press_count = getattr(self._device, "press_count", None)
        if press_count is not None:
            self._seen_presses = {button: press_count(button) for button in self.recording_buttons}
