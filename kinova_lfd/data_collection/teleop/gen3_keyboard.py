"""Gen3 keyboard teleoperator emitting TWIST_TCP actions (WASD laptop layout)."""

from __future__ import annotations

import logging
from queue import Queue
from typing import Any

import numpy as np
from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

from .config_gen3_keyboard import Gen3KeyboardTeleopConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
try:
    from pynput import keyboard as pynput_keyboard
except Exception as e:  # pragma: no cover - display / headless
    PYNPUT_AVAILABLE = False
    pynput_keyboard = None  # type: ignore[assignment]
    logger.info("Could not import pynput for Gen3KeyboardTeleop: %s", e)

TWIST_KEYS = ("tcp.vx", "tcp.vy", "tcp.vz", "tcp.wx", "tcp.wy", "tcp.wz", "gripper.pos")


class Gen3KeyboardTeleop(Teleoperator):
    """SE(3) keyboard teleop for Gen3 using a laptop WASD layout.

    Translation: W/S X, A/D Y, Q/E Z. Rotation: I/K roll, J/L pitch, U/O yaw.
    Gripper: P. Episode controls are owned by ``init_gen3_recording_listener``
    (``[`` / ``]`` / Esc). Assumes the MuJoCo Simulate window is not focused.
    """

    config_class = Gen3KeyboardTeleopConfig
    name = "gen3_keyboard"

    def __init__(self, config: Gen3KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self._event_queue: Queue[tuple[str | None, bool]] = Queue()
        self._pressed: set[str] = set()
        self._listener = None
        self._connected = False
        self._gripper_closed = False
        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_rot = np.zeros(3, dtype=np.float64)
        self._build_key_map()

    def _build_key_map(self) -> None:
        ps = self.config.pos_sensitivity
        rs = self.config.rot_sensitivity
        self._pos_keys = {
            "w": np.asarray([1.0, 0.0, 0.0]) * ps,
            "s": np.asarray([-1.0, 0.0, 0.0]) * ps,
            "a": np.asarray([0.0, 1.0, 0.0]) * ps,
            "d": np.asarray([0.0, -1.0, 0.0]) * ps,
            "q": np.asarray([0.0, 0.0, 1.0]) * ps,
            "e": np.asarray([0.0, 0.0, -1.0]) * ps,
        }
        self._rot_keys = {
            "i": np.asarray([1.0, 0.0, 0.0]) * rs,
            "k": np.asarray([-1.0, 0.0, 0.0]) * rs,
            "j": np.asarray([0.0, 1.0, 0.0]) * rs,
            "l": np.asarray([0.0, -1.0, 0.0]) * rs,
            "u": np.asarray([0.0, 0.0, 1.0]) * rs,
            "o": np.asarray([0.0, 0.0, -1.0]) * rs,
        }
        self._gripper_key = "p"

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in TWIST_KEYS}

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

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if not (PYNPUT_AVAILABLE and pynput_can_capture() and pynput_keyboard is not None):
            logger.warning(
                "Gen3KeyboardTeleop: pynput cannot capture keys in this environment; " "actions will be zeros."
            )
            self._listener = None
            self._connected = True
            return

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._connected = True
        print(self)

    def __str__(self) -> str:
        msg = "Gen3 Keyboard Teleop (TWIST_TCP)\n"
        msg += "\tMove   X: W/S   Y: A/D   Z: Q/E\n"
        msg += "\tRotate roll: I/K   pitch: J/L   yaw: U/O\n"
        msg += "\tToggle gripper: P\n"
        return msg

    def _key_char(self, key) -> str | None:
        try:
            ch = key.char
        except AttributeError:
            return None
        if ch is None:
            return None
        return ch.lower()

    def _on_press(self, key) -> None:
        ch = self._key_char(key)
        if ch is None:
            return
        self._event_queue.put((ch, True))

    def _on_release(self, key) -> None:
        ch = self._key_char(key)
        if ch is None:
            return
        self._event_queue.put((ch, False))

    def _drain(self) -> None:
        while not self._event_queue.empty():
            ch, pressed = self._event_queue.get_nowait()
            if ch == self._gripper_key:
                if pressed and ch not in self._pressed:
                    self._gripper_closed = not self._gripper_closed
                    self._pressed.add(ch)
                elif not pressed:
                    self._pressed.discard(ch)
                continue

            if ch in self._pos_keys:
                if pressed and ch not in self._pressed:
                    self._delta_pos += self._pos_keys[ch]
                    self._pressed.add(ch)
                elif not pressed and ch in self._pressed:
                    self._delta_pos -= self._pos_keys[ch]
                    self._pressed.discard(ch)
            elif ch in self._rot_keys:
                if pressed and ch not in self._pressed:
                    self._delta_rot += self._rot_keys[ch]
                    self._pressed.add(ch)
                elif not pressed and ch in self._pressed:
                    self._delta_rot -= self._rot_keys[ch]
                    self._pressed.discard(ch)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain()
        gripper = 1.0 if self._gripper_closed else 0.0
        return {
            "tcp.vx": float(self._delta_pos[0]),
            "tcp.vy": float(self._delta_pos[1]),
            "tcp.vz": float(self._delta_pos[2]),
            "tcp.wx": float(self._delta_rot[0]),
            "tcp.wy": float(self._delta_rot[1]),
            "tcp.wz": float(self._delta_rot[2]),
            "gripper.pos": gripper,
        }

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        # Episode control is owned by init_gen3_recording_listener ([ / ] / Esc).
        return {
            TeleopEvents.IS_INTERVENTION: bool(self._pressed),
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._connected = False
        self._pressed.clear()
        self._delta_pos[:] = 0.0
        self._delta_rot[:] = 0.0
