"""Keyboard controller for SE(3) control (standalone, no Omni/Carb dependency)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pynput import keyboard as pynput_keyboard

from ..device_base import DeviceBase, DeviceCfg


class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    Uses pynput instead of Omniverse's carb/omni keyboard interface, so it can run
    outside of an Isaac Sim / Omniverse context.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    R
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================
    """

    def __init__(self, cfg: Se3KeyboardCfg):
        super().__init__(cfg)

        self._additional_callbacks: dict[str, Callable] = {}

        self._create_key_bindings()

        self._pressed_keys: set[str] = set()

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    def start(self):
        self._listener.start()

    def stop(self):
        """Stop the keyboard listener."""
        if hasattr(self, "_listener") and self._listener.is_alive():
            self._listener.stop()

    def __str__(self) -> str:
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): R\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    def reset(self):
        super().reset()
        self._close_gripper = False

    def _key_to_char(self, key) -> str | None:
        """Convert a pynput key object to an uppercase character string, or None."""
        try:
            return key.char.upper()
        except AttributeError:
            return None

    def _on_press(self, key):
        char = self._key_to_char(key)
        if char is None:
            return

        if char == "F":
            self.reset()
            return

        if char == "R":
            self._close_gripper = not self._close_gripper
            self._gripper_value = 1.0 if self._close_gripper else 0.0
            return

        # Guard against key-repeat events firing duplicate additions
        if char in self._pressed_keys:
            return
        self._pressed_keys.add(char)

        if char in self._POS_KEYS:
            self._delta_pos += self._INPUT_KEY_MAPPING[char]
        elif char in self._ROT_KEYS:
            self._delta_rot += self._INPUT_KEY_MAPPING[char]

        # User-registered callbacks
        if char in self._additional_callbacks:
            self._additional_callbacks[char]()

    def _on_release(self, key):
        char = self._key_to_char(key)
        if char is None:
            return

        self._pressed_keys.discard(char)

        if char in self._POS_KEYS:
            self._delta_pos -= self._INPUT_KEY_MAPPING[char]
        elif char in self._ROT_KEYS:
            self._delta_rot -= self._INPUT_KEY_MAPPING[char]

    def _create_key_bindings(self):
        self._POS_KEYS = {"W", "S", "A", "D", "Q", "E"}
        self._ROT_KEYS = {"Z", "X", "T", "G", "C", "V"}

        self._INPUT_KEY_MAPPING = {
            # x-axis (forward/back)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left/right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up/down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }


@dataclass
class Se3KeyboardCfg(DeviceCfg):
    """Configuration for SE3 keyboard devices."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.4
    rot_sensitivity: float = 0.8
    reference_frame: str = "base"
