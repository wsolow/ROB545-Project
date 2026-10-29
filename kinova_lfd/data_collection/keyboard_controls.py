"""Non-overlapping episode keyboard controls for Gen3 recording."""

from __future__ import annotations

import logging
from typing import Any

from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
try:
    from pynput import keyboard as pynput_keyboard
except Exception as e:  # pragma: no cover
    PYNPUT_AVAILABLE = False
    pynput_keyboard = None  # type: ignore[assignment]
    logger.info("Could not import pynput for recording controls: %s", e)


def apply_gen3_recording_control(control: str, events: dict[str, bool]) -> None:
    """Map recording control keys onto the shared events dict.

    ``control`` is one of ``"end"`` (``]``), ``"rerecord"`` (``[``), or ``"esc"``.
    """
    if control == "end":
        print("'] pressed. Ending episode and saving...")
        events["exit_early"] = True
    elif control == "rerecord":
        print("[ pressed. Discarding episode and re-recording...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif control == "esc":
        print("Escape pressed. Stopping data recording...")
        events["stop_recording"] = True
        events["exit_early"] = True


def init_gen3_recording_listener() -> tuple[Any | None, dict[str, bool]]:
    """Start a pynput listener for ``[`` / ``]`` / Esc episode controls.

    Returns ``(listener, events)``. ``listener`` may be ``None`` if capture is
    unavailable; ``events`` is always a mutable flag dict.
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if not (PYNPUT_AVAILABLE and pynput_can_capture() and pynput_keyboard is not None):
        logger.warning(
            "Gen3 recording keyboard controls unavailable (no pynput capture). "
            "Use episode_time timeout or gamepad teleop events instead."
        )
        return None, events

    def _on_press(key) -> None:
        try:
            ch = key.char
        except AttributeError:
            ch = None
            if key == pynput_keyboard.Key.esc:
                apply_gen3_recording_control("esc", events)
            elif key == pynput_keyboard.Key.enter:
                apply_gen3_recording_control("end", events)
            return

        if ch == "]":
            apply_gen3_recording_control("end", events)
        elif ch == "[":
            apply_gen3_recording_control("rerecord", events)

    listener = pynput_keyboard.Listener(on_press=_on_press)
    listener.start()
    return listener, events
