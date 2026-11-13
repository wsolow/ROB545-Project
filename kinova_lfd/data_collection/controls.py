"""Episode controls for Gen3 recording.

Every way an operator can end, discard, reset, or stop an episode funnels into one
shared ``events`` flag dict through :func:`apply_recording_control`:

- keyboard keys (:func:`init_recording_listener`): ``]`` / Enter, ``[``, R, Esc. These do
  not overlap ``Se3Keyboard`` motion keys and stay active with every action source;
- device buttons (e.g. Xbox X / B / A / Back via ``LeRobotDeviceTeleop.get_recording_controls``);
- standard LeRobot ``TeleopEvents`` from any teleop.

The last two are polled each control tick by :func:`poll_source_controls`.
"""

from typing import Any

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.keyboard_input import pynput_can_capture

PYNPUT_AVAILABLE = _pynput_available
try:
    from pynput import keyboard as pynput_keyboard
except Exception:  # pragma: no cover
    PYNPUT_AVAILABLE = False
    pynput_keyboard = None  # type: ignore[assignment]


def apply_recording_control(control: str, events: dict[str, bool]) -> None:
    """Set the ``events`` flags for one recording control.

    ``control`` is one of ``"end"`` (``]`` / Enter / Xbox X), ``"rerecord"`` (``[`` / Xbox B),
    ``"reset"`` (R / Xbox A), or ``"stop"`` (Esc / Xbox Back).
    """
    if control == "end":
        print("[record] End: ending episode and saving...")
        events["exit_early"] = True
    elif control == "rerecord":
        print("[record] Re-record: discarding episode and re-recording...")
        events["rerecord_episode"] = True
        events["exit_early"] = True
    elif control == "stop":
        print("[record] Stop: stopping data recording...")
        events["stop_recording"] = True
        events["reset_environment"] = True
        events["exit_early"] = True
    elif control == "reset":
        print("[record] Reset: resetting environment and discarding episode...")
        events["reset_environment"] = True
        events["exit_early"] = True
    else:
        raise ValueError(f"Unknown recording control {control!r}.")


def init_recording_listener() -> tuple[Any | None, dict[str, bool]]:
    """Create the shared ``events`` dict and start the keyboard listener feeding it.

    Returns ``(listener, events)``. ``listener`` is ``None`` when pynput cannot capture
    keys (episodes then end on their timers or on device buttons).
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "reset_environment": False,
    }

    if not (PYNPUT_AVAILABLE and pynput_can_capture() and pynput_keyboard is not None):
        return None, events

    def _on_press(key) -> None:
        try:
            ch = key.char
        except AttributeError:
            if key == pynput_keyboard.Key.esc:
                apply_recording_control("stop", events)
            elif key == pynput_keyboard.Key.enter:
                apply_recording_control("end", events)
            return

        if ch == "]":
            apply_recording_control("end", events)
        elif ch == "[":
            apply_recording_control("rerecord", events)
        elif ch == "r":
            apply_recording_control("reset", events)

    listener = pynput_keyboard.Listener(on_press=_on_press)
    listener.start()
    return listener, events


def poll_source_controls(source: Any, events: dict[str, bool]) -> None:
    """Merge an action source's device controls and LeRobot teleop events into ``events``."""
    get_recording_controls = getattr(source, "get_recording_controls", None)
    if callable(get_recording_controls):
        for control in get_recording_controls():
            apply_recording_control(control, events)

    get_teleop_events = getattr(source, "get_teleop_events", None)
    if callable(get_teleop_events):
        _apply_teleop_events(get_teleop_events(), events)


def _apply_teleop_events(teleop_events: dict[TeleopEvents, bool], events: dict[str, bool]) -> None:
    """Map standard LeRobot teleop episode events onto the recording controls.

    Success ends and saves the episode; re-record discards and re-records; a failure or
    a bare terminate discards the episode and resets the environment (never saves).
    """
    if teleop_events.get(TeleopEvents.RERECORD_EPISODE):
        apply_recording_control("rerecord", events)
    elif teleop_events.get(TeleopEvents.SUCCESS):
        apply_recording_control("end", events)
    elif teleop_events.get(TeleopEvents.FAILURE) or teleop_events.get(TeleopEvents.TERMINATE_EPISODE):
        apply_recording_control("reset", events)
