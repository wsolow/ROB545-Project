"""Send one small tool twist to the physical Gen3, then pause.

The twist servo drops a command older than incoming_command_timeout (0.5 s),
so the same setpoint is repeated at 30 Hz for one second. Sending nothing
after that lets the timeout stop the arm before the bridge exits.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

_ROBOT_IP = "192.168.1.10"
_GRIPPER = 0.0
_DT = 1.0 / 30.0
_MOVE_S = 1.5
_PAUSE_S = 0.5
# Tool-frame linear m/s, then angular deg/s. 0.01 m/s for 1 s is about 10 mm.
_TWIST = np.array([0.00, -0.05, 0.00, 0.00, 0.00, 0.0])


def _tool_xyz(feedback: object) -> np.ndarray:
    """Tool position from cyclic feedback, in meters."""
    base = feedback.base  # type: ignore[attr-defined]
    return np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z], dtype=np.float64)


def _fmt(values: np.ndarray, digits: int = 4) -> str:
    """Format a vector with a fixed number of digits."""
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def _stream(bridge: KortexUDPBridge, duration_s: float, twist: np.ndarray | None) -> None:
    """Repeat one twist at 30 Hz, or send nothing when twist is None."""
    start_xyz = _tool_xyz(bridge.latest_feedback())
    opened = time.monotonic()
    next_tick = opened
    while True:
        elapsed = time.monotonic() - opened
        if elapsed >= duration_s:
            break
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        next_tick += _DT
        if twist is not None:
            bridge.kortex.send_twist(twist, _GRIPPER)
        xyz = _tool_xyz(bridge.latest_feedback())
        print(f"t={elapsed:6.3f}s  tool xyz m {_fmt(xyz)}  delta mm {_fmt((xyz - start_xyz) * 1000.0, 2)}")


def main() -> None:
    """Move a few millimeters along tool +X, then let the command time out."""
    bridge = KortexUDPBridge(_ROBOT_IP, dry_run=False)
    try:
        bridge.start()
        print(f"tool +X at {_TWIST[0]:.3f} m/s for {_MOVE_S:.1f} s")
        _stream(bridge, _MOVE_S, _TWIST)
        print(f"pause {_PAUSE_S:.1f} s")
        _stream(bridge, _PAUSE_S, None)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
