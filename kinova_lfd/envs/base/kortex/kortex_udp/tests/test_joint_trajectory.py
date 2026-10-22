"""Send one joint-position trajectory to the physical Gen3.

The bridge keeps the last command, and the C++ controller plays that goal at
1 kHz. This script sends the goal once, then only prints feedback.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

_ROBOT_IP = "192.168.1.10"
_GRIPPER = 0.0
_DURATION_S = 5.0
# Added to the measured actuator angle, in radians. Zero holds the current pose.
_DELTA_RAD = np.array([0.0, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1])
_WATCH_S = _DURATION_S + 0.6
_PRINT_DT = 0.1


def _joints_rad(feedback: object) -> np.ndarray:
    """Joint positions from cyclic feedback, in radians, in the actuator frame."""
    return np.deg2rad(np.array([actuator.position for actuator in feedback.actuators], dtype=np.float64))  # type: ignore[attr-defined]


def _fmt(values: np.ndarray, digits: int = 2) -> str:
    """Format a vector with a fixed number of digits."""
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def main() -> None:
    """Move the arm from its measured pose with a single joint-position command."""
    bridge = KortexUDPBridge(_ROBOT_IP, dry_run=False)
    error_deg = float("inf")
    try:
        bridge.start()
        start = _joints_rad(bridge.latest_feedback())
        goal = start + _DELTA_RAD
        print(f"start deg {_fmt(np.rad2deg(start))}")
        print(f"goal deg  {_fmt(np.rad2deg(goal))}  duration {_DURATION_S:.1f} s")
        bridge.kortex.send_joint_position(goal, _GRIPPER, _DURATION_S)

        opened = time.monotonic()
        next_print = opened
        while True:
            elapsed = time.monotonic() - opened
            if elapsed >= _WATCH_S:
                break
            now = time.monotonic()
            if now >= next_print:
                joints = _joints_rad(bridge.latest_feedback())
                error_deg = np.rad2deg(np.max(np.abs(joints - goal)))
                print(f"t={elapsed:6.3f}s  joints deg {_fmt(np.rad2deg(joints))}  goal error {error_deg:.2f} deg")
                next_print += _PRINT_DT
            else:
                time.sleep(0.001)

        final = _joints_rad(bridge.latest_feedback())
        error_deg = float(np.rad2deg(np.max(np.abs(final - goal))))
        print(f"final deg {_fmt(np.rad2deg(final))}  goal error {error_deg:.2f} deg")
    finally:
        bridge.close()

    if error_deg > 1.0:
        print("\nFAIL")
        raise SystemExit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
