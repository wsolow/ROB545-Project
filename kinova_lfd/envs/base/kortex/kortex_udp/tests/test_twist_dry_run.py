"""Dry-run the Gen3 twist servo against the 1 kHz bridge.

The C++ process integrates joint commands with Pinocchio. This script sends a
few tool-frame twists at 30 Hz, prints each command next to the resulting tool
motion, and checks that a dropped command stream stops the arm.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

if TYPE_CHECKING:
    from collections.abc import Callable

_DT = 1.0 / 30.0
_GRIPPER = 0.0


def _rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build R = Rz(yaw) Ry(pitch) Rx(roll), matching the bridge RPY."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _tool_pose(feedback: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tool position (m), rotation, and joint positions (deg)."""
    base = feedback.base  # type: ignore[attr-defined]
    position = np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z], dtype=np.float64)
    rotation = _rotation_xyz(
        np.deg2rad(base.tool_pose_theta_x),
        np.deg2rad(base.tool_pose_theta_y),
        np.deg2rad(base.tool_pose_theta_z),
    )
    joints = np.array([actuator.position for actuator in feedback.actuators], dtype=np.float64)  # type: ignore[attr-defined]
    return position, rotation, joints


def _fmt(values: np.ndarray, digits: int = 3) -> str:
    """Format a vector with a fixed number of digits."""
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def _rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw in degrees for R = Rz Ry Rx."""
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.rad2deg(np.array([roll, pitch, yaw]))


def _run_segment(
    bridge: KortexUDPBridge,
    name: str,
    expected: str,
    duration_s: float,
    send: Callable[[], None] | None,
    command: np.ndarray,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Send one setpoint stream and print the motion at 30 Hz."""
    origin_p, origin_r, _origin_q = _tool_pose(bridge.latest_feedback())
    print(f"\n== {name} ==")
    print(f"expected: {expected}")
    samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    start = time.monotonic()
    next_tick = start
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        next_tick += _DT
        if send is not None:
            send()
        position, rotation, joints = _tool_pose(bridge.latest_feedback())
        tool_delta_mm = origin_r.T @ (position - origin_p) * 1000.0
        relative = origin_r.T @ rotation
        yaw_deg = float(np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0])))
        stamp = time.monotonic() - start
        samples.append((stamp, tool_delta_mm, joints, np.array([yaw_deg])))
        line = "  ".join(
            [
                f"t={stamp:6.3f}s cmd m/s,deg/s {_fmt(command, 3)}",
                f"tool xyz m {_fmt(position, 4)}",
                f"rpy deg {_fmt(_rpy_deg(rotation), 2)}",
                f"tool delta mm {_fmt(tool_delta_mm, 2)}",
                f"yaw deg {yaw_deg:6.2f}",
                f"joints deg {_fmt(joints, 2)}",
            ]
        )
        print(line)
    return samples


def main() -> None:
    """Exercise translation, timeout, and tool yaw on the dry-run bridge."""
    bridge = KortexUDPBridge("192.168.1.10", dry_run=True)
    failures: list[str] = []
    try:
        bridge.start()

        def send_twist(twist: np.ndarray) -> None:
            """Forward one tool twist and a closed gripper."""
            bridge.kortex.send_twist(twist, _GRIPPER)

        plus_x = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_samples = _run_segment(
            bridge,
            "tool +X",
            "tool frame moves along +X by about 15-20 mm",
            0.40,
            lambda: send_twist(plus_x),
            plus_x,
        )
        x_delta = x_samples[-1][1]
        if x_delta[0] < 8.0 or abs(x_delta[0]) <= max(abs(x_delta[1]), abs(x_delta[2])):
            failures.append(f"tool +X motion was {x_delta} mm, expected dominant +X")

        # The same twist values, resent the way a held keyboard key is. The host
        # stamp must keep this alive past incoming_command_timeout.
        held_samples = _run_segment(
            bridge,
            "held twist",
            "repeating the same twist for longer than the timeout keeps the tool moving",
            1.20,
            lambda: send_twist(plus_x),
            plus_x,
        )
        held_early = [sample for sample in held_samples if sample[0] >= 0.60]
        held_late = [sample for sample in held_samples if sample[0] >= 1.05]
        if held_early and held_late:
            held_mm = float(held_late[0][1][0] - held_early[0][1][0])
            if held_mm < 10.0:
                failures.append(f"held twist moved {held_mm:.2f} mm after 0.6 s, expected it to stay live")
        else:
            failures.append("held twist segment did not collect the post-timeout window")

        pause = np.zeros(6)
        pause_samples = _run_segment(
            bridge,
            "command timeout",
            "arm keeps the last twist for about 0.5 s, then coasts to a stop",
            1.15,
            None,
            pause,
        )
        # A 20 Hz teleop step is 50 ms. Motion through this window means a late
        # poll no longer expires the twist.
        moving = [sample for sample in pause_samples if sample[0] >= 0.12]
        still_moving = [sample for sample in pause_samples if sample[0] >= 0.40]
        if moving and still_moving:
            held_mm = float(still_moving[0][1][0] - moving[0][1][0])
            if held_mm < 8.0:
                failures.append(f"tool moved {held_mm:.2f} mm before the timeout, expected the twist to stay live")
        else:
            failures.append("timeout segment did not collect the live-command window")
        early = [sample for sample in pause_samples if sample[0] >= 0.52]
        later = [sample for sample in pause_samples if sample[0] >= 0.70]
        if early and later:
            coast_mm = float(later[0][1][0] - early[0][1][0])
            if coast_mm < 1.0:
                failures.append(f"tool coasted {coast_mm:.2f} mm after the timeout, expected a ramp down")
        else:
            failures.append("timeout segment did not collect the coast window")
        held = [sample for sample in pause_samples if sample[0] >= 0.95]
        if len(held) >= 2:
            drift = np.max(np.abs(held[-1][2] - held[0][2]))
            if drift > 0.05:
                failures.append(f"joints drifted {drift:.3f} deg after the coast")
        else:
            failures.append("timeout segment did not collect samples after 0.95 s")

        plus_z = np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
        z_samples = _run_segment(
            bridge,
            "tool +Z",
            "tool frame moves along +Z",
            0.40,
            lambda: send_twist(plus_z),
            plus_z,
        )
        z_delta = z_samples[-1][1]
        if z_delta[2] < 6.0 or abs(z_delta[2]) <= max(abs(z_delta[0]), abs(z_delta[1])):
            failures.append(f"tool +Z motion was {z_delta} mm, expected dominant +Z")

        yaw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])
        yaw_samples = _run_segment(
            bridge,
            "tool +yaw",
            "tool frame yaws about +Z by several degrees",
            0.40,
            lambda: send_twist(yaw),
            yaw,
        )
        if float(yaw_samples[-1][3][0]) < 3.0:
            failures.append(f"tool yaw was {float(yaw_samples[-1][3][0]):.2f} deg, expected > 3")

        held_joints = np.deg2rad(_tool_pose(bridge.latest_feedback())[2])

        def send_joints() -> None:
            """Record the current joint pose. The trajectory controller holds it."""
            bridge.kortex.send_joint_position(held_joints, _GRIPPER)

        joint_samples = _run_segment(
            bridge,
            "joint position hold",
            "joint controller records the goal and holds the measured pose",
            0.25,
            send_joints,
            np.zeros(6),
        )
        joint_drift = np.max(np.abs(joint_samples[-1][2] - joint_samples[0][2]))
        if joint_drift > 0.05:
            failures.append(f"joint-position mode moved the arm by {joint_drift:.3f} deg")
        if bridge.latest_feedback().command_mode != 1:
            failures.append("joint-position command was not selected")
    finally:
        bridge.close()

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
