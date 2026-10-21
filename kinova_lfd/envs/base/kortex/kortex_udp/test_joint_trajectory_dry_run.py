"""Dry-run the synchronized Gen3 joint trajectory.

Sends joint-position goals at 30 Hz. The C++ controller shares one trapezoidal
progress curve across the arm, and DryRunRobot integrates the resulting joint
commands at 1 kHz. This script prints that motion and checks the limits.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

if TYPE_CHECKING:
    from collections.abc import Sequence

_DT = 1.0 / 30.0
_GRIPPER = 0.0
# URDF velocity limits, rad/s, joints 1..7.
_MAX_VELOCITY = np.array([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
# Kinova joint_limits.yaml accelerations, rad/s^2.
_MAX_ACCELERATION = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0])


def _joints_rad(feedback: object) -> np.ndarray:
    """Joint positions from cyclic feedback, in radians."""
    return np.deg2rad(np.array([actuator.position for actuator in feedback.actuators], dtype=np.float64))  # type: ignore[attr-defined]


def _joints_vel(feedback: object) -> np.ndarray:
    """Joint velocities from cyclic feedback, in rad/s."""
    return np.deg2rad(np.array([actuator.velocity for actuator in feedback.actuators], dtype=np.float64))  # type: ignore[attr-defined]


def _fmt(values: np.ndarray, digits: int = 2) -> str:
    """Format a vector with a fixed number of digits."""
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def _progress(start: np.ndarray, goal: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Fraction of each joint's travel, or 1 when that joint was not asked to move."""
    delta = goal - start
    done = np.ones(start.size)
    moving = np.abs(delta) > 1e-4
    done[moving] = (joints[moving] - start[moving]) / delta[moving]
    return done


def _follow(
    bridge: KortexUDPBridge,
    name: str,
    expected: str,
    goal: np.ndarray,
    duration_s: float,
    window_s: float,
) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Resend one goal at 30 Hz and record the 1 kHz motion."""
    start = _joints_rad(bridge.latest_feedback())
    print(f"\n== {name} ==")
    print(f"expected: {expected}")
    print(f"goal rad {_fmt(goal, 3)}  requested duration {duration_s:.3f} s")
    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    max_velocity = np.zeros(7)
    max_acceleration = np.zeros(7)
    previous_velocity: np.ndarray | None = None
    previous_time: float | None = None
    opened = time.monotonic()
    next_send = opened
    next_print = opened
    while True:
        now = time.monotonic()
        elapsed = now - opened
        if elapsed >= window_s:
            break
        if now >= next_send:
            bridge.kortex.send_joint_position(goal, _GRIPPER, duration_s)
            next_send += _DT
        feedback = bridge.latest_feedback()
        joints = _joints_rad(feedback)
        velocity = _joints_vel(feedback)
        max_velocity = np.maximum(max_velocity, np.abs(velocity))
        if previous_velocity is None or previous_time is None:
            previous_velocity = velocity
            previous_time = elapsed
        elif elapsed - previous_time >= 0.005:
            max_acceleration = np.maximum(
                max_acceleration, np.abs(velocity - previous_velocity) / (elapsed - previous_time)
            )
            previous_velocity = velocity
            previous_time = elapsed
        samples.append((elapsed, joints.copy(), velocity.copy()))
        if now >= next_print:
            done = _progress(start, goal, joints)
            print(
                f"t={elapsed:6.3f}s  joints deg {_fmt(np.rad2deg(joints))}  "
                f"vel deg/s {_fmt(np.rad2deg(velocity))}  progress {_fmt(done, 3)}"
            )
            next_print += _DT
        else:
            time.sleep(0.001)
    return samples, max_velocity, max_acceleration


def _arrival(samples: Sequence[tuple[float, np.ndarray, np.ndarray]], start: np.ndarray, goal: np.ndarray) -> float:
    """First time every moving joint has completed 95 percent of its travel."""
    for stamp, joints, _velocity in samples:
        if np.all(_progress(start, goal, joints) >= 0.95):
            return stamp
    return float("inf")


def main() -> None:
    """Check a shared timing curve, then a duration that the limits must extend."""
    bridge = KortexUDPBridge("192.168.1.10", dry_run=True)
    failures: list[str] = []
    try:
        bridge.start()
        start = _joints_rad(bridge.latest_feedback())
        goal = start.copy()
        goal[0] += 0.35
        goal[1] += 0.90
        samples, peak_velocity, peak_acceleration = _follow(
            bridge,
            "shared timing",
            "joint 2 travels farther, so it moves faster, and both joints finish together",
            goal,
            3.0,
            3.6,
        )
        midpoint = min(samples, key=lambda sample: abs(sample[0] - 1.2))
        progress = _progress(start, goal, midpoint[1])
        if abs(float(progress[0] - progress[1])) > 0.05:
            failures.append(f"joints did not share progress at t={midpoint[0]:.3f}: {progress}")
        ratio = float(peak_velocity[1] / max(peak_velocity[0], 1e-6))
        if not 2.0 <= ratio <= 3.2:
            failures.append(f"speed ratio joint2/joint1 was {ratio:.2f}, expected about 2.6")
        arrival = _arrival(samples, start, goal)
        if not 2.4 <= arrival <= 3.3:
            failures.append(f"shared move arrived at {arrival:.3f} s, expected near 3 s")
        end_error = np.rad2deg(np.max(np.abs(samples[-1][1] - goal)))
        if end_error > 0.5:
            failures.append(f"shared move finished {end_error:.2f} deg from the goal")

        start = _joints_rad(bridge.latest_feedback())
        short_goal = start.copy()
        short_goal[0] += 0.15
        short_goal[3] += 0.70
        short_samples, short_velocity, short_acceleration = _follow(
            bridge,
            "duration extended",
            "0.05 s is too fast, so the trajectory lengthens and stays inside the limits",
            short_goal,
            0.05,
            2.6,
        )
        short_arrival = _arrival(short_samples, start, short_goal)
        if short_arrival < 1.0:
            failures.append(f"short request arrived at {short_arrival:.3f} s, limits should extend it")
        if short_arrival > 2.4:
            failures.append(f"extended trajectory did not finish, arrival {short_arrival:.3f} s")
        peak_velocity = np.maximum(peak_velocity, short_velocity)
        peak_acceleration = np.maximum(peak_acceleration, short_acceleration)
        if np.any(peak_velocity > _MAX_VELOCITY * 1.05):
            failures.append(f"velocity exceeded the URDF limit: {peak_velocity} rad/s")
        if np.any(peak_acceleration > _MAX_ACCELERATION * 1.25):
            failures.append(f"acceleration exceeded the limit: {peak_acceleration} rad/s^2")
        print(f"\npeak velocity rad/s {_fmt(peak_velocity, 3)}")
        print(f"peak acceleration rad/s^2 {_fmt(peak_acceleration, 3)}")
        print(f"limit velocity rad/s {_fmt(_MAX_VELOCITY, 3)}")
        print(f"limit acceleration rad/s^2 {_fmt(_MAX_ACCELERATION, 3)}")
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
