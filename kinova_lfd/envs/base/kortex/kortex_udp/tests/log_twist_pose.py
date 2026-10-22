"""Log a commanded tool twist next to the measured Gen3 pose.

Two motions only, each at most 1.5 s. Linear speed is at most 0.01 m/s and
angular speed is at most 0.01 deg/s. The gripper opening is held at the value
measured before the first twist.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

_ROBOT_IP = "192.168.1.10"
_DT = 1.0 / 30.0
_MOVE_S = 1.5
_LOG_PATH = Path(__file__).resolve().parent / "twist_pose_log.npz"
# Tool frame: linear m/s, then angular deg/s.
_TRIALS = (
    ("linear_+X", np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("angular_+wy", np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0])),
)


def _fmt(values: np.ndarray, digits: int = 4) -> str:
    """Format a vector with a fixed number of digits."""
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def _sample(feedback: object) -> dict[str, np.ndarray]:
    """Read tool pose, tool twist, and arm joints from one feedback message."""
    base = feedback.base  # type: ignore[attr-defined]
    joints = np.array([actuator.position for actuator in feedback.actuators], dtype=np.float64)  # type: ignore[attr-defined]
    return {
        "xyz": np.array([base.tool_pose_x, base.tool_pose_y, base.tool_pose_z], dtype=np.float64),
        "theta_deg": np.array(
            [base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z],
            dtype=np.float64,
        ),
        "linear": np.array(
            [base.tool_twist_linear_x, base.tool_twist_linear_y, base.tool_twist_linear_z],
            dtype=np.float64,
        ),
        "angular": np.array(
            [base.tool_twist_angular_x, base.tool_twist_angular_y, base.tool_twist_angular_z],
            dtype=np.float64,
        ),
        "joints_deg": joints,
    }


def _gripper_opening(feedback: object) -> float:
    """Return the measured gripper opening in [0, 1]."""
    motors = feedback.interconnect.gripper_feedback.motor  # type: ignore[attr-defined]
    if not motors:
        return 0.0
    return float(np.clip(motors[0].position / 100.0, 0.0, 1.0))


def _run_trial(
    bridge: KortexUDPBridge,
    name: str,
    twist: np.ndarray,
    gripper: float,
) -> dict[str, np.ndarray]:
    """Repeat one twist at 30 Hz and record the pose at each send."""
    rows: list[dict[str, np.ndarray]] = []
    opened = time.monotonic()
    next_tick = opened
    while True:
        elapsed = time.monotonic() - opened
        if elapsed >= _MOVE_S:
            break
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        next_tick += _DT
        bridge.kortex.send_twist(twist, gripper)
        stamp = time.monotonic() - opened
        sample = _sample(bridge.latest_feedback())
        rows.append(sample)
        print(
            "  ".join(
                [
                    f"{name} t={stamp:6.3f}s",
                    f"cmd {_fmt(twist, 3)}",
                    f"xyz m {_fmt(sample['xyz'], 4)}",
                    f"theta deg {_fmt(sample['theta_deg'], 2)}",
                    f"meas linear {_fmt(sample['linear'], 4)}",
                    f"meas angular {_fmt(sample['angular'], 3)}",
                    f"joints deg {_fmt(sample['joints_deg'], 2)}",
                ]
            )
        )
    stacked = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    origin = stacked["xyz"][0]
    delta_mm = (stacked["xyz"][-1] - origin) * 1000.0
    joint_delta = stacked["joints_deg"][-1] - stacked["joints_deg"][0]
    print(f"{name} delta xyz mm {_fmt(delta_mm, 2)}  joint delta deg {_fmt(joint_delta, 3)}")
    stacked["command"] = np.tile(twist, (stacked["xyz"].shape[0], 1))
    stacked["t"] = np.linspace(0.0, _MOVE_S, stacked["xyz"].shape[0], endpoint=False)
    return stacked


def main() -> None:
    """Send the two allowed twists and save the pose log."""
    bridge = KortexUDPBridge(_ROBOT_IP, dry_run=False)
    saved: dict[str, np.ndarray] = {}
    try:
        bridge.start()
        gripper = _gripper_opening(bridge.latest_feedback())
        start = _sample(bridge.latest_feedback())
        print(f"start xyz m {_fmt(start['xyz'], 4)}  theta deg {_fmt(start['theta_deg'], 2)}")
        print(f"start joints deg {_fmt(start['joints_deg'], 2)}  gripper {gripper:.3f}")
        for name, twist in _TRIALS:
            print(f"\n== {name} for {_MOVE_S:.1f} s ==")
            trial = _run_trial(bridge, name, twist, gripper)
            for key, value in trial.items():
                saved[f"{name}/{key}"] = value
            time.sleep(0.2)
    finally:
        bridge.close()
        if saved:
            np.savez(_LOG_PATH, **saved)
            print(f"wrote {_LOG_PATH}")


if __name__ == "__main__":
    main()
