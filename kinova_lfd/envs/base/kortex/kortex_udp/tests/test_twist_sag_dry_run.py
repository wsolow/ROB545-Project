"""Replay the logged Gen3 pose with the shoulder sag seen on the robot.

The real 1.5 s logs, at 0.01 m/s and 0.01 deg/s, left the Kortex tool pose
frozen while joint 2 crept positive. Forward kinematics of those joints is
about 2 mm down for the linear twist and 3 mm down for the angular twist.
The dry-run plant sags the same way unless the position target moves far
enough from the feedback to be tracked.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KORTEX_UDP_DRY_RUN_SAG"] = "1"

from kinova_lfd.envs.base.kortex.kortex_udp.log_twist_pose import _MOVE_S, _TRIALS
from kinova_lfd.envs.base.kortex.kortex_udp.test_twist_dry_run import _fmt, _tool_pose
from kinova_lfd.envs.base.kortex.kortex_udp.kortex_udp_bridge import KortexUDPBridge

_LOG = os.path.join(os.path.dirname(__file__), "twist_pose_log.npz")
_REAL_JOINT2_DEG = {"linear_+X": 0.315, "angular_+wy": 0.431}
_REAL_DZ_MM = {"linear_+X": -2.03, "angular_+wy": -3.37}


def _stream(bridge: KortexUDPBridge, twist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat one twist for the logged duration and return the pose change."""
    origin_p, origin_r, origin_q = _tool_pose(bridge.latest_feedback())
    opened = time.monotonic()
    next_tick = opened
    while time.monotonic() - opened < _MOVE_S:
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        next_tick += 1.0 / 30.0
        bridge.kortex.send_twist(twist, 0.0)
    position, rotation, joints = _tool_pose(bridge.latest_feedback())
    tool_mm = origin_r.T @ (position - origin_p) * 1000.0
    world_mm = (position - origin_p) * 1000.0
    return tool_mm, world_mm, joints - origin_q


def main() -> None:
    """Check that the logged twists move in the commanded direction instead of down."""
    start_deg = np.load(_LOG)["linear_+X/joints_deg"][0]
    os.environ["KORTEX_UDP_DRY_RUN_JOINTS_DEG"] = ",".join(f"{value:.8f}" for value in start_deg)
    failures: list[str] = []
    bridge = KortexUDPBridge("192.168.1.10", dry_run=True)
    try:
        bridge.start()
        for name, twist in _TRIALS:
            tool_mm, world_mm, joint_delta = _stream(bridge, twist)
            print(
                f"{name} tool mm {_fmt(tool_mm, 2)}  world mm {_fmt(world_mm, 2)}  "
                f"joint delta deg {_fmt(joint_delta, 3)}"
            )
            print(
                f"  real robot was joint 2 {_REAL_JOINT2_DEG[name]:+.3f} deg and "
                f"tool z {_REAL_DZ_MM[name]:+.2f} mm"
            )
            if world_mm[2] < -1.0:
                failures.append(f"{name} world z dropped {world_mm[2]:.2f} mm")
            if abs(float(joint_delta[1])) > 0.2:
                failures.append(f"{name} joint 2 moved {joint_delta[1]:.3f} deg")
            if name == "linear_+X" and not 10.0 <= float(tool_mm[0]) <= 20.0:
                failures.append(f"{name} tool x was {tool_mm[0]:.2f} mm, expected about 15 mm")
            if name == "angular_+wy" and float(np.linalg.norm(tool_mm)) > 2.0:
                failures.append(f"{name} tool moved {np.linalg.norm(tool_mm):.2f} mm")
    finally:
        bridge.close()
    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    print("PASS")


if __name__ == "__main__":
    main()
