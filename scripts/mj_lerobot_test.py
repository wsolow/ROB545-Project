"""mj_lerobot_test.py.

Smoke-test the MuJoCo Gen3 LeRobot Robot wrapper.

Exercises both ``JOINTS_POS`` and ``TWIST_TCP`` action specs through
``MjGen3Robot.send_action`` / ``get_observation``, and checks that the arm
actually moves under each mode.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from kinova_lfd.envs.lerobot import MjGen3Robot, MjGen3RobotConfig
from kinova_lfd.envs.lerobot.base import ARM_JOINT_NAMES, EE_POSE_KEYS, GRIPPER_NAME

parser = argparse.ArgumentParser(description="Test MjGen3Robot LeRobot interface in MuJoCo.")
parser.add_argument("--device", type=str, default="cuda", help="Sim device (cuda/cpu).")
parser.add_argument(
    "--mode",
    type=str,
    default="both",
    choices=("joints", "twist", "both"),
    help="Which action mode(s) to exercise.",
)
parser.add_argument("--joint-steps", type=int, default=60, help="Steps to hold the joint target.")
parser.add_argument("--twist-steps", type=int, default=50, help="Steps to apply the TCP twist.")
parser.add_argument("--joint-delta", type=float, default=0.25, help="Rad offset applied to joint_2.")
parser.add_argument("--twist-vz", type=float, default=0.08, help="TCP linear Z velocity (m/s).")
parser.add_argument(
    "--no-cameras",
    action="store_true",
    help="Disable wrist/tabletop cameras in the observation dict.",
)

args_cli = parser.parse_args()


def _joint_obs(obs: dict) -> np.ndarray:
    return np.asarray([obs[f"{name}.pos"] for name in (*ARM_JOINT_NAMES, GRIPPER_NAME)], dtype=np.float64)


def _ee_obs(obs: dict) -> np.ndarray:
    return np.asarray([obs[key] for key in EE_POSE_KEYS], dtype=np.float64)


def _print_obs_summary(obs: dict, prefix: str = "") -> None:
    joints = _joint_obs(obs)
    ee = _ee_obs(obs)
    print(f"{prefix}joints={np.array2string(joints, precision=3, suppress_small=True)}")
    print(f"{prefix}ee_xyz={np.array2string(ee[:3], precision=3, suppress_small=True)}")
    for cam in ("wrist", "tabletop"):
        if cam in obs:
            print(f"{prefix}{cam}.shape={obs[cam].shape} dtype={obs[cam].dtype}")


def _make_robot(action_spec: str) -> MjGen3Robot:
    use_cams = not args_cli.no_cameras
    cfg = MjGen3RobotConfig(
        device=args_cli.device,
        action_spec=action_spec,
        use_wrist_camera=use_cams,
        use_tabletop_camera=use_cams,
        include_joint_positions=True,
        include_ee_pose=True,
    )
    return MjGen3Robot(cfg)


def test_joints_pos() -> bool:
    """Command a joint-space offset and verify joint_2 tracks it."""
    print("\n=== JOINTS_POS ===")
    robot = _make_robot("JOINTS_POS")
    print(f"action_features: {list(robot.action_features)}")
    print(f"observation_features: {list(robot.observation_features)}")

    robot.connect()
    try:
        obs0 = robot.get_observation()
        _print_obs_summary(obs0, prefix="  start ")

        q0 = _joint_obs(obs0)
        target = q0.copy()
        target[1] = q0[1] + args_cli.joint_delta  # joint_2

        action = {f"{name}.pos": float(target[i]) for i, name in enumerate((*ARM_JOINT_NAMES, GRIPPER_NAME))}
        print(f"  commanding joint_2: {q0[1]:+.3f} -> {target[1]:+.3f} rad for {args_cli.joint_steps} steps")

        for _ in range(args_cli.joint_steps):
            robot.send_action(action)

        obs1 = robot.get_observation()
        _print_obs_summary(obs1, prefix="  end   ")
        q1 = _joint_obs(obs1)

        moved = abs(q1[1] - q0[1])
        err = abs(q1[1] - target[1])
        ok = moved > 0.5 * abs(args_cli.joint_delta) and err < 0.1
        print(f"  Δjoint_2={moved:.4f} rad  |target_err|={err:.4f}  -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        robot.disconnect()


def test_twist_tcp() -> bool:
    """Command a TCP-frame twist and verify the end-effector pose changes.

    Twists are expressed in the tool frame (same as Kinova teleop), so a +vz
    command moves along the gripper axis rather than world +Z.
    """
    print("\n=== TWIST_TCP ===")
    robot = _make_robot("TWIST_TCP")
    print(f"action_features: {list(robot.action_features)}")

    robot.connect()
    try:
        obs0 = robot.get_observation()
        _print_obs_summary(obs0, prefix="  start ")
        ee0 = _ee_obs(obs0)

        action = {
            "tcp.vx": 0.0,
            "tcp.vy": 0.0,
            "tcp.vz": float(args_cli.twist_vz),
            "tcp.wx": 0.0,
            "tcp.wy": 0.0,
            "tcp.wz": 0.0,
            "gripper.pos": float(obs0[f"{GRIPPER_NAME}.pos"]),
        }
        print(f"  commanding tcp.vz={args_cli.twist_vz:+.3f} m/s (tool frame) for {args_cli.twist_steps} steps")

        for _ in range(args_cli.twist_steps):
            robot.send_action(action)

        # Stop the arm so the final observation is not mid-motion noise.
        stop = {**action, "tcp.vz": 0.0}
        for _ in range(5):
            robot.send_action(stop)

        obs1 = robot.get_observation()
        _print_obs_summary(obs1, prefix="  end   ")
        ee1 = _ee_obs(obs1)

        dxyz = ee1[:3] - ee0[:3]
        dist = float(np.linalg.norm(dxyz))
        # Expect a clear Cartesian move; exact direction depends on TCP orientation.
        ok = dist > 0.005
        print(
            f"  Δee_xyz={np.array2string(dxyz, precision=4, suppress_small=True)}  "
            f"|Δ|={dist:.4f} m  -> {'PASS' if ok else 'FAIL'}"
        )
        return ok
    finally:
        robot.disconnect()


def main() -> int:
    results: dict[str, bool] = {}

    if args_cli.mode in ("joints", "both"):
        results["JOINTS_POS"] = test_joints_pos()
    if args_cli.mode in ("twist", "both"):
        results["TWIST_TCP"] = test_twist_tcp()

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
