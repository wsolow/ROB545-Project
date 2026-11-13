"""kortex_ibvs.py.

Keyboard teleoperation of the real Gen3 with an on-demand IBVS grasp.

Written by Will Solow, 2026.
"""

import argparse
import time

import numpy as np

from kinova_lfd.agents.ibvs import KortexIbvsTeleopAdapter
from kinova_lfd.agents.ibvs.util import add_ibvs_args, build_controller, load_setpoint_rgb, rgbd_to_numpy
from kinova_lfd.controllers import IBVSController
from kinova_lfd.envs.base.viewers import KortexViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.teleop.devices import Se3Keyboard, Se3KeyboardCfg

parser = argparse.ArgumentParser(description="Keyboard teleop + IBVS grasping on the real Kortex Gen3.")

parser.add_argument("--task", type=str, default="Kortex-TCP-Gen3-v0", choices=["Kortex-TCP-Gen3-v0"])
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--dt", type=float, default=1.0 / 20.0, help="Control period (s).")
parser.add_argument("--pos-sensitivity", type=float, default=0.15, help="Teleop linear speed (m/s).")
parser.add_argument(
    "--rot-sensitivity", type=float, default=float(np.deg2rad(10.0)), help="Teleop angular speed (rad/s)."
)
parser.add_argument(
    "--setpoint-dir",
    type=str,
    default="exp/ibvs_setpoints/kortex",
    help="Directory where B saves wrist-frame setpoints.",
)
add_ibvs_args(parser)
args_cli = parser.parse_args()


def _wait_for_wrist_frame(env, timeout: float = 10) -> dict:
    """Block until the wrist camera has produced a frame."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        rgbd = env._get_latest_rgbd_wrist()
        if rgbd is not None:
            return rgbd
        time.sleep(0.05)
    raise RuntimeError(f"No wrist camera frame within {timeout:.1f}s; check the RTSP stream.")


def main() -> None:
    """Run keyboard teleop with an on-demand IBVS grasp on the Kortex Gen3."""
    env_cfg = {
        "robot_ip": args_cli.robot_ip,
        "use_wrist_camera": True,
        "use_tabletop_camera": True,
        "dt": args_cli.dt,
    }
    env = create_env(args_cli.task, env_cfg)
    env.reset()

    def make_controller(setpoint_rgb: np.ndarray, K: np.ndarray) -> IBVSController:
        return build_controller(args_cli, setpoint_rgb, K, dt=env.step_dt)

    controller = None
    if args_cli.setpoint is not None:
        _, _, K = rgbd_to_numpy(_wait_for_wrist_frame(env))
        controller = make_controller(load_setpoint_rgb(args_cli.setpoint), K)
    else:
        print("[INFO][IBVS-TELEOP] No --setpoint given; press B to capture one from the wrist camera.")

    teleop = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=args_cli.pos_sensitivity, rot_sensitivity=args_cli.rot_sensitivity)
    )
    policy = KortexIbvsTeleopAdapter(
        env,
        teleop,
        controller_factory=make_controller,
        controller=controller,
        setpoint_dir=args_cli.setpoint_dir,
    )
    KortexViewer(
        env,
        policy,
        view_tabletop=True,
        view_wrist=True,
    ).run()


if __name__ == "__main__":
    main()
