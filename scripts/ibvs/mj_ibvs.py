"""mj_ibvs.py.

Keyboard teleoperation of the MuJoCo Gen3 with an on-demand IBVS grasp.

Run from the repository root:
    python -m scripts.ibvs.mj_ibvs [--setpoint <wrist_png>]

Written by Will Solow, 2026.
"""

import argparse

import numpy as np

from kinova_lfd.agents.ibvs import MjIbvsPolicyAdapter
from kinova_lfd.agents.ibvs.util import add_ibvs_args, build_controller, load_setpoint_rgb, rgbd_to_numpy
from kinova_lfd.controllers import IBVSController
from kinova_lfd.envs.base.viewers import MjViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.teleop.devices import Se3Keyboard, Se3KeyboardCfg

parser = argparse.ArgumentParser(description="Keyboard teleop + IBVS grasping in MuJoCo Gen3 environments.")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Environment.")
parser.add_argument("--pos-sensitivity", type=float, default=0.15, help="Teleop linear speed (m/s).")
parser.add_argument("--rot-sensitivity", type=float, default=0.2, help="Teleop angular speed (rad/s).")
parser.add_argument(
    "--setpoint-dir",
    type=str,
    default="exp/ibvs_setpoints/mj",
    help="Directory where B saves wrist-frame setpoints.",
)
add_ibvs_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    """Run keyboard teleop with an on-demand IBVS grasp in MuJoCo."""
    env = create_env(args_cli.task, {})
    env.reset()

    def make_controller(setpoint_rgb: np.ndarray, K: np.ndarray) -> IBVSController:
        return build_controller(args_cli, setpoint_rgb, K, dt=env.step_dt)

    controller = None
    if args_cli.setpoint is not None:
        rgbd = env._get_latest_rgbd_wrist()
        if rgbd is None:
            raise RuntimeError("Wrist RGBD unavailable after reset; cannot initialize IBVS.")
        _, _, K = rgbd_to_numpy(rgbd)
        controller = make_controller(load_setpoint_rgb(args_cli.setpoint), K)
    else:
        print("[INFO][IBVS-TELEOP] No --setpoint given; press B to capture one from the wrist camera.")

    teleop = Se3Keyboard(
        Se3KeyboardCfg(pos_sensitivity=args_cli.pos_sensitivity, rot_sensitivity=args_cli.rot_sensitivity)
    )
    policy = MjIbvsPolicyAdapter(
        env,
        teleop,
        controller_factory=make_controller,
        controller=controller,
        setpoint_dir=args_cli.setpoint_dir,
    )
    MjViewer(
        env,
        policy,
        view_tabletop=True,
        view_wrist=True,
    ).run()


if __name__ == "__main__":
    main()
