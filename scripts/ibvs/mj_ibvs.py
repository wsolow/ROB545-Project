"""mj_ibvs.py.

Run image-based visual servoing in MuJoCo toward a wrist-camera setpoint image.

Run from the repository root with ``python -m scripts.ibvs.mj_ibvs --setpoint <png>``.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from kinova_lfd.controllers import IBVSController
from kinova_lfd.envs.base.mujoco.viewer import MjViewer
from kinova_lfd.envs.factory import create_env
from scripts.ibvs.ibvs_util import add_ibvs_args, build_controller, load_setpoint_rgb, rgbd_to_numpy, tcp_pose_numpy

parser = argparse.ArgumentParser(description="IBVS control for MuJoCo Gen3 environments.")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Environment.")
add_ibvs_args(parser)

args_cli = parser.parse_args()


class IbvsPolicyAdapter:
    """MjViewer-compatible policy: ignores Gym obs, feeds wrist RGBD + TCP pose into IBVS."""

    def __init__(self, env, controller: IBVSController) -> None:
        self.env = env
        self.controller = controller

    def __call__(self, observation) -> torch.Tensor:
        del observation  # IBVS uses wrist RGBD + TCP pose, not joint observations.
        rgbd = self.env._get_latest_rgbd_wrist()
        tcp_pose = tcp_pose_numpy(self.env)
        dt = float(getattr(self.env, "dt", self.controller.dt))
        if rgbd is None:
            twist = self.controller(
                np.zeros((480, 640, 3), dtype=np.uint8),
                depth=None,
                tcp_pose=tcp_pose,
                dt=dt,
            )
        else:
            rgb, depth, _ = rgbd_to_numpy(rgbd)
            twist = self.controller(rgb, depth, tcp_pose=tcp_pose, dt=dt)
        return torch.as_tensor(twist, device=self.env.device, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    """Run IBVS with a MuJoCo Gen3 environment and wrist-camera setpoint."""
    env = create_env(args_cli.task, {})
    env.reset()

    rgbd = env._get_latest_rgbd_wrist()
    if rgbd is None:
        raise RuntimeError("Wrist RGBD unavailable after reset; cannot initialize IBVS.")
    _, _, K = rgbd_to_numpy(rgbd)

    setpoint_rgb = load_setpoint_rgb(args_cli.setpoint)
    controller = build_controller(args_cli, setpoint_rgb, K, dt=float(getattr(env, "dt", 0.01)))
    policy = IbvsPolicyAdapter(env, controller)

    MjViewer(
        env,
        policy,
        view_tabletop=args_cli.tabletop_camera,
        view_wrist=args_cli.wrist_camera,
    ).run()


if __name__ == "__main__":
    main()
