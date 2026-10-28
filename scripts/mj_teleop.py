"""mj_teleop.py.

Script to run a keyboard or joystick teleoperation with Mujoco manipulation environments.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

# from kinova_lfd.envs.base.mujoco.viewer import NativeMujocoViewer
from kinova_lfd.envs.base.mujoco.viewer import MjViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.teleop.devices import (
    Se3Keyboard,
    Se3KeyboardCfg,
    VRHeadset,
    VRHeadsetCfg,
    VRJoystick,
    VRJoystickCfg,
    XboxController,
    XboxControllerCfg,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Kortex environments.")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Environment.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices={"keyboard", "vr_joystick", "vr_headset", "xbox"},
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--log_dir", type=str, default="exp/tabletop_admittance/", help="Log location")
parser.add_argument(
    "--save_dir",
    type=str,
    default="exp/ibvs_setpoints/",
    help="Directory for wrist RGB snapshots (press B while teleoping).",
)
parser.add_argument(
    "--tabletop_camera",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="View table camera",
)
parser.add_argument(
    "--wrist_camera",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="View wrist camera",
)

# parse the arguments
args_cli = parser.parse_args()


def _rgb_from_rgbd(rgbd: dict) -> np.ndarray | None:
    """Convert env wrist RGBD tensor dict to HxWx3 uint8 RGB."""
    if rgbd is None or "rgb" not in rgbd:
        return None
    rgb = rgbd["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.squeeze().detach().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(rgb.dtype, np.floating):
        max_val = float(np.nanmax(rgb)) if rgb.size else 1.0
        rgb = (
            (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            if max_val <= 1.0 + 1e-3
            else np.clip(rgb, 0, 255).astype(np.uint8)
        )
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def main() -> None:
    """Run keyboard teleoperation with Kortex API.

    Defaults to Kortex-Gen3-v0 environment and passes the twist_tcp Action Specification.
    Press B (keyboard teleop) to save the current wrist RGB into --save_dir.
    """
    env_cfg = {}

    env = create_env(args_cli.task, env_cfg)

    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.15, rot_sensitivity=0.2, reference_frame="tcp"))
    elif args_cli.teleop_device.lower() == "vr_joystick":
        teleop_interface = VRJoystick(VRJoystickCfg(pos_sensitivity=0.15, rot_sensitivity=0.5))
    elif args_cli.teleop_device.lower() == "vr_headset":
        teleop_interface = VRHeadset(VRHeadsetCfg(pos_sensitivity=0.15, rot_sensitivity=0.5))
    elif args_cli.teleop_device.lower() == "xbox":
        teleop_interface = XboxController(XboxControllerCfg(pos_sensitivity=0.12, rot_sensitivity=25))
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_count = {"n": 0}

    def _save_wrist_rgb() -> None:
        rgb = _rgb_from_rgbd(env._get_latest_rgbd_wrist())
        if rgb is None:
            print("[teleop] No wrist RGB available to save.")
            return
        pose = env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy().reshape(-1)
        z = float(pose[2]) if pose is not None else float("nan")
        save_count["n"] += 1
        out = save_dir / f"wrist_z{z:.3f}_{save_count['n']:03d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[teleop] Saved wrist RGB -> {out} (tcp_z={z:.3f})")

    if isinstance(teleop_interface, Se3Keyboard):
        teleop_interface._additional_callbacks["B"] = _save_wrist_rgb
        print(f"[teleop] Press B to save wrist RGB into {save_dir.resolve()}")

    env.reset()

    MjViewer(env, teleop_interface, view_tabletop=args_cli.tabletop_camera, view_wrist=args_cli.wrist_camera).run()


if __name__ == "__main__":
    main()
