"""mj_ibvs.py.

Run image-based visual servoing in MuJoCo toward a wrist-camera setpoint image.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from kinova_lfd.controllers import BlockHsvCfg, IBVSController, get_block_hsv_preset
from kinova_lfd.envs.base.mujoco.viewer import MjViewer
from kinova_lfd.envs.factory import create_env

parser = argparse.ArgumentParser(description="IBVS control for MuJoCo Gen3 environments.")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Environment.")
parser.add_argument(
    "--setpoint",
    type=str,
    required=True,
    help="Path to a wrist-camera RGB setpoint image taken from the MuJoCo environment.",
)
parser.add_argument(
    "--block-color",
    type=str,
    default="pink",
    help="Named HSV preset (pink, red, green, blue, yellow).",
)
parser.add_argument(
    "--hsv-low",
    type=int,
    nargs=3,
    default=None,
    metavar=("H", "S", "V"),
    help="Optional HSV lower bound override (OpenCV 0-179, 0-255, 0-255).",
)
parser.add_argument(
    "--hsv-high",
    type=int,
    nargs=3,
    default=None,
    metavar=("H", "S", "V"),
    help="Optional HSV upper bound override (OpenCV 0-179, 0-255, 0-255).",
)
parser.add_argument("--lambda", dest="lambda_gain", type=float, default=0.35, help="IBVS gain λ.")
parser.add_argument(
    "--z-default",
    type=float,
    default=0.28,
    help="Fallback feature depth (m) when wrist depth is unavailable.",
)
parser.add_argument("--max-lin-vel", type=float, default=0.15, help="Max linear TCP speed (m/s).")
parser.add_argument("--max-ang-vel", type=float, default=0.15, help="Max angular TCP speed (rad/s).")
parser.add_argument(
    "--grasp-z-offset",
    type=float,
    default=0.24,
    help="World-Z meters below IBVS pre-grasp (≈0.24 for z0.28 setpoint -> ~0.03 m).",
)
parser.add_argument(
    "--grasp-z-abs",
    type=float,
    default=0.03,
    help="Absolute world-Z ceiling for grasp height (pinch near block).",
)
parser.add_argument(
    "--ibvs-error-threshold",
    type=float,
    default=0.02,
    help="Normalized feature-error norm that ends the IBVS phase.",
)
parser.add_argument(
    "--ibvs-settle-steps",
    type=int,
    default=30,
    help="Consecutive steps below error threshold before DESCEND.",
)
parser.add_argument(
    "--use-angular",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Include IBVS angular DoF (default: translation-only, no tilt).",
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

args_cli = parser.parse_args()


def _load_setpoint_rgb(path: str | Path) -> np.ndarray:
    """Load a setpoint image as RGB uint8 HWC."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read setpoint image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _build_block_hsv(args: argparse.Namespace) -> BlockHsvCfg:
    cfg = get_block_hsv_preset(args.block_color)
    if args.hsv_low is not None or args.hsv_high is not None:
        if args.hsv_low is None or args.hsv_high is None:
            raise ValueError("Provide both --hsv-low and --hsv-high to override the preset.")
        cfg.hsv_ranges = [(tuple(args.hsv_low), tuple(args.hsv_high))]
    return cfg


def _rgbd_to_numpy(rgbd: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Convert MuJoCo wrist RGBD tensors to numpy RGB, depth, K."""
    rgb = rgbd["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.squeeze().detach().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = (
            (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            if float(np.nanmax(rgb)) <= 1.0 + 1e-3
            else np.clip(rgb, 0, 255).astype(np.uint8)
        )
    else:
        rgb = rgb.astype(np.uint8)

    depth = rgbd.get("depth")
    if depth is not None and isinstance(depth, torch.Tensor):
        depth = depth.squeeze().detach().cpu().numpy()

    K = rgbd["intrinsic_k"]
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return rgb, depth, K


def _tcp_pose_numpy(env) -> np.ndarray:
    """Return 7D TCP pose (xyz + quat wxyz) as numpy."""
    pose = env._robot_tool_pose_b
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:7]


class IbvsPolicyAdapter:
    """MjViewer-compatible policy: ignores Gym obs, feeds wrist RGBD + TCP pose into IBVS."""

    def __init__(self, env, controller: IBVSController) -> None:
        self.env = env
        self.controller = controller

    def __call__(self, observation) -> torch.Tensor:
        del observation  # IBVS uses wrist RGBD + TCP pose, not joint observations.
        rgbd = self.env._get_latest_rgbd_wrist()
        tcp_pose = _tcp_pose_numpy(self.env)
        dt = float(getattr(self.env, "dt", self.controller.dt))
        if rgbd is None:
            twist = self.controller(
                np.zeros((480, 640, 3), dtype=np.uint8),
                depth=None,
                tcp_pose=tcp_pose,
                dt=dt,
            )
        else:
            rgb, depth, _ = _rgbd_to_numpy(rgbd)
            twist = self.controller(rgb, depth, tcp_pose=tcp_pose, dt=dt)
        return torch.as_tensor(twist, device=self.env.device, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    """Run IBVS with a MuJoCo Gen3 environment and wrist-camera setpoint."""
    env = create_env(args_cli.task, {})
    env.reset()

    rgbd = env._get_latest_rgbd_wrist()
    if rgbd is None:
        raise RuntimeError("Wrist RGBD unavailable after reset; cannot initialize IBVS.")
    _, _, K = _rgbd_to_numpy(rgbd)

    setpoint_rgb = _load_setpoint_rgb(args_cli.setpoint)
    block_hsv = _build_block_hsv(args_cli)
    controller = IBVSController(
        setpoint_rgb,
        intrinsics_K=K,
        block_hsv=block_hsv,
        lambda_gain=args_cli.lambda_gain,
        z_default=args_cli.z_default,
        max_lin_vel=args_cli.max_lin_vel,
        max_ang_vel=args_cli.max_ang_vel,
        gripper_cmd=0.0,
        grasp_z_abs=args_cli.grasp_z_abs,
        ibvs_error_threshold=args_cli.ibvs_error_threshold,
        ibvs_settle_steps=args_cli.ibvs_settle_steps,
        use_angular=args_cli.use_angular,
        dt=float(getattr(env, "dt", 0.01)),
    )
    policy = IbvsPolicyAdapter(env, controller)

    MjViewer(
        env,
        policy,
        view_tabletop=args_cli.tabletop_camera,
        view_wrist=args_cli.wrist_camera,
    ).run()


if __name__ == "__main__":
    main()
