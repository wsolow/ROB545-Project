"""ibvs_util.py.

Shared helpers for the MuJoCo and Kortex IBVS scripts.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from kinova_lfd.controllers import BlockHsvCfg, IBVSController, get_block_hsv_preset


def load_setpoint_rgb(path: str | Path) -> np.ndarray:
    """Load a setpoint image as RGB uint8 HWC."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read setpoint image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_setpoint_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    """Save an RGB uint8 HWC image so that :func:`load_setpoint_rgb` round-trips it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write setpoint image: {out}")
    return out


def build_block_hsv(args: argparse.Namespace) -> BlockHsvCfg:
    """Build the block HSV config from the named preset plus optional overrides."""
    cfg = get_block_hsv_preset(args.block_color)
    if args.hsv_low is not None or args.hsv_high is not None:
        if args.hsv_low is None or args.hsv_high is None:
            raise ValueError("Provide both --hsv-low and --hsv-high to override the preset.")
        cfg.hsv_ranges = [(tuple(args.hsv_low), tuple(args.hsv_high))]
    return cfg


def rgbd_to_numpy(rgbd: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Convert a wrist RGB-D dict (torch or numpy) to numpy RGB HWC uint8, depth, and K."""
    rgb = rgbd["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb).squeeze()
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
    rgb = np.ascontiguousarray(rgb)

    depth = rgbd.get("depth")
    if depth is not None:
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        depth = np.asarray(depth).squeeze()

    K = rgbd["intrinsic_k"]
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return rgb, depth, K


def tcp_pose_numpy(env) -> np.ndarray:
    """Return 7D TCP pose (xyz + quat wxyz) as numpy."""
    pose = env._robot_tool_pose_b
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:7]


def build_controller(
    args: argparse.Namespace,
    setpoint_rgb: np.ndarray,
    intrinsics_k: np.ndarray,
    dt: float,
) -> IBVSController:
    """Construct an :class:`IBVSController` from the shared CLI arguments."""
    return IBVSController(
        setpoint_rgb,
        intrinsics_k=intrinsics_k,
        block_hsv=build_block_hsv(args),
        lambda_gain=args.lambda_gain,
        z_default=args.z_default,
        max_lin_vel=args.max_lin_vel,
        max_ang_vel=args.max_ang_vel,
        gripper_cmd=0.0,
        grasp_z_abs=args.grasp_z_abs,
        ibvs_error_threshold=args.ibvs_error_threshold,
        ibvs_settle_steps=args.ibvs_settle_steps,
        use_angular=args.use_angular,
        dt=dt,
    )
