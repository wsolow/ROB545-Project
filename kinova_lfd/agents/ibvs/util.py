import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from kinova_lfd.controllers import BlockHsvCfg, IBVSController, get_block_hsv_preset


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


def add_ibvs_args(parser: argparse.ArgumentParser) -> None:
    """Add IBVS related arguments to the argparser."""
    parser.add_argument(
        "--setpoint",
        type=str,
        required=False,
        help="Path to a wrist-camera RGB setpoint image.",
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
    parser.add_argument("--max-lin-vel", type=float, default=0.05, help="Max linear TCP speed (m/s).")
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
        default=True,
        help="Include IBVS angular DoF (default: translation-only, no tilt).",
    )
