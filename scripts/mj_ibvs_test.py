"""mj_ibvs_test.py.

Random-init IBVS grasp test under MjViewer: sample a solvable TCP pose
(block visible in the wrist camera), run IBVS→descend→close→lift, log
phases / targets, and score success from the pink block's world height.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import override

import cv2
import numpy as np
import torch

from kinova_lfd.controllers import IBVSController, get_block_hsv_preset
from kinova_lfd.controllers.ibvs import BlockHsvCfg, GraspPhase, extract_block_corners_px
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.core.tensor_math import matrix_from_quat
from kinova_lfd.envs.base.mujoco.viewer import MjViewer
from kinova_lfd.envs.factory import create_env

parser = argparse.ArgumentParser(description="Random-init IBVS grasp test in MuJoCo.")
parser.add_argument("--task", type=str, default="Mj-Gen3-v0", help="Environment.")
parser.add_argument(
    "--setpoint",
    type=str,
    default="exp/ibvs_setpoints/wrist_pregrasp_z0.28.png",
    help="Path to wrist-camera RGB setpoint image.",
)
parser.add_argument("--block-color", type=str, default="pink", help="HSV preset name.")
parser.add_argument("--hsv-low", type=int, nargs=3, default=None, metavar=("H", "S", "V"))
parser.add_argument("--hsv-high", type=int, nargs=3, default=None, metavar=("H", "S", "V"))
parser.add_argument("--lambda", dest="lambda_gain", type=float, default=0.5, help="IBVS gain λ.")
parser.add_argument("--z-default", type=float, default=0.4, help="Fallback feature depth (m).")
parser.add_argument("--max-lin-vel", type=float, default=0.15, help="Max linear TCP speed (m/s).")
parser.add_argument("--max-ang-vel", type=float, default=0.5, help="Max angular TCP speed (rad/s).")
parser.add_argument("--grasp-z-offset", type=float, default=0.08, help="World-Z descend after IBVS (m).")
parser.add_argument("--ibvs-error-threshold", type=float, default=0.05, help="IBVS settle error norm.")
parser.add_argument("--seed", type=int, default=0, help="RNG seed for TCP init.")
parser.add_argument("--max-init-tries", type=int, default=25, help="Solvable-init retries.")
parser.add_argument("--xy-noise", type=float, default=0.04, help="TCP XY half-range about block (m).")
parser.add_argument(
    "--z-range",
    type=float,
    nargs=2,
    default=(0.26, 0.32),
    metavar=("Z_MIN", "Z_MAX"),
    help="TCP Z sample range (m).",
)
parser.add_argument("--lift-margin", type=float, default=0.04, help="Success ΔZ for block height (m).")
parser.add_argument("--log-every", type=int, default=25, help="Periodic log interval (viewer steps).")
parser.add_argument("--max-steps", type=int, default=15000, help="Abort if FSM never reaches DONE.")
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

BLOCK_XY = np.array([0.35, 0.0], dtype=np.float64)


def _load_setpoint_rgb(path: str | Path) -> np.ndarray:
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

    depth = rgbd.get("depth")
    if depth is not None and isinstance(depth, torch.Tensor):
        depth = depth.squeeze().detach().cpu().numpy()

    K = rgbd["intrinsic_k"]
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    return rgb, depth, np.asarray(K, dtype=np.float64).reshape(3, 3)


def _tcp_pose_numpy(env) -> np.ndarray:
    pose = env._robot_tool_pose_b
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:7]


def _block_pose_numpy(env) -> np.ndarray:
    pose = env._cubes["pink_block"].data.root_link_pose_w
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    return np.asarray(pose, dtype=np.float64).reshape(-1)[:7]


def _servo_tcp_xyz(env, target_xyz: np.ndarray, *, max_steps: int = 1500, kp: float = 1.5, tol: float = 0.01) -> float:
    """P-control twist servo to a world XYZ target; keep current orientation."""
    target = torch.as_tensor(target_xyz, dtype=torch.float32, device=env.device)
    err = float("inf")
    for _ in range(max_steps):
        pose = env._robot_tool_pose_b[0]
        err_w = target - pose[:3]
        err = float(torch.linalg.norm(err_w))
        if err < tol:
            zero = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
            for _ in range(20):
                env.step(zero, action_spec=ActionSpec.TWIST_TCP)
            return err
        R = matrix_from_quat(pose[3:7].unsqueeze(0))[0]
        v_tcp = R.T @ (kp * err_w)
        n = torch.linalg.norm(v_tcp)
        if n > 0.12:
            v_tcp = v_tcp * (0.12 / n)
        action = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
        action[0, :3] = v_tcp
        env.step(action, action_spec=ActionSpec.TWIST_TCP)
    return err


def sample_solvable_init(
    env,
    rng: np.random.Generator,
    block_hsv: BlockHsvCfg,
    *,
    xy_noise: float,
    z_range: tuple[float, float],
    max_tries: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random TCP near the block until the pink block is visible."""
    z_lo, z_hi = float(z_range[0]), float(z_range[1])
    for attempt in range(1, max_tries + 1):
        env.reset()
        # Settle cameras / physics briefly.
        zero = torch.zeros((1, 7), device=env.device, dtype=torch.float32)
        for _ in range(10):
            env.step(zero, action_spec=ActionSpec.TWIST_TCP)

        xy = BLOCK_XY + rng.uniform(-xy_noise, xy_noise, size=2)
        z = float(rng.uniform(z_lo, z_hi))
        target = np.array([xy[0], xy[1], z], dtype=np.float64)
        err = _servo_tcp_xyz(env, target)
        tcp = _tcp_pose_numpy(env)
        rgbd = env._get_latest_rgbd_wrist()
        if rgbd is None:
            print(f"[init] try {attempt}/{max_tries}: no wrist RGB (tcp_err={err:.3f})")
            continue
        rgb, _, _ = _rgbd_to_numpy(rgbd)
        corners = extract_block_corners_px(rgb, block_hsv)
        if corners is None:
            print(
                f"[init] try {attempt}/{max_tries}: block not visible at "
                f"target=[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] "
                f"tcp=[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]"
            )
            continue
        print(
            f"[init] accepted try {attempt}/{max_tries}: "
            f"target=[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] "
            f"tcp=[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}] servo_err={err:.4f}"
        )
        return target, tcp

    raise RuntimeError(f"Failed to find a solvable wrist-visible init in {max_tries} tries.")


class LoggingIbvsPolicy:
    """IBVS policy that logs phase changes, TCP targets, and block height success."""

    def __init__(
        self,
        env,
        controller: IBVSController,
        *,
        lift_margin: float,
        log_every: int,
        max_steps: int,
    ) -> None:
        self.env = env
        self.controller = controller
        self.lift_margin = float(lift_margin)
        self.log_every = max(int(log_every), 1)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.done = False
        self.success: bool | None = None
        self._last_phase = controller.phase
        self.block_z_init = float(_block_pose_numpy(env)[2])
        self.block_pose_init = _block_pose_numpy(env).copy()
        self.block_z_final: float | None = None

    def _maybe_log_phase(self) -> None:
        phase = self.controller.phase
        if phase is self._last_phase:
            return
        print(f"[phase] {self._last_phase.name} -> {phase.name}")
        sp = self.controller._pose_setpoint
        if sp is not None and phase in (GraspPhase.DESCEND, GraspPhase.CLOSE, GraspPhase.LIFT):
            pre = self.controller._pregrasp_pose
            if pre is not None:
                print(
                    f"[target] pose setpoint xyz=[{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}] "
                    f"(pregrasp_z={pre[2]:.4f})"
                )
            else:
                print(f"[target] pose setpoint xyz=[{sp[0]:.4f}, {sp[1]:.4f}, {sp[2]:.4f}]")
        if phase is GraspPhase.DESCEND:
            print("[phase] pre-grasp reached; descending to grasp height")
        elif phase is GraspPhase.CLOSE:
            print("[phase] grasp: closing gripper")
        elif phase is GraspPhase.LIFT:
            print("[phase] lift: raising block")
        elif phase is GraspPhase.DONE:
            self._score_and_finish()
        self._last_phase = phase

    def _score_and_finish(self) -> None:
        block = _block_pose_numpy(self.env)
        tcp = _tcp_pose_numpy(self.env)
        self.block_z_final = float(block[2])
        dz = self.block_z_final - self.block_z_init
        xy_err = float(np.linalg.norm(block[:2] - tcp[:2]))
        self.success = self.block_z_final >= self.block_z_init + self.lift_margin
        print(
            f"[result] block_z_init={self.block_z_init:.4f} "
            f"block_z_final={self.block_z_final:.4f} Δz={dz:.4f} "
            f"lift_margin={self.lift_margin:.4f} block_xy_to_tcp={xy_err:.4f}"
        )
        if self.success:
            print("[result] SUCCESS: grasp and lift cleared block-height threshold")
        else:
            print(
                f"[result] FAIL: block still near table "
                f"(block_xyz=[{block[0]:.4f}, {block[1]:.4f}, {block[2]:.4f}], "
                f"tcp_xyz=[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}], "
                f"phase={self.controller.phase.name})"
            )
        self.done = True

    def __call__(self, observation) -> torch.Tensor:
        del observation
        self.step_count += 1
        rgbd = self.env._get_latest_rgbd_wrist()
        tcp_pose = _tcp_pose_numpy(self.env)
        dt = float(getattr(self.env, "dt", self.controller.dt))
        if rgbd is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = None
        else:
            rgb, depth, _ = _rgbd_to_numpy(rgbd)

        twist = self.controller(rgb, depth, tcp_pose=tcp_pose, dt=dt)
        self._maybe_log_phase()

        if self.step_count % self.log_every == 0 and not self.done:
            block = _block_pose_numpy(self.env)
            sp = self.controller._pose_setpoint
            sp_str = (
                f"[{sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}]" if sp is not None else "None"
            )
            err = self.controller._last_error_norm
            err_str = f"{err:.4f}" if err is not None else "n/a"
            print(
                f"[step {self.step_count:5d}] phase={self.controller.phase.name} "
                f"tcp=[{tcp_pose[0]:.3f}, {tcp_pose[1]:.3f}, {tcp_pose[2]:.3f}] "
                f"target={sp_str} ibvs_err={err_str} "
                f"block_z={block[2]:.4f}"
            )

        if not self.done and self.step_count >= self.max_steps:
            print(f"[result] FAIL: exceeded --max-steps={self.max_steps} (phase={self.controller.phase.name})")
            block = _block_pose_numpy(self.env)
            self.block_z_final = float(block[2])
            self.success = False
            self.done = True

        return torch.as_tensor(twist, device=self.env.device, dtype=torch.float32).unsqueeze(0)


class IbvsTestViewer(MjViewer):
    """MjViewer that stops when the logging policy reports done."""

    def __init__(self, env, policy: LoggingIbvsPolicy, **kwargs):
        super().__init__(env, policy, **kwargs)
        self._test_policy = policy

    @override
    def _execute_step(self) -> bool:
        ok = super()._execute_step()
        if self._test_policy.done:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self._interrupted = True
            print("[viewer] stopping (test finished)")
        return ok

    @override
    def close(self) -> None:
        # MuJoCo passive viewer + OpenCV windows can segfault on v.close().
        # Drop the handle and let process exit clean up native resources.
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.viewer = None


def main() -> None:
    rng = np.random.default_rng(args_cli.seed)
    env = create_env(args_cli.task, {})
    env.reset()

    block_hsv = _build_block_hsv(args_cli)
    target, tcp = sample_solvable_init(
        env,
        rng,
        block_hsv,
        xy_noise=args_cli.xy_noise,
        z_range=tuple(args_cli.z_range),
        max_tries=args_cli.max_init_tries,
    )
    block = _block_pose_numpy(env)
    print(
        f"[scene] pink_block xyz=[{block[0]:.4f}, {block[1]:.4f}, {block[2]:.4f}] "
        f"quat_wxyz=[{block[3]:.3f}, {block[4]:.3f}, {block[5]:.3f}, {block[6]:.3f}]"
    )
    print(
        f"[scene] tcp xyz=[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}] "
        f"offset_from_block_xy=[{tcp[0] - block[0]:.4f}, {tcp[1] - block[1]:.4f}] "
        f"sampled_target=[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
    )

    rgbd = env._get_latest_rgbd_wrist()
    if rgbd is None:
        raise RuntimeError("Wrist RGBD unavailable after solvable init.")
    _, _, K = _rgbd_to_numpy(rgbd)

    controller = IBVSController(
        _load_setpoint_rgb(args_cli.setpoint),
        intrinsics_K=K,
        block_hsv=block_hsv,
        lambda_gain=args_cli.lambda_gain,
        z_default=args_cli.z_default,
        max_lin_vel=args_cli.max_lin_vel,
        max_ang_vel=args_cli.max_ang_vel,
        gripper_cmd=0.0,
        grasp_z_offset=args_cli.grasp_z_offset,
        ibvs_error_threshold=args_cli.ibvs_error_threshold,
        dt=float(getattr(env, "dt", 0.01)),
    )
    policy = LoggingIbvsPolicy(
        env,
        controller,
        lift_margin=args_cli.lift_margin,
        log_every=args_cli.log_every,
        max_steps=args_cli.max_steps,
    )
    print(
        f"[test] starting IBVS (block_z_init={policy.block_z_init:.4f}, "
        f"lift_margin={args_cli.lift_margin:.4f}, setpoint={args_cli.setpoint})"
    )

    IbvsTestViewer(
        env,
        policy,
        view_tabletop=args_cli.tabletop_camera,
        view_wrist=args_cli.wrist_camera,
    ).run()

    # Hard-exit: MuJoCo passive viewer teardown often segfaults with OpenCV windows.
    os._exit(0 if policy.success is True else 1)


if __name__ == "__main__":
    main()
