"""kortex_ibvs.py.

Keyboard teleoperation of the real Gen3 with an on-demand IBVS grasp.

The operator drives the arm with the keyboard. Pressing G hands control to the IBVS
controller, which servos to the wrist-camera setpoint, descends, closes the gripper
and lifts. Control then returns to the operator to reset the scene. G pressed during
the grasp interrupts it, and stop/home/resume/quit work in every mode.

Key bindings (in addition to the Se3Keyboard motion keys, P = gripper, F = reset):
    G  toggle IBVS grasp (start from teleop / interrupt while grasping)
    X  stop the robot (zero twist, gripper held)
    H  home the robot (MOVE_TCP to HOME_TCP_CART, gripper held)
    R  resume teleop after stop / home
    B  save the current wrist frame as a new setpoint (and use it)
    Z  quit

Run from the repository root:
    python -m scripts.ibvs.kortex_ibvs.kortex_ibvs --setpoint <wrist_png>

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from kinova_lfd.agents.moderator import HOME_TCP_CART
from kinova_lfd.controllers import IBVSController
from kinova_lfd.controllers.ibvs import GraspPhase, extract_block_corners_px
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.kortex.viewer import KortexViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.teleop.devices import Se3Keyboard, Se3KeyboardCfg
from scripts.ibvs.ibvs_util import (
    build_controller,
    load_setpoint_rgb,
    rgbd_to_numpy,
    save_setpoint_rgb,
    tcp_pose_numpy,
)

parser = argparse.ArgumentParser(description="Keyboard teleop + IBVS grasping on the real Kortex Gen3.")
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

parser.add_argument("--task", type=str, default="Kortex-TCP-Gen3-v0", choices=["Kortex-TCP-Gen3-v0"])
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--dt", type=float, default=1.0 / 20.0, help="Control period (s).")
parser.add_argument("--pos-sensitivity", type=float, default=0.15, help="Teleop linear speed (m/s).")
parser.add_argument("--rot-sensitivity", type=float, default=10.0, help="Teleop angular speed.")
parser.add_argument(
    "--setpoint-dir",
    type=str,
    default="exp/ibvs_setpoints",
    help="Directory where B saves wrist-frame setpoints.",
)
parser.add_argument(
    "--first-frame-timeout",
    type=float,
    default=10.0,
    help="Seconds to wait for the first wrist frame at startup.",
)

args_cli = parser.parse_args()


class Mode(Enum):
    """Who is commanding the robot."""

    TELEOP = auto()
    IBVS = auto()
    STOP = auto()
    HOME = auto()
    QUIT = auto()


KEY_TOGGLE = "G"
KEY_STOP = "X"
KEY_HOME = "H"
KEY_RESUME = "R"
KEY_SAVE = "B"
KEY_QUIT = "Z"

# Pending keys are resolved in this order when several arrive in one step.
_KEY_PRIORITY = (KEY_QUIT, KEY_STOP, KEY_HOME, KEY_RESUME, KEY_TOGGLE, KEY_SAVE)


class KortexIbvsTeleopAdapter:
    """KortexViewer policy that switches between keyboard teleop and an IBVS grasp.

    Key presses arrive on the pynput thread and are only latched there. All mode
    changes happen at the start of :meth:`__call__` on the control thread, so the
    controller and keyboard state are never modified mid-step.

    Outside of IBVS the keyboard's gripper value is the gripper command. When IBVS
    hands control back (grasp done or interrupted) the controller's gripper command is
    copied into the keyboard so the gripper does not snap open and drop the block.
    """

    def __init__(
        self,
        env,
        teleop: Se3Keyboard,
        controller_factory: Callable[[np.ndarray, np.ndarray], IBVSController],
        controller: IBVSController | None = None,
        setpoint_dir: str | Path = "exp/ibvs_setpoints",
        home_tcp: np.ndarray = HOME_TCP_CART,
    ) -> None:
        """Initialize the adapter.

        Args:
            env: Kortex TCP Gen3 environment.
            teleop: Keyboard device used for teleoperation.
            controller_factory: ``(setpoint_rgb, K) -> IBVSController``, used when a new
                setpoint is captured with B.
            controller: Initial controller, or ``None`` until a setpoint is captured.
            setpoint_dir: Where captured setpoints are written.
            home_tcp: Home pose ``[x, y, z, rx, ry, rz, gripper]`` (the gripper entry is
                replaced with the current gripper command).

        """
        self.env = env
        self.teleop = teleop
        self.controller = controller
        self._controller_factory = controller_factory
        self._setpoint_dir = Path(setpoint_dir)
        self._home_tcp = np.asarray(home_tcp, dtype=np.float64).copy()

        self.mode = Mode.TELEOP
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()

        for key in _KEY_PRIORITY:
            self.teleop.add_callback(key, self._make_latch(key))



    def start(self) -> None:
        """Start the keyboard listener."""
        self.teleop.reset()
        self.teleop.start()
        print(self._help_text())

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.teleop.stop()

    @property
    def should_quit(self) -> bool:
        """Whether the operator pressed quit."""
        return self.mode is Mode.QUIT

    @property
    def status_text(self) -> str:
        """Short mode description drawn on the camera previews."""
        if self.mode is Mode.IBVS and self.controller is not None:
            err = self.controller._last_error_norm
            err_txt = f" err={err:.3f}" if err is not None else " (no block)"
            return f"IBVS:{self.controller.phase.name}{err_txt}"
        if self.mode is Mode.TELEOP and self.controller is None:
            return "TELEOP (no setpoint: press B)"
        return self.mode.name

    def annotate_wrist(self, bgr: np.ndarray) -> np.ndarray:
        """Draw the setpoint corners (green) and the live detection (red) on the wrist preview."""
        if self.controller is None:
            return bgr
        target = self.controller.setpoint_corners_px.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(bgr, [target], isClosed=True, color=(0, 255, 0), thickness=2)
        live = extract_block_corners_px(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.controller.block_hsv)
        if live is not None:
            cv2.polylines(bgr, [live.reshape(-1, 1, 2).astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
        return bgr


    def __call__(self, observation) -> tuple[np.ndarray, ActionSpec]:
        """Resolve pending keys, then return the action for the current mode."""
        del observation  # Teleop and IBVS both use the TCP pose / wrist camera directly.
        self._resolve_keys()

        if self.mode is Mode.IBVS:
            return self._ibvs_action()
        if self.mode is Mode.TELEOP:
            return self._teleop_action()
        if self.mode is Mode.HOME:
            return self._home_action()
        return self._hold_action()  


    @property
    def _gripper(self) -> float:
        """Gripper command used outside of IBVS (owned by the keyboard)."""
        return float(self.teleop._gripper_value)

    def _teleop_action(self) -> tuple[np.ndarray, ActionSpec]:
        action = np.asarray(self.teleop.advance(self.env._robot_tool_pose_b), dtype=np.float64)
        return action, ActionSpec.TWIST_TCP

    def _hold_action(self) -> tuple[np.ndarray, ActionSpec]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper]), ActionSpec.TWIST_TCP

    def _home_action(self) -> tuple[np.ndarray, ActionSpec]:
        action = self._home_tcp.copy()
        action[-1] = self._gripper
        return action, ActionSpec.MOVE_TCP

    def _ibvs_action(self) -> tuple[np.ndarray, ActionSpec]:
        assert self.controller is not None
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.controller.gripper_cmd]), ActionSpec.TWIST_TCP

        rgb, depth, _ = rgbd_to_numpy(rgbd)
        twist = self.controller(rgb, depth, tcp_pose=tcp_pose_numpy(self.env), dt=self.env.step_dt)

        if self.controller.phase is GraspPhase.DONE:
            self._return_to_teleop("Grasp complete")
        return np.asarray(twist, dtype=np.float64), ActionSpec.TWIST_TCP


    def _make_latch(self, key: str) -> Callable[[], None]:
        def _latch() -> None:
            with self._pending_lock:
                self._pending.add(key)

        return _latch

    def _resolve_keys(self) -> None:
        """Consume latched key presses on the control thread."""
        with self._pending_lock:
            pending, self._pending = self._pending, set()
        if not pending:
            return
        handlers = {
            KEY_QUIT: self._on_quit,
            KEY_STOP: self._on_stop,
            KEY_HOME: self._on_home,
            KEY_RESUME: self._on_resume,
            KEY_TOGGLE: self._on_toggle,
            KEY_SAVE: self._on_save_setpoint,
        }
        for key in _KEY_PRIORITY:
            if key in pending:
                handlers[key]()

    def _on_quit(self) -> None:
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.QUIT, "Quitting")

    def _on_stop(self) -> None:
        if self.mode not in (Mode.TELEOP, Mode.IBVS, Mode.HOME):
            return
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.STOP, "Stopping the robot (R to Resume)")

    def _on_home(self) -> None:
        if self.mode not in (Mode.TELEOP, Mode.IBVS, Mode.STOP):
            return
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.HOME, "Returning to home position (R to Resume)")

    def _on_resume(self) -> None:
        if self.mode not in (Mode.STOP, Mode.HOME):
            return
        self._set_mode(Mode.TELEOP, "Resuming teleop")

    def _on_toggle(self) -> None:
        if self.mode is Mode.TELEOP:
            self._start_grasp()
        elif self.mode is Mode.IBVS:
            self._return_to_teleop("IBVS interrupted by operator")
        else:
            print(f"[INFO][IBVS-TELEOP] Ignoring {KEY_TOGGLE} while {self.mode.name}; press {KEY_RESUME} first.")

    def _on_save_setpoint(self) -> None:
        if self.mode is Mode.IBVS:
            print("[WARN][IBVS-TELEOP] Not capturing a setpoint during an IBVS grasp.")
            return
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None:
            print("[WARN][IBVS-TELEOP] No wrist frame available to save.")
            return
        rgb, _, K = rgbd_to_numpy(rgbd)
        z = float(tcp_pose_numpy(self.env)[2])
        out = save_setpoint_rgb(self._setpoint_dir / f"wrist_setpoint_z{z:.3f}_{time.strftime('%Y%m%d_%H%M%S')}.png", rgb)
        print(f"[INFO][IBVS-TELEOP] Saved wrist setpoint -> {out} (tcp_z={z:.3f})")
        try:
            self.controller = self._controller_factory(rgb, K)
            print("[INFO][IBVS-TELEOP] New setpoint is active.")
        except ValueError as exc:
            print(f"[WARN][IBVS-TELEOP] Saved, but not using it as the setpoint: {exc}")


    def _start_grasp(self) -> None:
        if self.controller is None:
            print(f"[WARN][IBVS-TELEOP] No setpoint loaded; press {KEY_SAVE} to capture one first.")
            return
        self.controller.reset_phase()
        self._set_mode(Mode.IBVS, "Starting IBVS grasp")

    def _return_to_teleop(self, reason: str) -> None:
        self._sync_gripper_from_controller()
        self._set_mode(Mode.TELEOP, f"{reason}; returning control to teleop")

    def _sync_gripper_from_controller(self) -> None:
        """Make the keyboard's gripper state match what IBVS last commanded."""
        if self.controller is None:
            return
        closed = float(self.controller.gripper_cmd) >= 0.5
        self.teleop._close_gripper = closed
        self.teleop._gripper_value = 1.0 if closed else 0.0

    def _set_mode(self, mode: Mode, message: str) -> None:
        print(f"[INFO][IBVS-TELEOP] {message} ({self.mode.name} -> {mode.name})")
        self.mode = mode

    @staticmethod
    def _help_text() -> str:
        return (
            "IBVS teleop keys:\n"
            f"\t{KEY_TOGGLE}: start IBVS grasp / interrupt grasp\n"
            f"\t{KEY_STOP}: stop robot\n"
            f"\t{KEY_HOME}: home robot\n"
            f"\t{KEY_RESUME}: resume teleop\n"
            f"\t{KEY_SAVE}: save wrist frame as new setpoint\n"
            f"\t{KEY_QUIT}: quit"
        )


def _wait_for_wrist_frame(env, timeout: float) -> dict:
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
        "use_tabletop_camera": args_cli.tabletop_camera,
        "dt": args_cli.dt,
    }
    env = create_env(args_cli.task, env_cfg)
    env.reset()

    def make_controller(setpoint_rgb: np.ndarray, K: np.ndarray) -> IBVSController:
        return build_controller(args_cli, setpoint_rgb, K, dt=env.step_dt)

    controller = None
    if args_cli.setpoint is not None:
        _, _, K = rgbd_to_numpy(_wait_for_wrist_frame(env, args_cli.first_frame_timeout))
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
        view_tabletop=args_cli.tabletop_camera,
        view_wrist=args_cli.wrist_camera,
    ).run()


if __name__ == "__main__":
    main()
