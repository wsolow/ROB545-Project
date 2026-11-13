import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch

from kinova_lfd.agents.ibvs.util import rgbd_to_numpy, save_setpoint_rgb
from kinova_lfd.agents.moderator import HOME_TCP_CART
from kinova_lfd.controllers import IBVSController
from kinova_lfd.controllers.ibvs import GraspPhase, extract_block_corners_px
from kinova_lfd.core.np_math import np_convert_quat
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.teleop.devices import Se3Keyboard


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


class IbvsTeleopAdapter:
    """Viewer policy that switches between keyboard teleop and an IBVS grasp.

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
            env: environment.
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
        pose = self.env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        tcp_pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:7]
        tcp_pose[3:7] = np_convert_quat(tcp_pose[3:7], to="wxyz")
        twist = np.asarray(self.controller(rgb, depth, tcp_pose=tcp_pose, dt=self.env.step_dt), dtype=np.float64)

        if self.controller.phase is GraspPhase.DONE:
            self._return_to_teleop("Grasp complete")

        return twist, ActionSpec.TWIST_TCP

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
        pose = self.env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        z = float(np.asarray(pose, dtype=np.float64).reshape(-1)[:7][2])
        out = save_setpoint_rgb(
            self._setpoint_dir / f"wrist_setpoint_z{z:.3f}_{time.strftime('%Y%m%d_%H%M%S')}.png", rgb
        )
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
