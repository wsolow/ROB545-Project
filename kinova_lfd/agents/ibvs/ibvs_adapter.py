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
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.le_robot.base import to_numpy
from kinova_lfd.teleop.devices import DeviceBase
from kinova_lfd.teleop.factory import teleop_device_name


class Mode(Enum):
    """Who is commanding the robot."""

    TELEOP = auto()
    IBVS = auto()
    STOP = auto()
    HOME = auto()
    QUIT = auto()


ACTION_TOGGLE = "toggle"
ACTION_STOP = "stop"
ACTION_HOME = "home"
ACTION_RESUME = "resume"
ACTION_SAVE = "save"
ACTION_QUIT = "quit"

# Pending actions are resolved in this order when several arrive in one step.
_ACTION_PRIORITY = (ACTION_QUIT, ACTION_STOP, ACTION_HOME, ACTION_RESUME, ACTION_TOGGLE, ACTION_SAVE)

ACTION_DESCRIPTIONS = {
    ACTION_TOGGLE: "Start IBVS grasp / interrupt grasp",
    ACTION_STOP: "Stop robot",
    ACTION_HOME: "Home robot",
    ACTION_RESUME: "Resume teleop",
    ACTION_SAVE: "Save wrist frame as new setpoint",
    ACTION_QUIT: "Quit",
}

# Teleop device name (see ``teleop_device_name``) -> action -> key / button.
BINDINGS: dict[str, dict[str, str]] = {
    "keyboard": {
        ACTION_TOGGLE: "G",
        ACTION_STOP: "X",
        ACTION_HOME: "H",
        ACTION_RESUME: "R",
        ACTION_SAVE: "B",
        ACTION_QUIT: "Z",
    },
    "xbox": {
        ACTION_TOGGLE: "x",
        ACTION_STOP: "b",
        ACTION_HOME: "a",
        ACTION_RESUME: "start",
        ACTION_SAVE: "y",
        ACTION_QUIT: "back",
    },
}


class IbvsTeleopAdapter:
    """Viewer policy that switches between teleop and an IBVS grasp.

    Mode keys / buttons come from :data:`BINDINGS` for the teleop device in use. Presses
    arrive on the device's input thread and are only latched there. All mode
    changes happen at the start of :meth:`__call__` on the control thread, so the
    controller and teleop device state are never modified mid-step.

    Outside of IBVS the teleop device's gripper value is the gripper command. When IBVS
    hands control back (grasp done or interrupted) the controller's gripper command is
    copied into the device so the gripper does not snap open and drop the block.
    """

    def __init__(
        self,
        env,
        teleop: DeviceBase,
        controller_factory: Callable[[np.ndarray, np.ndarray], IBVSController],
        controller: IBVSController | None = None,
        setpoint_dir: str | Path = "exp/ibvs_setpoints",
        home_tcp: np.ndarray = HOME_TCP_CART,
    ) -> None:
        """Initialize the adapter.

        Args:
            env: environment.
            teleop: Teleop device (keyboard, Xbox, ...). Devices without an entry in
                :data:`BINDINGS` (e.g. VR) can move the arm but have no mode buttons.
            controller_factory: ``(setpoint_rgb, K) -> IBVSController``, used when a new
                setpoint is captured (the save button).
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

        device_name = teleop_device_name(teleop)
        self._bindings = BINDINGS.get(device_name, {})
        if not self._bindings:
            print(f"[WARN][IBVS-TELEOP] No IBVS bindings for teleop device {device_name!r}; mode buttons disabled.")
        for action, button in self._bindings.items():
            self.teleop.add_callback(button, self._make_latch(action))
        if self._bindings and hasattr(self.teleop, "button_bindings"):
            # Show this adapter's meaning of each button in the device's startup banner.
            self.teleop.button_bindings = {
                button: ACTION_DESCRIPTIONS[action] for action, button in self._bindings.items()
            }

    def start(self) -> None:
        """Start the teleop device listener."""
        self.teleop.reset()
        self.teleop.start()
        print(self._help_text())

    def stop(self) -> None:
        """Stop the teleop device listener."""
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
            return f"TELEOP (no setpoint: press {self._button(ACTION_SAVE)})"
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

    def _teleop_action(self) -> tuple[np.ndarray, ActionSpec]:
        # MuJoCo envs return a (1, 7) device tensor; Kortex returns a (7,) numpy array.
        action = to_numpy(self.teleop.advance(self.env._robot_tool_pose_b)).astype(np.float64).reshape(-1)
        return action, ActionSpec.TWIST_TCP

    def _ibvs_action(self) -> tuple[np.ndarray, ActionSpec]:
        assert self.controller is not None
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.controller.gripper_cmd]), ActionSpec.TWIST_TCP

        rgb, depth, _ = rgbd_to_numpy(rgbd)
        pose = self.env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        # KortexEnv returns [x, y, z, qx, qy, qz, qw], which is what IBVSController expects.
        tcp_pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:7].copy()
        twist = np.asarray(self.controller(rgb, depth, tcp_pose=tcp_pose, dt=self.env.step_dt), dtype=np.float64)

        if self.controller.phase is GraspPhase.DONE:
            self._return_to_teleop("Grasp complete")

        return twist, ActionSpec.TWIST_TCP

    @property
    def _gripper(self) -> float:
        """Gripper command used outside of IBVS (owned by the teleop device)."""
        return float(self.teleop._gripper_value)

    def _hold_action(self) -> tuple[np.ndarray, ActionSpec]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper]), ActionSpec.TWIST_TCP

    def _home_action(self) -> tuple[np.ndarray, ActionSpec]:
        action = self._home_tcp.copy()
        action[-1] = self._gripper
        return action, ActionSpec.MOVE_TCP

    def _button(self, action: str) -> str:
        """Key / button bound to ``action`` on the current device (for messages)."""
        return self._bindings.get(action, "<unbound>")

    def _make_latch(self, action: str) -> Callable[[], None]:
        def _latch() -> None:
            with self._pending_lock:
                self._pending.add(action)

        return _latch

    def _resolve_keys(self) -> None:
        """Consume latched button presses on the control thread."""
        with self._pending_lock:
            pending, self._pending = self._pending, set()
        if not pending:
            return
        handlers = {
            ACTION_QUIT: self._on_quit,
            ACTION_STOP: self._on_stop,
            ACTION_HOME: self._on_home,
            ACTION_RESUME: self._on_resume,
            ACTION_TOGGLE: self._on_toggle,
            ACTION_SAVE: self._on_save_setpoint,
        }
        for action in _ACTION_PRIORITY:
            if action in pending:
                handlers[action]()

    def _on_quit(self) -> None:
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.QUIT, "Quitting")

    def _on_stop(self) -> None:
        if self.mode not in (Mode.TELEOP, Mode.IBVS, Mode.HOME):
            return
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.STOP, f"Stopping the robot ({self._button(ACTION_RESUME)} to Resume)")

    def _on_home(self) -> None:
        if self.mode not in (Mode.TELEOP, Mode.IBVS, Mode.STOP):
            return
        if self.mode is Mode.IBVS:
            self._sync_gripper_from_controller()
        self._set_mode(Mode.HOME, f"Returning to home position ({self._button(ACTION_RESUME)} to Resume)")

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
            print(
                f"[INFO][IBVS-TELEOP] Ignoring {self._button(ACTION_TOGGLE)} while {self.mode.name}; "
                f"press {self._button(ACTION_RESUME)} first."
            )

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
            print(f"[WARN][IBVS-TELEOP] No setpoint loaded; press {self._button(ACTION_SAVE)} to capture one first.")
            return
        self.controller.reset_phase()

        self._set_mode(Mode.IBVS, "Starting IBVS grasp")

    def _return_to_teleop(self, reason: str) -> None:
        self._sync_gripper_from_controller()
        self._set_mode(Mode.TELEOP, f"{reason}; returning control to teleop")

    def _sync_gripper_from_controller(self) -> None:
        """Make the teleop device's gripper state match what IBVS last commanded."""
        if self.controller is None:
            return
        closed = float(self.controller.gripper_cmd) >= 0.5
        self.teleop._close_gripper = closed
        self.teleop._gripper_value = 1.0 if closed else 0.0

    def _set_mode(self, mode: Mode, message: str) -> None:
        print(f"[INFO][IBVS-TELEOP] {message} ({self.mode.name} -> {mode.name})")
        self.mode = mode

    def _help_text(self) -> str:
        if not self._bindings:
            return "IBVS teleop: no mode buttons bound for this device."
        lines = [f"\t{button}: {ACTION_DESCRIPTIONS[action]}" for action, button in self._bindings.items()]
        return "IBVS teleop controls:\n" + "\n".join(lines)
