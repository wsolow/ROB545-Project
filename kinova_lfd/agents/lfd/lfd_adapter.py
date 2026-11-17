import threading
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import numpy as np
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.le_robot.base import TWIST_TCP_KEYS, to_numpy
from kinova_lfd.imitation.realtime import RealtimeDiffusionController
from kinova_lfd.teleop.devices import DeviceBase
from kinova_lfd.teleop.factory import teleop_device_name


class Mode(Enum):
    """Who is commanding the robot."""

    TELEOP = auto()
    POLICY = auto()
    STOP = auto()
    QUIT = auto()


ACTION_TOGGLE = "toggle"
ACTION_STOP = "stop"
ACTION_RESUME = "resume"
ACTION_QUIT = "quit"

# Pending actions are resolved in this order when several arrive in one step.
_ACTION_PRIORITY = (ACTION_QUIT, ACTION_STOP, ACTION_RESUME, ACTION_TOGGLE)

ACTION_DESCRIPTIONS = {
    ACTION_TOGGLE: "Start diffusion policy / interrupt policy",
    ACTION_STOP: "Stop robot",
    ACTION_RESUME: "Resume teleop",
    ACTION_QUIT: "Quit",
}

# Teleop device name (see ``teleop_device_name``) -> action -> key / button.
BINDINGS: dict[str, dict[str, str]] = {
    "keyboard": {ACTION_TOGGLE: "G", ACTION_STOP: "X", ACTION_RESUME: "R", ACTION_QUIT: "Z"},
    "xbox": {ACTION_TOGGLE: "x", ACTION_STOP: "b", ACTION_RESUME: "start", ACTION_QUIT: "back"},
}


class DiffusionTeleopAdapter:
    """Viewer policy that switches between teleop and a diffusion policy.

    Mode keys / buttons come from :data:`BINDINGS` for the teleop device in use. Presses
    arrive on the device's input thread and are only latched there. All mode
    changes happen at the start of :meth:`__call__` on the control thread, so the
    diffusion controller and teleop device state are never modified mid-step.

    Only TCP twists are sent to the robot: teleop, stop, and the policy all return
    ``ActionSpec.TWIST_TCP``. The policy observation comes from the LeRobot robot
    (``robot.get_observation()``), matching how the training data was recorded.

    Outside of POLICY the teleop device's gripper value is the gripper command. When the
    policy hands control back, its last gripper command is copied into the device so
    the gripper does not snap open and drop the block.
    """

    def __init__(
        self,
        robot: Robot,
        teleop: DeviceBase,
        controller: RealtimeDiffusionController,
        preprocessor: PolicyProcessorPipeline,
        ds_features: dict[str, Any],
        max_policy_steps: int = 0,
        done_fn: Callable[[Any], bool] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            robot: Connected LeRobot Gen3 robot (``MjGen3Robot`` / ``KortexTcpGen3Robot``)
                configured with ``action_spec="TWIST_TCP"``.
            teleop: Teleop device (keyboard, Xbox, ...). Devices without an entry in
                :data:`BINDINGS` (e.g. VR) can move the arm but have no mode buttons.
            controller: Realtime diffusion controller (with postprocessor) producing robot-unit actions.
            preprocessor: Policy preprocessor pipeline.
            ds_features: Dataset features filtered to the policy's inputs.
            max_policy_steps: Hand back to teleop after this many policy steps (0 = no limit).
            done_fn: Optional ``done_fn(env) -> bool``; hands back to teleop when it returns True.

        """
        action_keys = tuple(robot.action_features)
        if action_keys != TWIST_TCP_KEYS:
            raise ValueError(f"DiffusionTeleopAdapter needs a TWIST_TCP robot; got action keys {list(action_keys)}.")
        policy_keys = tuple(ds_features["action"]["names"])
        if policy_keys != TWIST_TCP_KEYS:
            raise ValueError(f"DiffusionTeleopAdapter needs a TWIST_TCP policy; got action names {list(policy_keys)}.")

        self.robot = robot
        self.env = robot.env
        self.teleop = teleop
        self.controller = controller
        self.preprocessor = preprocessor
        self.ds_features = ds_features
        self.max_policy_steps = int(max_policy_steps)
        self.done_fn = done_fn

        self.mode = Mode.TELEOP
        self._policy_steps = 0
        self._policy_gripper = 0.0
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()

        device_name = teleop_device_name(teleop)
        self._bindings = BINDINGS.get(device_name, {})
        if not self._bindings:
            print(f"[WARN][LFD-TELEOP] No diffusion bindings for teleop device {device_name!r}; mode buttons disabled.")
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
        """Stop the teleop device listener and any in-flight diffusion inference."""
        self.teleop.stop()
        self.controller.reset()

    @property
    def should_quit(self) -> bool:
        """Whether the operator pressed quit."""
        return self.mode is Mode.QUIT

    @property
    def status_text(self) -> str:
        """Short mode description drawn on the camera previews."""
        if self.mode is Mode.POLICY:
            limit = f"/{self.max_policy_steps}" if self.max_policy_steps > 0 else ""
            return f"POLICY step {self._policy_steps}{limit} (replans {self.controller._replan_count})"
        return self.mode.name

    def __call__(self, observation) -> tuple[np.ndarray, ActionSpec]:
        """Resolve pending keys, then return the twist for the current mode."""
        del observation  # Teleop uses the TCP pose; the policy reads the LeRobot robot observation.
        self._resolve_keys()

        if self.mode is Mode.POLICY:
            return self._policy_action()
        if self.mode is Mode.TELEOP:
            return self._teleop_action()
        return self._hold_action()

    def _teleop_action(self) -> tuple[np.ndarray, ActionSpec]:
        # MuJoCo envs return a (1, 7) device tensor; Kortex returns a (7,) numpy array.
        action = to_numpy(self.teleop.advance(self.env._robot_tool_pose_b)).astype(np.float64).reshape(-1)
        return action, ActionSpec.TWIST_TCP

    def _policy_action(self) -> tuple[np.ndarray, ActionSpec]:
        obs = self.robot.get_observation()
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=self.ds_features,
            device=self.controller.policy.config.device,
            robot_type=self.robot.name,
        )
        action = self.controller.step(self.preprocessor(obs_frame))
        robot_action = make_robot_action(action, self.ds_features)
        twist = np.asarray([robot_action[key] for key in TWIST_TCP_KEYS], dtype=np.float64)
        self._policy_gripper = float(twist[-1])
        self._policy_steps += 1

        if self.max_policy_steps > 0 and self._policy_steps >= self.max_policy_steps:
            self._return_to_teleop(f"Reached {self.max_policy_steps} policy steps")
        elif self.done_fn is not None and self.done_fn(self.env):
            self._return_to_teleop("Policy task complete")

        return twist, ActionSpec.TWIST_TCP

    @property
    def _gripper(self) -> float:
        """Gripper command used outside of POLICY (owned by the teleop device)."""
        return float(self.teleop._gripper_value)

    def _hold_action(self) -> tuple[np.ndarray, ActionSpec]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper]), ActionSpec.TWIST_TCP

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
            ACTION_RESUME: self._on_resume,
            ACTION_TOGGLE: self._on_toggle,
        }
        for action in _ACTION_PRIORITY:
            if action in pending:
                handlers[action]()

    def _on_quit(self) -> None:
        if self.mode is Mode.POLICY:
            self._leave_policy()
        self._set_mode(Mode.QUIT, "Quitting")

    def _on_stop(self) -> None:
        if self.mode not in (Mode.TELEOP, Mode.POLICY):
            return
        if self.mode is Mode.POLICY:
            self._leave_policy()
        self._set_mode(Mode.STOP, f"Stopping the robot ({self._button(ACTION_RESUME)} to Resume)")

    def _on_resume(self) -> None:
        if self.mode is not Mode.STOP:
            return
        self._set_mode(Mode.TELEOP, "Resuming teleop")

    def _on_toggle(self) -> None:
        if self.mode is Mode.TELEOP:
            self._start_policy()
        elif self.mode is Mode.POLICY:
            self._return_to_teleop("Policy interrupted by operator")
        else:
            print(
                f"[INFO][LFD-TELEOP] Ignoring {self._button(ACTION_TOGGLE)} while {self.mode.name}; "
                f"press {self._button(ACTION_RESUME)} first."
            )

    def _start_policy(self) -> None:
        # Fresh observation history; the first policy step runs a blocking warmup inference.
        self.controller.reset()
        self._policy_steps = 0
        self._policy_gripper = self._gripper
        self._set_mode(Mode.POLICY, "Starting diffusion policy")

    def _return_to_teleop(self, reason: str) -> None:
        self._leave_policy()
        self._set_mode(Mode.TELEOP, f"{reason}; returning control to teleop")

    def _leave_policy(self) -> None:
        """Cancel in-flight inference and hand the policy's gripper state to the teleop device."""
        self.controller.reset()
        closed = self._policy_gripper >= 0.5
        self.teleop._close_gripper = closed
        self.teleop._gripper_value = 1.0 if closed else 0.0

    def _set_mode(self, mode: Mode, message: str) -> None:
        print(f"[INFO][LFD-TELEOP] {message} ({self.mode.name} -> {mode.name})")
        self.mode = mode

    def _help_text(self) -> str:
        if not self._bindings:
            return "Diffusion teleop: no mode buttons bound for this device."
        lines = [f"\t{button}: {ACTION_DESCRIPTIONS[action]}" for action, button in self._bindings.items()]
        return "Diffusion teleop controls:\n" + "\n".join(lines)
