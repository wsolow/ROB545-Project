"""Shared LeRobot Robot base wrapping KinovaLfD Gen3 gym environments."""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import Any

import numpy as np
import torch

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.envs.factory import create_env
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")
GRIPPER_NAME = "gripper"
EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz")

TWIST_TCP_KEYS = ("tcp.vx", "tcp.vy", "tcp.vz", "tcp.wx", "tcp.wy", "tcp.wz", f"{GRIPPER_NAME}.pos")
MOVE_TCP_KEYS = ("tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz", f"{GRIPPER_NAME}.pos")
JOINT_ACTION_KEYS = tuple(f"{name}.pos" for name in (*ARM_JOINT_NAMES, GRIPPER_NAME))


def parse_action_spec(action_spec: str | ActionSpec) -> ActionSpec:
    """Resolve a config string or enum into an ``ActionSpec``."""
    if isinstance(action_spec, ActionSpec):
        return action_spec
    try:
        return ActionSpec[str(action_spec).upper()]
    except KeyError as exc:
        valid = ", ".join(spec.name for spec in ActionSpec)
        raise ValueError(f"Unknown action_spec={action_spec!r}. Expected one of: {valid}") from exc


def action_keys_for_spec(action_spec: ActionSpec) -> tuple[str, ...]:
    """Flat ``RobotAction`` keys for the given ``ActionSpec``."""
    if action_spec == ActionSpec.TWIST_TCP:
        return TWIST_TCP_KEYS
    if action_spec == ActionSpec.MOVE_TCP:
        return MOVE_TCP_KEYS
    if action_spec in (ActionSpec.JOINTS_POS, ActionSpec.JOINTS_VEL):
        return JOINT_ACTION_KEYS
    raise ValueError(
        f"ActionSpec.{action_spec.name} is not supported by Gen3LfDRobot. "
        "Use TWIST_TCP, MOVE_TCP, JOINTS_POS, or JOINTS_VEL."
    )


def action_dict_to_array(action: RobotAction, action_spec: ActionSpec) -> np.ndarray:
    """Pack a flat LeRobot action dict into a 1-D numpy vector."""
    keys = action_keys_for_spec(action_spec)
    missing = [key for key in keys if key not in action]
    if missing:
        raise KeyError(f"RobotAction missing keys for ActionSpec.{action_spec.name}: {missing}")
    return np.asarray([float(action[key]) for key in keys], dtype=np.float32)


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch / numpy values (possibly CUDA) to a host ``ndarray``."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rgb_to_hwc_uint8(rgbd_or_rgb: dict[str, Any] | np.ndarray | torch.Tensor | None) -> np.ndarray | None:
    """Convert env RGB / RGBD payloads to LeRobot ``(H, W, 3)`` uint8."""
    if rgbd_or_rgb is None:
        return None

    rgb: Any = rgbd_or_rgb["rgb"] if isinstance(rgbd_or_rgb, dict) else rgbd_or_rgb
    if rgb is None:
        return None

    rgb = to_numpy(rgb)
    # Drop leading batch dim when present: (1, 3, H, W) or (1, H, W, 3).
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

    if np.issubdtype(rgb.dtype, np.floating):
        max_val = float(np.nanmax(rgb)) if rgb.size else 1.0
        if max_val <= 1.0 + 1e-3:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


class Gen3LfDRobot(Robot):
    """LeRobot ``Robot`` adapter around a KinovaLfD Gen3 ``GymEnv``."""

    task_name: str

    def __init__(self, config: RobotConfig):
        super().__init__(config)
        self.config = config
        self.env: GymEnv | None = None
        self._is_connected = False
        self._action_spec = parse_action_spec(getattr(config, "action_spec", ActionSpec.TWIST_TCP))

    @abstractmethod
    def _env_cfg_dict(self) -> dict[str, Any]:
        """Keyword args forwarded to ``create_env`` for this backend."""

    @property
    def _joint_names(self) -> tuple[str, ...]:
        return (*ARM_JOINT_NAMES, GRIPPER_NAME)

    @property
    def _action_keys(self) -> tuple[str, ...]:
        return action_keys_for_spec(self._action_spec)

    @property
    def _motors_ft(self) -> dict[str, type]:
        if not getattr(self.config, "include_joint_positions", True):
            return {}
        return {f"{name}.pos": float for name in self._joint_names}

    @property
    def _ee_ft(self) -> dict[str, type]:
        if not getattr(self.config, "include_ee_pose", True):
            return {}
        return {key: float for key in EE_POSE_KEYS}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        height = int(getattr(self.config, "camera_height", 480))
        width = int(getattr(self.config, "camera_width", 640))
        features: dict[str, tuple[int, int, int]] = {}
        if getattr(self.config, "use_wrist_camera", False):
            features["wrist"] = (height, width, 3)
        if getattr(self.config, "use_tabletop_camera", False):
            features["tabletop"] = (height, width, 3)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._ee_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._action_keys}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.env is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Kortex / MuJoCo Gen3 wrappers do not use LeRobot motor calibration."""

    def configure(self) -> None:
        """No one-time hardware configuration beyond env construction."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # Not applicable for Gen3 gym backends.
        self.env = create_env(self.task_name, self._env_cfg_dict())
        self._post_create_env(self.env)
        self.env.reset()
        self._is_connected = True

    def _post_create_env(self, env: GymEnv) -> None:
        """Optional backend-specific tweaks after ``create_env``."""

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        assert self.env is not None
        obs: RobotObservation = {}

        if getattr(self.config, "include_joint_positions", True):
            joints = to_numpy(self.env._joint_positions).astype(np.float32, copy=False).reshape(-1)
            # MuJoCo may expose more DoFs than the Gen3 arm+gripper; take the first 8.
            joints = joints[: len(self._joint_names)]
            for name, value in zip(self._joint_names, joints, strict=False):
                obs[f"{name}.pos"] = float(value)

        if getattr(self.config, "include_ee_pose", True):
            pose = to_numpy(self.env._robot_tool_pose_b).astype(np.float32, copy=False).reshape(-1)
            for key, value in zip(EE_POSE_KEYS, pose[:7], strict=True):
                obs[key] = float(value)

        if getattr(self.config, "use_wrist_camera", False):
            wrist = rgb_to_hwc_uint8(self.env._get_latest_rgbd_wrist())
            if wrist is not None:
                obs["wrist"] = wrist

        if getattr(self.config, "use_tabletop_camera", False):
            tabletop = rgb_to_hwc_uint8(self.env._get_latest_rgbd())
            if tabletop is not None:
                obs["tabletop"] = tabletop

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        assert self.env is not None
        vector = action_dict_to_array(action, self._action_spec)
        step_action: np.ndarray | torch.Tensor = vector

        # MuJoCo DirectRL envs expect a torch tensor on the sim device.
        if hasattr(self.env, "device"):
            step_action = torch.as_tensor(vector, device=self.env.device, dtype=torch.float32)
            if getattr(self.env, "num_envs", 1) > 0:
                step_action = step_action.unsqueeze(0)

        # mj_gen3 recording may substep so sim time matches wall-clock fps.
        n_steps = max(1, int(getattr(self.config, "physics_steps_per_action", 1)))
        for _ in range(n_steps):
            self.env.step(step_action, action_spec=self._action_spec)
        return {key: float(action[key]) for key in self._action_keys}

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        self.env = None
        self._is_connected = False
