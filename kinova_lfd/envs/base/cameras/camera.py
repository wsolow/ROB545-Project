"""camera.py.

A camera-only ``GymEnv``: ``CameraEnv`` wraps one camera (RealSense or Kortex wrist camera),
picked by the ``name`` field of its config through ``make_camera``.

Written by Will Solow, 2026.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv

if TYPE_CHECKING:
    from kinova_lfd.envs.base.cameras.kortex_wrist_camera import WristCamera
    from kinova_lfd.envs.base.cameras.realsense_camera import RealsenseCamera

CAMERAS = ("realsense", "wrist")


@dataclass
class CameraCfg:
    """Settings shared by every camera."""

    name: str = ""
    """Camera to create, one of ``CAMERAS``."""
    first_frame_timeout: float = 10.0
    """Seconds ``CameraEnv.reset`` waits for the first frame before raising."""


@dataclass
class RealsenseCameraCfg(CameraCfg):
    """Settings for ``RealsenseCamera``."""

    name: str = "realsense"
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class WristCameraCfg(CameraCfg):
    """Settings for the Kortex ``WristCamera``."""

    name: str = "wrist"
    robot_ip: str = "192.168.1.10"
    rate: int = 20
    username: str = "admin"
    password: str = "admin"


def make_camera(cfg: CameraCfg, router: Any | None = None) -> "RealsenseCamera | WristCamera":
    """Construct the camera named by ``cfg.name``.

    Camera modules are imported lazily, so the wrist camera does not require ``pyrealsense2``
    and the RealSense camera does not require ``kortex_api``.

    Args:
        cfg: Camera config. ``cfg.name`` picks the camera.
        router: Optional open Kortex TCP ``RouterClient``, used only by the wrist camera to read
            its intrinsics. When ``None`` the wrist camera opens a short-lived TCP session.

    """
    name = cfg.name.lower()
    if name == "realsense":
        from kinova_lfd.envs.base.cameras.realsense_camera import RealsenseCamera

        assert isinstance(cfg, RealsenseCameraCfg), f"Camera 'realsense' needs a RealsenseCameraCfg, got {type(cfg)}"
        return RealsenseCamera(width=cfg.width, height=cfg.height, fps=cfg.fps)
    if name == "wrist":
        from kinova_lfd.envs.base.cameras.kortex_wrist_camera import WristCamera

        assert isinstance(cfg, WristCameraCfg), f"Camera 'wrist' needs a WristCameraCfg, got {type(cfg)}"
        return WristCamera(
            robot_ip=cfg.robot_ip,
            rate=cfg.rate,
            router=router,
            username=cfg.username,
            password=cfg.password,
        )
    raise ValueError(f"Unsupported camera {cfg.name!r}. Use one of {CAMERAS}.")


class CameraEnv(GymEnv):
    """Environment that holds a single camera and returns its latest frame on every step.

    Observations are the camera's RGB-D dictionary:
      - ``rgb``: (3, H, W) uint8 RGB image
      - ``depth``: (H, W) uint16 depth image, or ``None`` for the wrist camera
      - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
      - ``camera_pose``: ``None`` (not available on hardware)
      - ``timestamp``: float timestamp in seconds
    """

    def __init__(self, cfg: CameraCfg, router: Any | None = None) -> None:
        """Create the camera described by ``cfg``. Streaming starts on ``reset``.

        Args:
            cfg: Camera config passed to ``make_camera``.
            router: Optional open Kortex TCP ``RouterClient`` for the wrist camera.

        """
        self.cfg = cfg
        self.camera = make_camera(cfg, router=router)

    def reset(self) -> dict[str, Any]:
        """Start streaming if needed, wait for the first frame, and return it."""
        self.camera.start()
        self.camera.wait_for_frame(self.cfg.first_frame_timeout)
        obs = self.camera._get_latest_rgbd_raw()
        assert obs is not None, "Camera reported a first frame but returned none."
        return obs

    def step(
        self, action: torch.Tensor | np.ndarray | None = None, action_spec: ActionSpec | None = None
    ) -> dict[str, Any] | None:
        """Return the latest camera frame without blocking. ``action`` is ignored.

        Returns ``None`` if no frame has arrived yet (``reset`` was not called).
        """
        return self.camera._get_latest_rgbd_raw()

    def _get_latest_rgbd(self) -> dict[str, Any] | None:
        """Return the latest camera frame without blocking."""
        return self.camera._get_latest_rgbd_raw()

    def close(self) -> None:
        """Stop the camera stream."""
        self.camera.stop()
