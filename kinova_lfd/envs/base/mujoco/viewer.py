import logging
import threading
from collections.abc import Callable
from typing import override

import cv2
import mujoco
import numpy as np
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import (
    PolicyProtocol,
    VerbosityLevel,
)
from mjlab.viewer.model_sync import (
    disable_model_sameframe_shortcuts,
)
from mjlab.viewer.native.keys import KEY_A, KEY_R
from mjlab.viewer.native.viewer import PlotCfg

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.teleop.devices.device_base import DeviceBase

logger = logging.getLogger(__name__)


def _as_preview_image(rgb: object) -> np.ndarray:
    """Convert an RGB tensor/array to an HWC uint8 image for the preview thread."""
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    image = np.asarray(rgb).squeeze()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.0 + 1e-3:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


class MjViewer(NativeMujocoViewer):
    def __init__(
        self,
        env: GymEnv,
        policy: PolicyProtocol | DeviceBase | None = None,
        frame_rate: float = 60.0,
        key_callback: Callable[[int], None] | None = None,
        plot_cfg: PlotCfg | None = None,
        enable_perturbations: bool = True,
        verbosity: VerbosityLevel = VerbosityLevel.SILENT,
        view_wrist: bool = True,
        view_tabletop: bool = True,
        ignore_mjlab_keys: bool = False,
        preview_fps: float = 20.0,
    ):
        super().__init__(env, policy, frame_rate, key_callback, plot_cfg, enable_perturbations, verbosity)
        if isinstance(self.policy, DeviceBase):
            self.policy.start()
        self._view_wrist = view_wrist
        self._view_tabletop = view_tabletop
        self._ignore_mjlab_keys = ignore_mjlab_keys
        self._rgbd_wrist = None
        self._rgbd_table = None
        self._preview_fps = preview_fps
        self._preview_stop = threading.Event()
        self._preview_lock = threading.Lock()
        self._preview_thread: threading.Thread | None = None
        self._viewer_closed = False

    @override
    def setup(self) -> None:
        """Set up MuJoCo viewer resources."""
        sim = self.env.unwrapped.sim
        self.mjm = sim.mj_model
        self.mjd = sim.mj_data
        assert self.mjm is not None
        disable_model_sameframe_shortcuts(self.mjm)
        if self.cfg.fovy is not None:
            self.mjm.vis.global_.fovy = self.cfg.fovy

        if self.env.unwrapped.num_envs > 1:
            self.vd = mujoco.MjData(self.mjm)

        self.pert = mujoco.MjvPerturb() if self.enable_perturbations else None
        self.vopt = mujoco.MjvOption()

        assert self.mjm is not None
        assert self.mjd is not None
        self.viewer = mujoco.viewer.launch_passive(
            self.mjm,
            self.mjd,
            key_callback=self._safe_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        if self.viewer is None:
            raise RuntimeError("Failed to launch MuJoCo viewer")

        if not self.cfg.enable_shadows:
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 1

        self._setup_camera()

        if self.enable_perturbations:
            self.log("[INFO] Interactive perturbations enabled", VerbosityLevel.INFO)

        if (self._view_wrist or self._view_tabletop) and self._preview_thread is None:
            self._preview_stop.clear()
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                name="gen3-camera-preview",
                daemon=True,
            )
            self._preview_thread.start()

    @override
    def _safe_key_callback(self, key: int) -> None:
        # During recording, skip mjlab Enter/P/Space/etc. so they do not reset
        # or toggle plots. Simulate built-in letter hotkeys still cannot be suppressed.
        if self._ignore_mjlab_keys:
            return
        # A and R move the arm. The base viewer also binds them to debug toggles.
        if isinstance(self.policy, DeviceBase) and key in (KEY_A, KEY_R):
            return
        super()._safe_key_callback(key)

    def open_display(self) -> None:
        """Launch the passive MuJoCo window for an external control loop.

        Use this from LeRobot recording / custom loops that own ``env.step``
        themselves. Call :meth:`sync_display` after each step, then
        :meth:`close_display` when finished. Do **not** call :meth:`run`.
        """
        self.setup()

    def sync_display(self) -> bool:
        """Submit camera previews and refresh the passive viewer without waiting for GUI work."""
        if self.viewer is None or not self.is_running():
            self._viewer_closed = True
            self._close_camera_preview()
            return False
        self._update_camera_previews()
        self.sync_env_to_viewer()
        return True

    def close_display(self) -> None:
        """Close the passive viewer and terminate the camera preview thread."""
        try:
            self._close_camera_preview()
        finally:
            self.close()

    @property
    def viewer_closed(self) -> bool:
        """Whether the MuJoCo window was closed by the user."""
        return self._viewer_closed

    def _close_camera_preview(self) -> None:
        """Stop the camera preview thread; safe to call repeatedly."""
        self._preview_stop.set()
        thread = self._preview_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning("Camera preview thread did not stop within the shutdown timeout")
            else:
                self._preview_thread = None
        else:
            self._preview_thread = None

        with self._preview_lock:
            self._rgbd_wrist = None
            self._rgbd_table = None

    def _update_camera_previews(self) -> None:
        """Publish the latest RGB-D references for the preview thread."""
        rgbd_wrist = None
        rgbd_table = None
        if self._view_wrist:
            rgbd_wrist = self.env.unwrapped._get_latest_rgbd_wrist()

        if self._view_tabletop:
            rgbd_table = self.env.unwrapped._get_latest_rgbd()

        with self._preview_lock:
            if self._view_wrist:
                self._rgbd_wrist = rgbd_wrist
            if self._view_tabletop:
                self._rgbd_table = rgbd_table

    def _preview_loop(self) -> None:
        """Display the latest RGB-D frames on a dedicated OpenCV thread."""
        try:
            if self._view_wrist:
                cv2.namedWindow("RGB Wrist", cv2.WINDOW_AUTOSIZE)
            if self._view_tabletop:
                cv2.namedWindow("RGB Tabletop", cv2.WINDOW_AUTOSIZE)

            frame_period = 1.0 / self._preview_fps if self._preview_fps > 0 else 0.0
            while True:
                if self._preview_stop.is_set():
                    break

                with self._preview_lock:
                    rgbd_wrist = self._rgbd_wrist
                    rgbd_table = self._rgbd_table

                try:
                    if self._view_wrist and rgbd_wrist is not None and "rgb" in rgbd_wrist:
                        wrist = _as_preview_image(rgbd_wrist["rgb"])
                        cv2.imshow("RGB Wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
                    if self._view_tabletop and rgbd_table is not None and "rgb" in rgbd_table:
                        tabletop = _as_preview_image(rgbd_table["rgb"])
                        cv2.imshow("RGB Tabletop", cv2.cvtColor(tabletop, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                except cv2.error as exc:
                    logger.warning("Camera preview stopped because OpenCV GUI failed: %s", exc)
                    break

                if frame_period > 0:
                    self._preview_stop.wait(frame_period)
        except Exception:
            logger.exception("Camera preview thread failed")
        finally:
            if self._view_wrist or self._view_tabletop:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

    @override
    def _execute_step(self) -> bool:
        """Run one obs/policy/step cycle.

        Returns True on success, False if step failed.
        """
        with torch.no_grad():
            obs = self.env.get_observations()
            self._update_camera_previews()
            if self.policy is not None:
                if isinstance(self.policy, DeviceBase):
                    actions = self.policy(self.env._robot_tool_pose_b)
                    action_spec = ActionSpec.TWIST_TCP
                else:
                    actions = self.policy(obs)
                    action_spec = ActionSpec.TWIST_TCP
            else:
                actions = torch.zeros(self.env.action_space.shape, device=self.env.device)
                action_spec = ActionSpec.JOINTS_POS
            self.env.step(actions, action_spec=action_spec)
            self._step_count += 1
            self._stats_steps += 1
            return True
