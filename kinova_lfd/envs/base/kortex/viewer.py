"""viewer.py.

OpenCV viewer for Kortex (real-robot) environments. Mirrors the ``MjViewer`` API
(``run`` / ``open_display`` / ``sync_display`` / ``close_display`` / ``viewer_closed``)
but, instead of a MuJoCo window, shows the real wrist and tabletop cameras side by side
in a single OpenCV window.

OpenCV HighGUI has no native split panes (one ``imshow`` image per window), so each
camera gets its own size trackbar and the two scaled images are composed into one canvas.

Written by Will Solow, 2026.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import cv2
import numpy as np

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.teleop.devices.device_base import DeviceBase

logger = logging.getLogger(__name__)

WINDOW_NAME = "Kortex Cameras"
WRIST_TRACKBAR = "Wrist size %"
TABLETOP_TRACKBAR = "Tabletop size %"
_MIN_SIZE_PCT = 20
_MAX_SIZE_PCT = 300
_PANE_GAP_PX = 4


class KortexPolicyProtocol(Protocol):
    """Policy accepted by :class:`KortexViewer`.

    ``__call__`` returns either an action (sent as ``TWIST_TCP``) or an
    ``(action, ActionSpec)`` tuple. The optional attributes below are used when present:

    - ``should_quit: bool``: the viewer exits its loop when this becomes ``True``.
    - ``status_text: str``: drawn on the camera previews.
    - ``annotate_wrist(bgr: np.ndarray) -> np.ndarray``: extra drawing on the wrist preview.
    - ``start()`` / ``stop()``: called when the viewer opens / closes.
    """

    def __call__(self, observation: Any) -> np.ndarray | tuple[np.ndarray, ActionSpec]: ...


def _as_preview_bgr(rgb: Any) -> np.ndarray:
    """Convert an RGB (H, W, 3) or (3, H, W) array to a contiguous BGR uint8 image."""
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
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize ``image`` to ``height`` pixels tall, keeping its aspect ratio."""
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, round(w * height / h))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def _compose_side_by_side(panes: list[np.ndarray], gap: int = _PANE_GAP_PX) -> np.ndarray:
    """Place ``panes`` left to right, top-aligned, on a black canvas."""
    height = max(p.shape[0] for p in panes)
    width = sum(p.shape[1] for p in panes) + gap * (len(panes) - 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for pane in panes:
        canvas[: pane.shape[0], x : x + pane.shape[1]] = pane
        x += pane.shape[1] + gap
    return canvas


def _draw_status(image: np.ndarray, text: str) -> None:
    """Draw ``text`` in the top-left corner with a dark backing box."""
    if not text:
        return
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (5, 5), (15 + w, 15 + h + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 10 + h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


class KortexViewer:
    """Run a policy on a Kortex env and preview the robot cameras in OpenCV windows.

    The env paces itself to ``cfg.dt`` inside ``step``, so the control loop runs at
    the env rate. Camera frames are pulled and drawn on a separate daemon thread so
    the blocking RealSense ``wait_for_frames`` never stalls the control loop. The
    preview thread is the only caller of ``env._get_latest_rgbd()``.
    """

    def __init__(
        self,
        env: GymEnv,
        policy: KortexPolicyProtocol | DeviceBase | None = None,
        view_wrist: bool = True,
        view_tabletop: bool = True,
        preview_fps: float = 20.0,
        preview_height: int = 360,
    ) -> None:
        """Initialize the viewer.

        Args:
            env: Kortex environment. Needs ``cfg.use_wrist_camera`` /
                ``cfg.use_tabletop_camera`` for the corresponding preview.
            policy: Policy or teleop device. ``None`` holds the robot still.
            view_wrist: Show the wrist camera.
            view_tabletop: Show the tabletop RealSense camera.
            preview_fps: Preview refresh rate.
            preview_height: Pane height (px) at 100% on the size trackbars.

        """
        self.env = env
        self.policy = policy
        cfg = getattr(env, "cfg", None)
        self._view_wrist = view_wrist and bool(getattr(cfg, "use_wrist_camera", False))
        self._view_tabletop = view_tabletop and bool(getattr(cfg, "use_tabletop_camera", False))
        if view_wrist and not self._view_wrist:
            logger.warning("Wrist preview requested but env.cfg.use_wrist_camera is False; disabling it.")
        if view_tabletop and not self._view_tabletop:
            logger.warning("Tabletop preview requested but env.cfg.use_tabletop_camera is False; disabling it.")
        self._preview_fps = preview_fps
        self._preview_height = preview_height

        self._status_text = ""
        self._preview_stop = threading.Event()
        self._preview_lock = threading.Lock()
        self._preview_thread: threading.Thread | None = None
        self._viewer_closed = False
        self._opened = False
        self._step_count = 0


    def run(self) -> None:
        """Run the control loop until the policy quits, a window closes, or Ctrl-C."""
        self.open_display()
        try:
            while self.sync_display():
                if bool(getattr(self.policy, "should_quit", False)):
                    print("[INFO][KortexViewer] Policy requested quit.")
                    break
                self._execute_step()
        except KeyboardInterrupt:
            print("[INFO][KortexViewer] Interrupted.")
        finally:
            self._hold_robot()
            self.close_display()
            self.env.close()

    def open_display(self) -> None:
        """Start the policy and the camera preview thread (for external control loops)."""
        if self._opened:
            return
        self._opened = True
        start = getattr(self.policy, "start", None)
        if callable(start):
            start()
        if (self._view_wrist or self._view_tabletop) and self._preview_thread is None:
            self._preview_stop.clear()
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                name="kortex-camera-preview",
                daemon=True,
            )
            self._preview_thread.start()

    def sync_display(self) -> bool:
        """Publish the policy status to the preview thread.

        Returns:
            ``False`` once a preview window has been closed, else ``True``.

        """
        if self._viewer_closed:
            self._close_camera_preview()
            return False
        with self._preview_lock:
            self._status_text = str(getattr(self.policy, "status_text", ""))
        return True

    def close_display(self) -> None:
        """Stop the preview thread and the policy. Safe to call repeatedly."""
        try:
            self._close_camera_preview()
        finally:
            stop = getattr(self.policy, "stop", None)
            if self._opened and callable(stop):
                stop()
            self._opened = False

    @property
    def viewer_closed(self) -> bool:
        """Whether a preview window was closed by the user."""
        return self._viewer_closed


    def _compute_action(self) -> tuple[np.ndarray, ActionSpec]:
        """Query the policy for the next action and its specification."""
        if self.policy is None:
            return np.zeros(7), ActionSpec.TWIST_TCP
        if isinstance(self.policy, DeviceBase):
            return np.asarray(self.policy(self.env._robot_tool_pose_b)), ActionSpec.TWIST_TCP
        out = self.policy(self.env._get_observations())
        if isinstance(out, tuple):
            action, action_spec = out
        else:
            action, action_spec = out, ActionSpec.TWIST_TCP
        return np.asarray(action, dtype=np.float64).reshape(-1), action_spec

    def _execute_step(self) -> bool:
        """Run one obs/policy/step cycle."""
        action, action_spec = self._compute_action()
        self.env.step(action, action_spec=action_spec)
        self._step_count += 1
        return True

    def _hold_robot(self) -> None:
        """Send a single zero twist (gripper unchanged) so the arm stops on exit."""
        try:
            prev = np.asarray(getattr(self.env, "_prev_actions", np.zeros(7))).reshape(-1)
            gripper = float(prev[-1]) if prev.size else 0.0
            self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]), action_spec=ActionSpec.TWIST_TCP)
        except Exception:
            logger.exception("Failed to send the stop command on exit")


    def _close_camera_preview(self) -> None:
        """Stop the camera preview thread; safe to call repeatedly."""
        self._preview_stop.set()
        thread = self._preview_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("Camera preview thread did not stop within the shutdown timeout")
        self._preview_thread = None

    def _pane_height(self, trackbar: str) -> int:
        """Pane height in pixels from its size trackbar (percent of ``preview_height``)."""
        try:
            pct = cv2.getTrackbarPos(trackbar, WINDOW_NAME)
        except cv2.error:
            pct = 100
        pct = max(_MIN_SIZE_PCT, pct)
        return max(1, round(self._preview_height * pct / 100.0))

    def _wrist_frame(self) -> np.ndarray | None:
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None or rgbd.get("rgb") is None:
            return None
        image = _as_preview_bgr(rgbd["rgb"])
        annotate = getattr(self.policy, "annotate_wrist", None)
        if callable(annotate):
            image = annotate(image)
        return image

    def _tabletop_frame(self) -> np.ndarray | None:
        rgbd = self.env._get_latest_rgbd()
        if rgbd is None or rgbd.get("rgb") is None:
            return None
        return _as_preview_bgr(rgbd["rgb"])

    def _window_closed(self, shown: bool) -> bool:
        if not shown:
            return False
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _create_window(self) -> None:
        """Create the shared window with one size trackbar per camera."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        if self._view_wrist:
            cv2.createTrackbar(WRIST_TRACKBAR, WINDOW_NAME, 100, _MAX_SIZE_PCT, lambda _: None)
        if self._view_tabletop:
            cv2.createTrackbar(TABLETOP_TRACKBAR, WINDOW_NAME, 100, _MAX_SIZE_PCT, lambda _: None)

    def _preview_loop(self) -> None:
        """Pull the latest frames and show them side by side in one window on a dedicated thread."""
        shown = False
        try:
            self._create_window()
            frame_period = 1.0 / self._preview_fps if self._preview_fps > 0 else 0.0
            while not self._preview_stop.is_set():
                with self._preview_lock:
                    status = self._status_text
                try:
                    panes = []
                    if self._view_wrist:
                        wrist = self._wrist_frame()
                        if wrist is not None:
                            panes.append(_resize_to_height(wrist, self._pane_height(WRIST_TRACKBAR)))
                    if self._view_tabletop:
                        tabletop = self._tabletop_frame()
                        if tabletop is not None:
                            panes.append(_resize_to_height(tabletop, self._pane_height(TABLETOP_TRACKBAR)))
                    if panes:
                        for pane in panes:
                            _draw_status(pane, status)
                        cv2.imshow(WINDOW_NAME, _compose_side_by_side(panes))
                        shown = True
                    cv2.waitKey(1)
                    if self._window_closed(shown):
                        print("[INFO][KortexViewer] Preview window closed.")
                        self._viewer_closed = True
                        break
                except cv2.error as exc:
                    logger.warning("Camera preview stopped because OpenCV GUI failed: %s", exc)
                    break

                if frame_period > 0:
                    self._preview_stop.wait(frame_period)
        except Exception:
            logger.exception("Camera preview thread failed")
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
