"""camera_viewer.py.

OpenCV viewer for a ``CameraEnv``. Mirrors the ``KortexViewer`` / ``MjViewer`` API
(``run`` / ``open_display`` / ``sync_display`` / ``close_display`` / ``viewer_closed``)
and shows the env's single camera in an OpenCV window scaled to ``preview_height``.

Pressing the record key (default V, read globally through pynput) starts recording the
displayed window, overlays included; pressing it again stops and saves an ``.mp4`` to
``video_dir``.

Written by Will Solow, 2026.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.envs.base.cameras.camera import CameraEnv
from kinova_lfd.envs.base.viewers.recorder import Preview, as_rgbd_frame


class CameraViewer:
    """Step a ``CameraEnv`` and show its latest frame in an OpenCV window.

    ``CameraEnv.step`` never blocks, so :meth:`run` steps the env at ``preview_fps`` on the
    calling thread and pushes each frame to the preview, which draws on its own daemon thread.
    External loops can instead call :meth:`open_display`, then :meth:`sync_display` with their
    own frames, then :meth:`close_display`.
    """

    def __init__(
        self,
        env: CameraEnv,
        preview_fps: float = 30.0,
        preview_height: int = 480,
        record_key: str | None = "v",
        video_dir: str | Path = "exp/camera_videos",
        resize_size: tuple[int, int] | None = None,
        annotate: Callable[[np.ndarray], np.ndarray] | None = None,
        window_name: str | None = None,
    ) -> None:
        """Initialize the viewer.

        Args:
            env: Camera environment to display.
            preview_fps: Preview refresh rate and :meth:`run` loop rate; also the frame rate
                of recorded videos.
            preview_height: Height (px) of the camera pane.
            record_key: Key that toggles video recording, or ``None`` to disable recording.
            video_dir: Directory where recorded videos are saved.
            resize_size: ``(H, W)`` to also show the raw frame resized to that size in a second
                window, or ``None`` to disable.
            annotate: Optional ``annotate(bgr) -> bgr`` drawing applied to the displayed frame.
            window_name: OpenCV window title. Defaults to ``"Camera (<cfg.name>)"``.

        """
        self.env = env
        self._period = 1.0 / preview_fps if preview_fps > 0 else 0.0
        name = getattr(getattr(env, "cfg", None), "name", "") or "camera"
        # The shared Preview has wrist + tabletop slots; the single camera uses the wrist slot
        # so ``annotate`` applies to it.
        self._preview = Preview(
            window_name or f"Camera ({name})",
            view_wrist=True,
            view_tabletop=False,
            fps=preview_fps,
            height=preview_height,
            frame_source=None,
            annotate_wrist=annotate,
            record_key=record_key,
            video_dir=video_dir,
            video_prefix=name,
            log_tag="CameraViewer",
            resize_size=resize_size,
        )
        self._opened = False
        self._step_count = 0

    def run(self) -> None:
        """Show the live camera feed until the window closes or Ctrl-C, then close the env."""
        self.open_display()
        try:
            next_tick = time.perf_counter()
            while True:
                obs = self.env.step()
                self._step_count += 1
                if not self.sync_display(None if obs is None else obs["rgb"]):
                    break
                if self._period > 0:
                    next_tick += self._period
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
        except KeyboardInterrupt:
            print("[INFO][CameraViewer] Interrupted.")
        finally:
            self.close_display()
            self.env.close()

    def open_display(self) -> None:
        """Start the camera preview thread (for external loops)."""
        if self._opened:
            return
        self._opened = True
        self._preview.start()

    def sync_display(self, rgb: Any | None = None, status: str = "") -> bool:
        """Push a camera frame and status text to the preview thread.

        Args:
            rgb: RGB image, ``(3, H, W)`` or ``(H, W, 3)``. ``None`` keeps the previous frame.
            status: Text drawn on the preview.

        Returns:
            ``False`` once the preview window has been closed, else ``True``.

        """
        if self._preview.closed:
            self._preview.stop()
            return False
        self._preview.publish(wrist=as_rgbd_frame(rgb), status=status)
        return True

    def close_display(self) -> None:
        """Stop the preview thread. Safe to call repeatedly."""
        self._preview.stop()
        self._opened = False

    @property
    def viewer_closed(self) -> bool:
        """Whether the preview window was closed by the user."""
        return self._preview.closed

    @property
    def recording(self) -> bool:
        """Whether a recording is in progress."""
        return self._preview.recording
