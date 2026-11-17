import threading
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


class RealsenseCamera:
    """Class that manages the Realsense camera streaming.

    Frames are read on a background thread (``start``) and cached, so
    ``_get_latest_rgbd_raw`` never blocks on the camera.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize the RealSense pipeline and read the color-stream intrinsics."""
        self.width = width
        self.height = height
        self.fps = fps

        # RealSense pipeline + streams.
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self._profile = self._pipeline.start(self._config)

        # Align depth to color so pixels correspond.
        self._align = rs.align(rs.stream.color)

        # Intrinsics for the color stream.
        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intrinsic_k = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Depth scale (meters per depth unit) retained for reference; we keep depth
        # in uint16 units to match the ROS2 RGB-D observation convention.
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self._latest_rgb: np.ndarray | None = None
        """Latest color frame in RGB (3, H, W) uint8."""
        self._latest_depth: np.ndarray | None = None
        """Latest aligned depth frame (H, W) uint16."""
        self._latest_timestamp: float | None = None
        self._lock = threading.Lock()
        self._first_frame = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        """Start the background frame reader (no-op if it is already running)."""
        if self._closed:
            raise RuntimeError("RealsenseCamera is closed. Create a new instance to continue streaming.")
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, name="RealsenseThread", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Continuously read and align frames, caching the latest RGB-D pair."""
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except RuntimeError:
                # Timed out waiting for a frame; check the stop flag and retry.
                continue
            frames = self._align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # (H, W, 3) BGR uint8 -> (3, H, W) RGB. cvtColor and astype(copy=True) both copy out of
            # the RealSense frame buffer, which is recycled once the frame is released.
            rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
            depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=True)
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth = depth
                self._latest_timestamp = float(time.time())
            self._first_frame.set()

    def wait_for_frame(self, timeout: float = 10.0) -> None:
        """Block until the first frame arrives, raising if none arrives within ``timeout`` seconds."""
        if not self._first_frame.wait(timeout):
            raise RuntimeError(f"[RealSense] Camera produced no frames in {timeout:.0f}s.")

    def stop(self) -> None:
        """Stop the reader thread and the RealSense pipeline."""
        if self._closed:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                return
        self._pipeline.stop()
        self._closed = True

    def _get_latest_rgbd_raw(self) -> dict[str, Any] | None:
        """Return the latest cached RGB-D snapshot without blocking.

        Returns:
            ``None`` before the first frame arrives, otherwise a new dictionary containing:
              - ``rgb``: (3, H, W) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: ``None`` (not available on hardware)
              - ``timestamp``: float timestamp in seconds

        """
        with self._lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            timestamp = self._latest_timestamp
        if rgb is None:
            return None
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": self._intrinsic_k.copy(),
            "camera_pose": None,
            "timestamp": timestamp,
        }
