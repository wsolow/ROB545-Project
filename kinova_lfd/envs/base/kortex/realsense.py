import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


class RealsenseCamera:
    """Class that manages the Realsense camera streaming ."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize the RealSense pipeline and RGB-D observation space."""
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
        self._closed = False

    def _get_latest_rgbd_raw(self) -> dict[str, Any]:
        """Grab the latest RGB-D snapshot in the raw ROS-style format.

        Returns:
            A dictionary containing:
              - ``rgb``: (H, W, 3) uint8 RGB image
              - ``depth``: (H, W) uint16 depth image
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: 7D float64 array (x, y, z, qx, qy, qz, qw) in xyzw
              - ``timestamp``: float timestamp in seconds

        """
        if self._closed:
            raise RuntimeError("RealsenseEnv is closed. Create a new instance to continue streaming.")

        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to acquire both color and depth frames from RealSense.")

        # (H, W, 3) BGR uint8 -> RGB
        color_bgr = np.asanyarray(color_frame.get_data())
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # (H, W) uint16 depth image (no conversion to meters here).
        depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=False)

        # Wall-clock timestamp in seconds.
        timestamp = float(time.time())

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsic_k": self._intrinsic_k,
            "timestamp": timestamp,
        }


class WristCamera:
    """Wrist Camera for the Kinova Gen3."""

    def __init__(self, robot_ip: str = "192.168.1.10", rate: int = 20) -> None:
        """Initialize the depth camera and RGB camera of the Gen3 wrist camera."""
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.robot_ip = robot_ip
        self._latest_depth_frame = None
        self._latest_rgb_frame = None
        self._curr_depth_frame = None
        self._lock = threading.Lock()
        self._rate = 1 / rate
        # RGB
        self._rgb_wrist_frame_cap = cv2.VideoCapture(f"rtsp://{self.robot_ip}/color", cv2.CAP_FFMPEG)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="WristRgbThread", daemon=True)
        self._thread.start()

    def run(self):
        while True:
            _, self._latest_rgb_frame = self._rgb_wrist_frame_cap.read()

    def _get_latest_rgbd_raw(self) -> dict[str, Any]:

        _, self._latest_rgb_frame = self._rgb_wrist_frame_cap.read()
        self._latest_depth_frame = None

        return {"rgb": self._latest_rgb_frame, "depth": self._latest_depth_frame}
