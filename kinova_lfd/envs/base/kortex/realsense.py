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


_KORTEX_RESOLUTION_WH = {
    1: (320, 240),
    2: (424, 240),
    3: (480, 270),
    4: (640, 480),
    5: (1280, 720),
    6: (1920, 1080),
}
"""Kortex ``VisionConfig.Resolution`` enum -> (width, height)."""


class WristCamera:
    """Wrist Camera for the Kinova Gen3.

    The color stream is read over RTSP on a background thread. Intrinsics are pulled
    once at construction through the Kortex ``VisionConfig`` service.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.10",
        rate: int = 20,
        router: Any | None = None,
        username: str = "admin",
        password: str = "admin",
    ) -> None:
        """Initialize the RGB stream and query the color-sensor intrinsics.

        Args:
            robot_ip: IP of the robot.
            rate: Nominal stream rate (Hz).
            router: Optional open Kortex TCP ``RouterClient``. When ``None`` a short-lived
                TCP session is opened to query the intrinsics.
            username: Kortex username used when ``router`` is ``None``.
            password: Kortex password used when ``router`` is ``None``.

        """
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.robot_ip = robot_ip
        self._latest_depth_frame = None
        self._latest_rgb_frame: np.ndarray | None = None
        """Latest raw color frame in BGR (H, W, 3), as returned by OpenCV."""
        self._latest_timestamp: float | None = None
        self._curr_depth_frame = None
        self._lock = threading.Lock()
        self._rate = 1 / rate

        self._intrinsic_k, self._intrinsic_wh = self._query_intrinsics(router, username, password)

        # RGB
        self._rgb_wrist_frame_cap = cv2.VideoCapture(f"rtsp://{self.robot_ip}/color", cv2.CAP_FFMPEG)

    def _query_intrinsics(
        self, router: Any | None, username: str, password: str
    ) -> tuple[np.ndarray, tuple[int, int] | None]:
        """Read the color-sensor intrinsics from the Kortex ``VisionConfig`` service.

        Returns:
            ``(K, (width, height))`` where ``K`` is the (3, 3) float64 intrinsic matrix and
            ``(width, height)`` is the resolution the intrinsics were calibrated for.

        """
        if router is None:
            from kinova_lfd.envs.base.kortex.kortex_tcp.kortex_tcp_bridge import DeviceConnection

            with DeviceConnection.create_tcp_connection(
                ip=self.robot_ip, username=username, password=password
            ) as tmp_router:
                return self._read_intrinsics(tmp_router)
        return self._read_intrinsics(router)

    @staticmethod
    def _read_intrinsics(router: Any) -> tuple[np.ndarray, tuple[int, int] | None]:
        """Query ``GetIntrinsicParameters`` for the color sensor on the vision module."""
        from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
        from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
        from kortex_api.autogen.messages import Common_pb2, VisionConfig_pb2

        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)

        vision_device_id = None
        for handle in device_manager.ReadAllDevices().device_handle:
            if handle.device_type == Common_pb2.VISION:
                vision_device_id = handle.device_identifier
                break
        if vision_device_id is None:
            raise RuntimeError("[Kortex] No vision module found; cannot read wrist-camera intrinsics.")

        sensor_id = VisionConfig_pb2.SensorIdentifier()
        sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
        intr = vision_config.GetIntrinsicParameters(sensor_id, vision_device_id)

        intrinsic_k = np.array(
            [
                [intr.focal_length_x, 0.0, intr.principal_point_x],
                [0.0, intr.focal_length_y, intr.principal_point_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return intrinsic_k, _KORTEX_RESOLUTION_WH.get(int(intr.resolution))

    def intrinsic_k_for(self, width: int, height: int) -> np.ndarray:
        """Return the intrinsics scaled to an image of ``width`` x ``height``."""
        K = self._intrinsic_k.copy()
        if self._intrinsic_wh is not None and self._intrinsic_wh != (width, height):
            sx = width / self._intrinsic_wh[0]
            sy = height / self._intrinsic_wh[1]
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        return K

    def start(self) -> None:
        """Start the background RTSP reader."""
        self._thread = threading.Thread(target=self.run, name="WristRgbThread", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Continuously read frames, keeping only successful reads."""
        while True:
            ok, frame = self._rgb_wrist_frame_cap.read()
            if not ok or frame is None:
                continue
            with self._lock:
                self._latest_rgb_frame = frame
                self._latest_timestamp = float(time.time())

    def _get_latest_rgbd_raw(self) -> dict[str, Any] | None:
        """Return the latest wrist frame in the same layout as the MuJoCo environment.

        Returns:
            ``None`` before the first frame arrives, otherwise a dictionary containing:
              - ``rgb``: (3, H, W) uint8 RGB image
              - ``depth``: ``None`` (the wrist depth stream is not read)
              - ``intrinsic_k``: (3, 3) float64 camera intrinsic matrix
              - ``camera_pose``: ``None`` (not available on hardware)
              - ``timestamp``: float timestamp in seconds

        """
        with self._lock:
            frame = self._latest_rgb_frame
            timestamp = self._latest_timestamp
        if frame is None:
            return None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        return {
            "rgb": rgb,
            "depth": None,
            "intrinsic_k": self.intrinsic_k_for(w, h),
            "camera_pose": None,
            "timestamp": timestamp,
        }
