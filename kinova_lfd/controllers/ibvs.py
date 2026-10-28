"""Image-based visual servoing (IBVS) with HSV block corner features.

Eye-in-hand control law (Chaumette & Hutchinson / ViSP):

    v_c = -λ L⁺ (s - s*)

where s are normalized image point features and L stacks the point interaction
matrices. Camera twist is mapped to the TCP/tool frame using the fixed
wrist-camera → pinch_site extrinsic from the Gen3 MJCF.

Written for ROB545; works with Mj-Gen3 and Kortex-TCP twist interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from kinova_lfd.controllers.pid import PIDController, PIDGains

# Wrist camera and pinch_site poses relative to bracelet_link (MuJoCo wxyz).
# From kinova_lfd/envs/base/mujoco/assets/kinova_gen3/xmls/gen3_gripper.xml
_BRACELET_CAM_POS = np.array([0.0, -0.05639, -0.063475], dtype=np.float64)
_BRACELET_CAM_QUAT_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_BRACELET_PINCH_POS = np.array([0.0, 0.0, -0.211525], dtype=np.float64)
_BRACELET_PINCH_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class BlockHsvCfg:
    """HSV segmentation settings for a colored block.

    OpenCV hue is in [0, 179]. Multiple ranges support wraparound colors (red).
    """

    name: str = "pink"
    hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = field(
        default_factory=lambda: [((140, 40, 40), (175, 255, 255))]
    )
    min_contour_area: float = 500.0
    approx_epsilon_frac: float = 0.02


# Named presets for sim (pink_block) and common real-world block colors.
BLOCK_HSV_PRESETS: dict[str, BlockHsvCfg] = {
    "pink": BlockHsvCfg(
        name="pink",
        hsv_ranges=[((140, 40, 40), (175, 255, 255))],
    ),
    "red": BlockHsvCfg(
        name="red",
        hsv_ranges=[
            ((0, 70, 50), (10, 255, 255)),
            ((170, 70, 50), (179, 255, 255)),
        ],
    ),
    "green": BlockHsvCfg(
        name="green",
        hsv_ranges=[((35, 50, 40), (85, 255, 255))],
    ),
    "blue": BlockHsvCfg(
        name="blue",
        hsv_ranges=[((90, 50, 40), (130, 255, 255))],
    ),
    "yellow": BlockHsvCfg(
        name="yellow",
        hsv_ranges=[((20, 80, 80), (35, 255, 255))],
    ),
}


def get_block_hsv_preset(name: str) -> BlockHsvCfg:
    """Return a copy of a named HSV preset."""
    key = name.lower().strip()
    if key not in BLOCK_HSV_PRESETS:
        known = ", ".join(sorted(BLOCK_HSV_PRESETS))
        raise ValueError(f"Unknown block color preset '{name}'. Known: {known}")
    preset = BLOCK_HSV_PRESETS[key]
    return BlockHsvCfg(
        name=preset.name,
        hsv_ranges=list(preset.hsv_ranges),
        min_contour_area=preset.min_contour_area,
        approx_epsilon_frac=preset.approx_epsilon_frac,
    )


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def default_cam_T_tcp() -> tuple[np.ndarray, np.ndarray]:
    """Fixed transform of pinch_site (TCP) relative to the wrist camera.

    Returns:
        t_cam_tcp: (3,) position of TCP in the camera frame.
        R_cam_tcp: (3, 3) rotation taking TCP-frame vectors into the camera frame
            (columns are TCP axes expressed in camera coordinates).
    """
    R_b_c = quat_wxyz_to_rotmat(_BRACELET_CAM_QUAT_WXYZ)
    R_b_tcp = quat_wxyz_to_rotmat(_BRACELET_PINCH_QUAT_WXYZ)
    R_cam_tcp = R_b_c.T @ R_b_tcp
    t_cam_tcp = R_b_c.T @ (_BRACELET_PINCH_POS - _BRACELET_CAM_POS)
    return t_cam_tcp, R_cam_tcp


def camera_twist_to_tcp(
    v_cam: np.ndarray,
    w_cam: np.ndarray,
    t_cam_tcp: np.ndarray,
    R_cam_tcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera-frame twist to TCP-frame twist for a rigid eye-in-hand mount."""
    v_cam = np.asarray(v_cam, dtype=np.float64).reshape(3)
    w_cam = np.asarray(w_cam, dtype=np.float64).reshape(3)
    t_cam_tcp = np.asarray(t_cam_tcp, dtype=np.float64).reshape(3)
    R_cam_tcp = np.asarray(R_cam_tcp, dtype=np.float64).reshape(3, 3)
    w_tcp = R_cam_tcp.T @ w_cam
    v_tcp = R_cam_tcp.T @ (v_cam + np.cross(w_cam, t_cam_tcp))
    return v_tcp, w_tcp


def interaction_matrix(x: float, y: float, z: float) -> np.ndarray:
    """2x6 interaction matrix for a point feature at normalized (x, y) and depth Z."""
    z = max(float(z), 1e-4)
    return np.array(
        [
            [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
            [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
        ],
        dtype=np.float64,
    )


def ensure_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to HxWx3 uint8."""
    img = np.asarray(rgb)
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        max_val = float(np.nanmax(img)) if img.size else 1.0
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8) if max_val <= 1.0 + 1e-3 else np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def order_corners_ccw(corners_px: np.ndarray) -> np.ndarray:
    """Order 4 image corners counter-clockwise about their centroid, starting at top-left-ish."""
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 corners, got {pts.shape[0]}")
    center = pts.mean(axis=0)
    # Image y grows downward; negate y so atan2 sorts CCW in image space visually.
    angles = np.arctan2(-(pts[:, 1] - center[1]), pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # Rotate cyclic order so index 0 is the point with smallest (x+y) (near top-left).
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def extract_block_corners_px(rgb: np.ndarray, cfg: BlockHsvCfg) -> np.ndarray | None:
    """Segment a colored block and return 4 ordered corner pixels, or None on failure."""
    rgb_u8 = ensure_rgb_u8(rgb)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in cfg.hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)))

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg.min_contour_area:
        return None

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # 4x2 float32
    return order_corners_ccw(box)


def pixels_to_normalized(corners_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert pixel corners (N, 2) to normalized image coords (N, 2)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    x = (pts[:, 0] - cx) / fx
    y = (pts[:, 1] - cy) / fy
    return np.stack([x, y], axis=1)


def sample_depths(depth: np.ndarray | None, corners_px: np.ndarray, z_default: float) -> np.ndarray:
    """Sample depth at corner pixels; fall back to z_default when missing/invalid."""
    pts = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    zs = np.full(pts.shape[0], float(z_default), dtype=np.float64)
    if depth is None:
        return zs

    d = np.asarray(depth)
    if d.ndim == 3:
        d = d.squeeze()
    if d.ndim != 2:
        return zs

    h, w = d.shape
    for i, (u, v) in enumerate(pts):
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        z = float(d[vi, ui])
        if np.isfinite(z) and z > 1e-4:
            # MuJoCo depth is meters; guard against accidental mm-scale values.
            zs[i] = z * 0.001 if z > 100.0 else z
    return zs


def quat_error_rotvec(q_des_wxyz: np.ndarray, q_cur_wxyz: np.ndarray) -> np.ndarray:
    """Rotation-vector error that rotates ``q_cur`` into ``q_des`` (both wxyz)."""
    q_des = np.asarray(q_des_wxyz, dtype=np.float64).reshape(4)
    q_cur = np.asarray(q_cur_wxyz, dtype=np.float64).reshape(4)
    # q_err = q_des * q_cur^{-1}
    w0, x0, y0, z0 = q_des
    w1, x1, y1, z1 = q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]
    q_err = np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )
    if q_err[0] < 0.0:
        q_err = -q_err
    w = float(np.clip(q_err[0], -1.0, 1.0))
    xyz = q_err[1:]
    n = float(np.linalg.norm(xyz))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(n, w)
    return (xyz / n) * angle


def world_twist_to_tcp(v_world: np.ndarray, w_world: np.ndarray, quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame twist in the TCP/tool frame."""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return R.T @ np.asarray(v_world, dtype=np.float64).reshape(3), R.T @ np.asarray(w_world, dtype=np.float64).reshape(3)


class GraspPhase(Enum):
    """Finite-state phases after visual alignment."""

    IBVS = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    DONE = auto()


class IBVSController:
    """HSV corner IBVS + Z-axis PID grasp state machine.

    Phases: IBVS alignment → descend by world-Z offset → close gripper → lift
    back to the pre-grasp height. Descend/lift twists come from ``self.pid``.
    """

    def __init__(
        self,
        setpoint_rgb: np.ndarray,
        *,
        intrinsics_K: np.ndarray,
        block_hsv: BlockHsvCfg | None = None,
        lambda_gain: float = 0.5,
        z_default: float = 0.4,
        cam_T_tcp: tuple[np.ndarray, np.ndarray] | None = None,
        max_lin_vel: float = 0.15,
        max_ang_vel: float = 0.5,
        gripper_cmd: float = 0.0,
        grasp_z_offset: float = 0.08,
        ibvs_error_threshold: float = 0.05,
        ibvs_settle_steps: int = 15,
        pose_tol: float = 0.008,
        close_hold_steps: int = 40,
        dt: float = 0.01,
        pid: PIDController | None = None,
        pid_ori: PIDController | None = None,
    ) -> None:
        self.K = np.asarray(intrinsics_K, dtype=np.float64).reshape(3, 3)
        self.block_hsv = block_hsv if block_hsv is not None else get_block_hsv_preset("pink")
        self.lambda_gain = float(lambda_gain)
        self.z_default = float(z_default)
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self.gripper_cmd = float(gripper_cmd)
        self.grasp_z_offset = float(grasp_z_offset)
        self.ibvs_error_threshold = float(ibvs_error_threshold)
        self.ibvs_settle_steps = int(ibvs_settle_steps)
        self.pose_tol = float(pose_tol)
        self.close_hold_steps = int(close_hold_steps)
        self.dt = float(dt)

        if cam_T_tcp is None:
            self.t_cam_tcp, self.R_cam_tcp = default_cam_T_tcp()
        else:
            self.t_cam_tcp = np.asarray(cam_T_tcp[0], dtype=np.float64).reshape(3)
            self.R_cam_tcp = np.asarray(cam_T_tcp[1], dtype=np.float64).reshape(3, 3)

        self.pid = pid or PIDController(
            dim=3,
            gains=PIDGains(kp=1.2, ki=0.0, kd=0.05),
            i_limit=0.2,
            out_limit=self.max_lin_vel,
        )
        self.pid_ori = pid_ori or PIDController(
            dim=3,
            gains=PIDGains(kp=1.0, ki=0.0, kd=0.02),
            i_limit=0.2,
            out_limit=self.max_ang_vel,
        )

        corners_px = extract_block_corners_px(setpoint_rgb, self.block_hsv)
        if corners_px is None:
            raise ValueError(
                f"Failed to extract 4 block corners from setpoint with HSV preset "
                f"'{self.block_hsv.name}'. Check the image and hsv_ranges."
            )
        self._setpoint_corners_px = corners_px
        self.s_star = pixels_to_normalized(corners_px, self.K)  # (4, 2)
        self._last_corners_px: np.ndarray | None = None
        self._last_error_norm: float | None = None

        self.phase = GraspPhase.IBVS
        self._ibvs_ok_steps = 0
        self._close_steps = 0
        self._pregrasp_pose: np.ndarray | None = None  # (7,) xyz + quat wxyz
        self._pose_setpoint: np.ndarray | None = None

    @property
    def setpoint_corners_px(self) -> np.ndarray:
        return self._setpoint_corners_px.copy()

    def reset_phase(self) -> None:
        """Return to IBVS and clear grasp setpoints / PID state."""
        self.phase = GraspPhase.IBVS
        self._ibvs_ok_steps = 0
        self._close_steps = 0
        self._pregrasp_pose = None
        self._pose_setpoint = None
        self.gripper_cmd = 0.0
        self.pid.reset()
        self.pid_ori.reset()

    def extract_features(
        self, rgb: np.ndarray, depth: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (corners_px, s_normalized, depths) or None if detection fails."""
        corners_px = extract_block_corners_px(rgb, self.block_hsv)
        if corners_px is None:
            return None
        s = pixels_to_normalized(corners_px, self.K)
        z = sample_depths(depth, corners_px, self.z_default)
        return corners_px, s, z

    def _clip_twist(self, v_tcp: np.ndarray, w_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lin_norm = np.linalg.norm(v_tcp)
        if lin_norm > self.max_lin_vel > 0:
            v_tcp = v_tcp * (self.max_lin_vel / lin_norm)
        ang_norm = np.linalg.norm(w_tcp)
        if ang_norm > self.max_ang_vel > 0:
            w_tcp = w_tcp * (self.max_ang_vel / ang_norm)
        return v_tcp, w_tcp

    def _ibvs_twist(self, rgb: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
        features = self.extract_features(rgb, depth)
        if features is None:
            self._last_corners_px = None
            self._last_error_norm = None
            self._ibvs_ok_steps = 0
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_cmd], dtype=np.float64)

        corners_px, s, z = features
        self._last_corners_px = corners_px

        n = s.shape[0]
        L = np.zeros((2 * n, 6), dtype=np.float64)
        e = np.zeros(2 * n, dtype=np.float64)
        for i in range(n):
            L[2 * i : 2 * i + 2] = interaction_matrix(float(s[i, 0]), float(s[i, 1]), float(z[i]))
            e[2 * i] = float(s[i, 0] - self.s_star[i, 0])
            e[2 * i + 1] = float(s[i, 1] - self.s_star[i, 1])

        self._last_error_norm = float(np.linalg.norm(e))
        v_cam = -self.lambda_gain * np.linalg.pinv(L) @ e  # (6,)
        v_tcp, w_tcp = camera_twist_to_tcp(v_cam[:3], v_cam[3:], self.t_cam_tcp, self.R_cam_tcp)
        v_tcp, w_tcp = self._clip_twist(v_tcp, w_tcp)
        return np.concatenate([v_tcp, w_tcp, [self.gripper_cmd]]).astype(np.float64)

    def _enter_descend(self, tcp_pose: np.ndarray) -> None:
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(7)
        self._pregrasp_pose = pose.copy()
        self._pose_setpoint = pose.copy()
        # Lower only in world Z.
        self._pose_setpoint[2] = pose[2] - self.grasp_z_offset
        self.phase = GraspPhase.DESCEND
        self.gripper_cmd = 0.0
        self.pid.reset()
        self.pid_ori.reset()

    def _enter_close(self) -> None:
        self.phase = GraspPhase.CLOSE
        self._close_steps = 0
        self.gripper_cmd = 1.0
        self.pid.reset()
        self.pid_ori.reset()

    def _enter_lift(self) -> None:
        assert self._pregrasp_pose is not None
        self._pose_setpoint = self._pregrasp_pose.copy()
        self.phase = GraspPhase.LIFT
        self.gripper_cmd = 1.0
        self.pid.reset()
        self.pid_ori.reset()

    def _pid_pose_twist(self, tcp_pose: np.ndarray, dt: float) -> np.ndarray:
        """PID world-pose tracking expressed as a TCP-frame twist."""
        assert self._pose_setpoint is not None
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(7)
        des = self._pose_setpoint

        # Keep XY and orientation at the pre-grasp values; only Z is offset.
        pos_err_world = np.array(
            [des[0] - pose[0], des[1] - pose[1], des[2] - pose[2]],
            dtype=np.float64,
        )
        # Zero XY tracking effort if we only care about Z motion once aligned —
        # still correct XY drift so the grasp stays centered.
        ori_err_world = quat_error_rotvec(des[3:7], pose[3:7])

        v_world = self.pid(pos_err_world, dt)
        w_world = self.pid_ori(ori_err_world, dt)
        v_tcp, w_tcp = world_twist_to_tcp(v_world, w_world, pose[3:7])
        v_tcp, w_tcp = self._clip_twist(v_tcp, w_tcp)
        return np.concatenate([v_tcp, w_tcp, [self.gripper_cmd]]).astype(np.float64)

    def _pose_reached(self, tcp_pose: np.ndarray) -> bool:
        assert self._pose_setpoint is not None
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(7)
        return float(np.linalg.norm(self._pose_setpoint[:3] - pose[:3])) < self.pose_tol

    def __call__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        tcp_pose: np.ndarray | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Compute a 7D TCP twist command ``[vx,vy,vz,wx,wy,wz,gripper]``.

        ``tcp_pose`` is required after IBVS converges (7D xyz + quat wxyz).
        """
        step_dt = self.dt if dt is None else float(dt)

        if self.phase is GraspPhase.IBVS:
            twist = self._ibvs_twist(rgb, depth)
            if (
                self._last_error_norm is not None
                and self._last_error_norm < self.ibvs_error_threshold
                and tcp_pose is not None
            ):
                self._ibvs_ok_steps += 1
                if self._ibvs_ok_steps >= self.ibvs_settle_steps:
                    self._enter_descend(tcp_pose)
            else:
                self._ibvs_ok_steps = 0
            return twist

        if tcp_pose is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_cmd], dtype=np.float64)

        if self.phase is GraspPhase.DESCEND:
            twist = self._pid_pose_twist(tcp_pose, step_dt)
            if self._pose_reached(tcp_pose):
                self._enter_close()
                twist[-1] = self.gripper_cmd
            return twist

        if self.phase is GraspPhase.CLOSE:
            # Hold Cartesian pose while closing.
            twist = self._pid_pose_twist(tcp_pose, step_dt)
            twist[:6] = 0.0
            twist[-1] = self.gripper_cmd
            self._close_steps += 1
            if self._close_steps >= self.close_hold_steps:
                self._enter_lift()
                twist[-1] = self.gripper_cmd
            return twist

        if self.phase is GraspPhase.LIFT:
            twist = self._pid_pose_twist(tcp_pose, step_dt)
            if self._pose_reached(tcp_pose):
                self.phase = GraspPhase.DONE
            return twist

        # DONE: hold with gripper closed.
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
