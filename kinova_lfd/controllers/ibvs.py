"""Image-based visual servoing (IBVS) with HSV block corner features.

Eye-in-hand control law (Chaumette & Hutchinson / ViSP):

    v_c = -lambda L⁺ (s - s*)

where s are normalized image point features and L stacks the point interaction
matrices. Camera twist is mapped to the TCP/tool frame using the fixed
wrist-camera -> pinch_site extrinsic from the Gen3 MJCF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from kinova_lfd.controllers.pid import PIDController, PIDGains

_BRACELET_CAM_POS = np.array([0.0, -0.05639, -0.063475])
_BRACELET_CAM_QUAT_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
_BRACELET_PINCH_POS = np.array([0.0, 0.0, -0.211525])
_BRACELET_PINCH_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])


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
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def default_cam_T_tcp() -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed transform of pinch_site (TCP) relative to the wrist camera.

    MJCF camera poses are in the MuJoCo camera body frame (looks along -Z).
    IBVS interaction matrices expect an OpenCV/optical frame (X right, Y down,
    Z forward/+look). Convert with ``diag(1, -1, -1)`` before returning.

    Returns:
        t_cam_tcp: (3,) position of TCP in the **optical** camera frame.
        R_cam_tcp: (3, 3) rotation taking TCP-frame vectors into the optical
            camera frame (columns are TCP axes expressed in optical coordinates).

    """
    R_b_c = quat_wxyz_to_rotmat(_BRACELET_CAM_QUAT_WXYZ)
    R_b_tcp = quat_wxyz_to_rotmat(_BRACELET_PINCH_QUAT_WXYZ)
    # Camera body frame (looks along -Z).
    R_cam_tcp = R_b_c.T @ R_b_tcp
    t_cam_tcp = R_b_c.T @ (_BRACELET_PINCH_POS - _BRACELET_CAM_POS)
    # OpenCV optical frame: X right, Y down, Z forward (= MuJoCo -Z).
    r_opt_mj = np.diag([1.0, -1.0, -1.0])
    t_cam_tcp = r_opt_mj @ t_cam_tcp
    R_cam_tcp = r_opt_mj @ R_cam_tcp
    return t_cam_tcp, R_cam_tcp


def camera_twist_to_tcp(
    v_cam: np.ndarray,
    w_cam: np.ndarray,
    t_cam_tcp: np.ndarray,
    R_cam_tcp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera-frame twist to TCP-frame twist for a rigid eye-in-hand mount."""
    w_tcp = R_cam_tcp.T @ w_cam
    v_tcp = R_cam_tcp.T @ (v_cam + np.cross(w_cam, t_cam_tcp))
    return v_tcp, w_tcp


def interaction_matrix(x: float, y: float, z: float) -> np.ndarray:
    """2x6 interaction matrix for a point feature at normalized (x, y) and depth Z."""
    z = max(z, 1e-4)
    return np.array(
        [
            [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y],
            [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x],
        ]
    )


def order_corners_ccw(corners_px: np.ndarray) -> np.ndarray:
    """Order 4 image corners counter-clockwise about their centroid, starting at top-left-ish."""
    pts = np.asarray(corners_px).reshape(-1, 2)
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
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
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
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (corners_px[:, 0] - cx) / fx
    y = (corners_px[:, 1] - cy) / fy
    return np.stack([x, y], axis=1)


def sample_depths(
    depth: np.ndarray | None,
    corners_px: np.ndarray,
    z_default: float,
    *,
    inset_frac: float = 0.65,
) -> np.ndarray:
    """Sample depth near corner pixels; fall back to z_default when missing/invalid.

    ``minAreaRect`` corners often land on the silhouette / table. Sample along the
    ray from the corner centroid toward each corner at ``inset_frac`` of the radius
    so the lookup hits the block face instead of the background.
    """
    pts = np.asarray(corners_px).reshape(-1, 2)
    zs = np.full(pts.shape[0], z_default)
    if depth is None:
        return zs

    d = np.asarray(depth)
    if d.ndim == 3:
        d = d.squeeze()
    if d.ndim != 2:
        return zs

    # Pull sample points inward from silhouette corners toward the centroid.
    frac = float(np.clip(inset_frac, 0.0, 1.0))
    if pts.shape[0] >= 1 and frac < 1.0:
        center = pts.mean(axis=0, keepdims=True)
        sample_pts = center + frac * (pts - center)
    else:
        sample_pts = pts

    h, w = d.shape
    for i, (u, v) in enumerate(sample_pts):
        ui = int(np.clip(round(u), 0, w - 1))
        vi = int(np.clip(round(v), 0, h - 1))
        z = float(d[vi, ui])
        if np.isfinite(z) and z > 1e-4:
            zs[i] = z * 0.001 if z > 100.0 else z
    return zs


def quat_error_rotvec(q_des_wxyz: np.ndarray, q_cur_wxyz: np.ndarray) -> np.ndarray:
    """Rotation-vector error that rotates ``q_cur`` into ``q_des`` (both wxyz)."""
    w0, x0, y0, z0 = q_des_wxyz
    w1, x1, y1, z1 = q_cur_wxyz[0], -q_cur_wxyz[1], -q_cur_wxyz[2], -q_cur_wxyz[3]
    q_err = np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )
    if q_err[0] < 0.0:
        q_err = -q_err
    w = float(np.clip(q_err[0], -1.0, 1.0))
    xyz = q_err[1:]
    n = float(np.linalg.norm(xyz))
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, w)
    return (xyz / n) * angle


def world_twist_to_tcp(
    v_world: np.ndarray, w_world: np.ndarray, quat_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame twist in the TCP/tool frame."""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    return R.T @ v_world, R.T @ w_world


class GraspPhase(Enum):
    """Finite-state phases after visual alignment."""

    IBVS = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    DONE = auto()


class IBVSController:
    """HSV corner IBVS + Z-axis PID grasp state machine.

    Phases: IBVS alignment -> descend by world-Z offset -> close gripper -> lift
    back to the pre-grasp height. Descend/lift twists come from ``self.pid``.
    """

    def __init__(
        self,
        setpoint_rgb: np.ndarray,
        *,
        intrinsics_k: np.ndarray,
        block_hsv: BlockHsvCfg | None = None,
        lambda_gain: float = 0.35,
        z_default: float = 0.28,
        cam_t_tcp: tuple[np.ndarray, np.ndarray] | None = None,
        max_lin_vel: float = 0.15,
        max_ang_vel: float = 0.15,
        gripper_cmd: float = 0.0,
        grasp_z_abs: float = 0.03,
        ibvs_error_threshold: float = 0.02,
        ibvs_settle_steps: int = 30,
        use_angular: bool = False,
        pose_tol: float = 0.008,
        xy_pose_tol: float = 0.02,
        close_hold_steps: int = 40,
        dt: float = 0.01,
        pid: PIDController | None = None,
        pid_ori: PIDController | None = None,
    ) -> None:
        self.K = intrinsics_k
        self.block_hsv = block_hsv if block_hsv is not None else get_block_hsv_preset("pink")
        self.lambda_gain = lambda_gain
        self.z_default = z_default
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.gripper_cmd = gripper_cmd
        self.grasp_z_abs = grasp_z_abs
        self.ibvs_error_threshold = ibvs_error_threshold
        self.ibvs_settle_steps = ibvs_settle_steps
        self.use_angular = use_angular
        self.pose_tol = pose_tol
        self.xy_pose_tol = xy_pose_tol
        self.close_hold_steps = close_hold_steps
        self.dt = dt
        self._last_twist: np.ndarray | None = None

        if cam_t_tcp is None:
            self.t_cam_tcp, self.R_cam_tcp = default_cam_T_tcp()
        else:
            self.t_cam_tcp = cam_t_tcp[0]
            self.R_cam_tcp = cam_t_tcp[1]

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
            self._last_twist = None
            self._ibvs_ok_steps = 0
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_cmd])

        corners_px, s, z = features
        self._last_corners_px = corners_px

        n = s.shape[0]
        L = np.zeros((2 * n, 6))
        e = np.zeros(2 * n)
        for i in range(n):
            L[2 * i : 2 * i + 2] = interaction_matrix(float(s[i, 0]), float(s[i, 1]), float(z[i]))
            e[2 * i] = float(s[i, 0] - self.s_star[i, 0])
            e[2 * i + 1] = float(s[i, 1] - self.s_star[i, 1])

        self._last_error_norm = float(np.linalg.norm(e))
        if self.use_angular:
            v_cam = -self.lambda_gain * np.linalg.pinv(L) @ e  # (6,)
        else:
            # Translation-only: avoid early wx/wy tilt from angular columns of L.
            v_lin = -self.lambda_gain * np.linalg.pinv(L[:, :3]) @ e
            v_cam = np.zeros(6)
            v_cam[:3] = v_lin
        v_tcp, w_tcp = camera_twist_to_tcp(v_cam[:3], v_cam[3:], self.t_cam_tcp, self.R_cam_tcp)
        v_tcp, w_tcp = self._clip_twist(v_tcp, w_tcp)
        twist = np.concatenate([v_tcp, w_tcp, [self.gripper_cmd]]).astype(np.float64)
        self._last_twist = twist
        return twist

    def _enter_descend(self, tcp_pose: np.ndarray) -> None:
        """Descend to pre-specified grasp height."""
        self._pregrasp_pose = tcp_pose.copy()
        self._pose_setpoint = tcp_pose.copy()
        self._pose_setpoint[2] = self.grasp_z_abs
        self.phase = GraspPhase.DESCEND
        self.gripper_cmd = 0.0
        self.pid.reset()
        self.pid_ori.reset()

    def _enter_close(self) -> None:
        """Close the gripper."""
        self.phase = GraspPhase.CLOSE
        self._close_steps = 0
        self.gripper_cmd = 1.0

    def _enter_lift(self) -> None:
        """Lift the gripper."""
        assert self._pregrasp_pose is not None
        self._pose_setpoint = self._pregrasp_pose.copy()
        self.phase = GraspPhase.LIFT
        self.gripper_cmd = 1.0
        self.pid.reset()
        self.pid_ori.reset()

    def _pid_pose_twist(self, tcp_pose: np.ndarray, dt: float) -> np.ndarray:
        """PID world-pose tracking expressed as a TCP-frame twist."""
        assert self._pose_setpoint is not None
        des = self._pose_setpoint

        # Keep XY and orientation at the pre-grasp values; only Z is offset.
        pos_err_world = np.array([des[0] - tcp_pose[0], des[1] - tcp_pose[1], des[2] - tcp_pose[2]])
        # Zero XY tracking effort if we only care about Z motion once aligned —
        # still correct XY drift so the grasp stays centered.
        ori_err_world = quat_error_rotvec(des[3:7], tcp_pose[3:7])

        v_world = self.pid(pos_err_world, dt)
        w_world = self.pid_ori(ori_err_world, dt)
        v_tcp, w_tcp = world_twist_to_tcp(v_world, w_world, tcp_pose[3:7])
        v_tcp, w_tcp = self._clip_twist(v_tcp, w_tcp)
        return np.concatenate([v_tcp, w_tcp, [self.gripper_cmd]])

    def _pose_reached(self, tcp_pose: np.ndarray) -> bool:
        """Check if the pose has been reached."""
        assert self._pose_setpoint is not None
        des = self._pose_setpoint
        dz = abs(float(des[2] - tcp_pose[2]))
        dxy = float(np.linalg.norm(des[:2] - tcp_pose[:2]))
        return dz < self.pose_tol and dxy < self.xy_pose_tol

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
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_cmd])

        if self.phase is GraspPhase.DESCEND:
            twist = self._pid_pose_twist(tcp_pose, step_dt)
            if self._pose_reached(tcp_pose):
                self._enter_close()
                twist[-1] = self.gripper_cmd
            return twist

        if self.phase is GraspPhase.CLOSE:
            # Keep Cartesian PID active so residual world-Z can finish while closing.
            twist = self._pid_pose_twist(tcp_pose, step_dt)
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

        return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
