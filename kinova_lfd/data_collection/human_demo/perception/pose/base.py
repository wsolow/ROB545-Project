"""6D pose estimation interface, config, factory, and the shared register / track loop.

Two layers:

- :class:`PoseEstimator` (client-facing): estimates every object's pose through one episode from
  the episode's ``SegmentationResult`` and writes a :class:`PoseResult` ``.npz``. Backends are
  chosen by ``PoseEstimatorCfg.name``.
- :class:`PoseModel` (in-process, used inside a server): per-object ``register`` / ``track``.
  :func:`estimate_object_track` runs the backend-independent loop on top of it:

  * register on the first frame with a SAM mask, then track frame to frame;
  * frames without a SAM mask are marked ``valid=False`` (the tracker keeps running so it stays warm);
  * re-register from the mask when the object reappears after having no mask, or when tracking
    looks lost (projected center outside the SAM box, or depth disagrees with the masked depth);
  * every re-registration is snapped to the symmetric equivalent closest to the last valid pose,
    so symmetric objects (e.g. a cube) do not flip orientation in the saved trajectory.

Written by Will Solow, 2026.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import RawEpisode
from kinova_lfd.data_collection.human_demo.perception.types import ObjectSpec

POSE_ESTIMATORS = ("foundationpose",)


@dataclass
class PoseEstimatorCfg:
    """Settings shared by every pose estimator."""

    name: str = ""
    """Backend to create, one of ``POSE_ESTIMATORS``."""


@dataclass
class TrackingCfg:
    """Register / track loop settings (backend independent)."""

    min_mask_pixels: int = 100
    """A SAM mask smaller than this counts as no mask (too occluded to register from)."""
    box_margin_frac: float = 0.25
    """Tracking is lost if the projected object center leaves the SAM box grown by this fraction."""
    max_depth_error_m: float = 0.05
    """Tracking is lost if the object center depth differs from the median masked depth by more."""
    reregister_every: int = 0
    """Also re-register every N frames with a mask (0 = only on reappearance / loss)."""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PoseEstimator(ABC):
    """Estimate object poses through a whole episode."""

    def __init__(self, cfg: PoseEstimatorCfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def estimate_episode(
        self,
        episode_dir: Path,
        segmentation_path: Path,
        objects: Sequence[ObjectSpec],
        out_path: Path,
        fps: float,
    ) -> dict[str, Any]:
        """Estimate ``objects`` in the raw episode using its segmentation; save a ``PoseResult`` to ``out_path``.

        Returns:
            A summary dict (frames, per-object valid / registered counts, seconds, ...).

        """
        raise NotImplementedError

    def check_ready(self) -> dict[str, Any]:
        """Raise if the backend is unavailable (e.g. its server is down); return backend info."""
        return {}

    def close(self) -> None:
        """Release resources."""


class PoseModel(ABC):
    """In-process model for one object at a time (``set_object`` selects it)."""

    @abstractmethod
    def set_object(self, mesh_path: Path, symmetry_tfs: np.ndarray) -> None:
        """Select the object (mesh in meters) for the following calls; resets tracking state."""

    @abstractmethod
    def register(
        self, rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Estimate the pose from scratch; returns ``(T_C_O (4, 4), score)``."""

    @abstractmethod
    def track(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Refine the previous pose on a new frame; returns ``T_C_O (4, 4)``."""

    @abstractmethod
    def set_pose(self, T_C_O: np.ndarray) -> None:
        """Overwrite the pose tracking continues from (used after symmetry snapping)."""


def snap_to_symmetry(T: np.ndarray, T_ref: np.ndarray, symmetry_tfs: np.ndarray) -> np.ndarray:
    """Return ``T @ S`` for the symmetry ``S`` whose rotation is closest to ``T_ref``'s."""
    candidates = T[None] @ symmetry_tfs
    # Rotation angle between R_ref and R_c is a monotone function of trace(R_ref^T R_c).
    traces = np.einsum("ij,kij->k", T_ref[:3, :3], candidates[:, :3, :3])
    return candidates[int(np.argmax(traces))]


def tracking_lost(
    T: np.ndarray, K: np.ndarray, box: np.ndarray, mask: np.ndarray, depth_m: np.ndarray, cfg: TrackingCfg
) -> str | None:
    """Return why the tracked pose disagrees with the SAM mask, or ``None`` if it looks fine."""
    t = T[:3, 3]
    if not np.all(np.isfinite(T)) or t[2] <= 0:
        return "invalid pose"
    u, v = (K @ t)[:2] / t[2]
    x1, y1, x2, y2 = box
    mx, my = cfg.box_margin_frac * (x2 - x1), cfg.box_margin_frac * (y2 - y1)
    if not (x1 - mx <= u <= x2 + mx and y1 - my <= v <= y2 + my):
        return "center outside box"
    masked = depth_m[mask & (depth_m > 0)]
    if masked.size and abs(t[2] - float(np.median(masked))) > cfg.max_depth_error_m:
        return "depth mismatch"
    return None


def estimate_object_track(
    model: PoseModel,
    episode: RawEpisode,
    masks: np.ndarray,
    boxes: np.ndarray,
    symmetry_tfs: np.ndarray,
    cfg: TrackingCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the register / track loop for one object (``model.set_object`` already called).

    Args:
        masks: (N, H, W) bool SAM masks for this object.
        boxes: (N, 4) xyxy SAM boxes (NaN where undetected).

    Returns:
        ``(T_C_O (N,4,4), valid (N,), registered (N,), scores (N,), stats)``.

    """
    n = len(episode)
    K = episode.intrinsic_k
    T_all = np.full((n, 4, 4), np.nan)
    valid = np.zeros(n, dtype=bool)
    registered = np.zeros(n, dtype=bool)
    scores = np.full(n, np.nan, dtype=np.float32)
    reasons: dict[str, int] = {}

    has_mask = masks.reshape(n, -1).sum(axis=1) >= cfg.min_mask_pixels
    T_valid_last: np.ndarray | None = None  # last pose backed by a mask (snapping reference)
    initialized = False
    prev_has_mask = False
    since_register = 0

    for i in range(n):
        if not initialized and not has_mask[i]:
            continue
        rgb, depth = episode.rgb(i), episode.depth_m(i)

        reason = None
        if not initialized:
            reason = "initial"
        else:
            T = model.track(rgb, depth, K)
            since_register += 1
            if has_mask[i]:
                if not prev_has_mask:
                    reason = "reappeared"
                elif cfg.reregister_every and since_register >= cfg.reregister_every:
                    reason = "periodic"
                else:
                    reason = tracking_lost(T, K, boxes[i], masks[i], depth, cfg)

        if reason is not None:
            T, score = model.register(rgb, depth, masks[i], K)
            if T_valid_last is not None:
                T = snap_to_symmetry(T, T_valid_last, symmetry_tfs)
            model.set_pose(T)
            initialized = True
            registered[i] = True
            scores[i] = score
            since_register = 0
            reasons[reason] = reasons.get(reason, 0) + 1

        T_all[i] = T
        valid[i] = has_mask[i]
        if has_mask[i]:
            T_valid_last = T
        prev_has_mask = bool(has_mask[i])

    stats = {
        "frames_with_mask": int(has_mask.sum()),
        "valid_frames": int(valid.sum()),
        "registrations": int(registered.sum()),
        "registration_reasons": reasons,
    }
    return T_all, valid, registered, scores, stats


def make_pose_estimator(cfg: PoseEstimatorCfg) -> PoseEstimator:
    """Construct the pose estimator named by ``cfg.name`` (backend modules are imported lazily)."""
    name = cfg.name.lower()
    if name == "foundationpose":
        from kinova_lfd.data_collection.human_demo.perception.pose.foundationpose import (
            FoundationPoseCfg,
            FoundationPoseEstimator,
        )

        assert isinstance(
            cfg, FoundationPoseCfg
        ), f"Pose estimator 'foundationpose' needs a FoundationPoseCfg, got {type(cfg).__name__}"
        return FoundationPoseEstimator(cfg)
    raise ValueError(f"Unsupported pose estimator {cfg.name!r}. Use one of {POSE_ESTIMATORS}.")
