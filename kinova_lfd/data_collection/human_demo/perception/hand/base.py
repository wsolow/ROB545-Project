"""Hand pose estimation interface, config, and factories.

Two layers, like ``pose/``:

- :class:`HandEstimator` (client-facing): reconstructs the left and right hand through one episode
  and writes a :class:`HandResult` ``.npz``. Backends are chosen by ``HandEstimatorCfg.name``.
- :class:`HandDetector` (in-process, used inside a hand server): finds hand boxes and their
  handedness in one frame. A server picks its detector at startup with :func:`make_hand_detector`;
  :func:`best_per_side` keeps the most confident detection of each hand.

Written by Will Solow, 2026.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.data_collection.human_demo.perception.types import HAND_SIDES

HAND_ESTIMATORS = ("hamer",)
HAND_DETECTORS = ("vitpose", "yolov8")
"""``vitpose``: HaMeR's original ViTDet person detector + ViTPose whole-body keypoints (heavy).
``yolov8``: WiLoR's YOLOv8 hand detector (lightweight)."""


@dataclass
class HandEstimatorCfg:
    """Settings shared by every hand pose estimator."""

    name: str = ""
    """Backend to create, one of ``HAND_ESTIMATORS``."""


class HandEstimator(ABC):
    """Reconstruct the left and right hand through a whole episode."""

    def __init__(self, cfg: HandEstimatorCfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def estimate_episode(self, episode_dir: Path, out_path: Path, fps: float) -> dict[str, Any]:
        """Reconstruct the hands in the raw episode at ``episode_dir``; save a ``HandResult`` to ``out_path``.

        Returns:
            A summary dict (frames, per-hand detected frame counts, seconds, ...).

        """
        raise NotImplementedError

    def check_ready(self) -> dict[str, Any]:
        """Raise if the backend is unavailable (e.g. its server is down); return backend info."""
        return {}

    def close(self) -> None:
        """Release resources."""


@dataclass
class HandDetection:
    """One hand found in one frame."""

    box: np.ndarray
    """(4,) pixel ``x1, y1, x2, y2``."""
    score: float
    is_right: bool


class HandDetector(ABC):
    """In-process hand detector (boxes + handedness) for single RGB frames."""

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        """Return every hand found in the ``(H, W, 3)`` uint8 RGB frame."""


def best_per_side(detections: Sequence[HandDetection], min_score: float = 0.0) -> list[HandDetection | None]:
    """Keep the highest-scoring detection of each hand, indexed like ``HAND_SIDES`` (``None`` if absent)."""
    best: list[HandDetection | None] = [None] * len(HAND_SIDES)
    for det in detections:
        if det.score < min_score:
            continue
        k = HAND_SIDES.index("right" if det.is_right else "left")
        if best[k] is None or det.score > best[k].score:  # type: ignore[union-attr]
            best[k] = det
    return best


def make_hand_detector(name: str, models_dir: Path, device: str = "cuda") -> HandDetector:
    """Construct the detector named ``name`` with weights under ``models_dir`` (imported lazily)."""
    name = name.lower()
    if name == "yolov8":
        from kinova_lfd.data_collection.human_demo.perception.hand.yolov8 import Yolov8HandDetector

        return Yolov8HandDetector(models_dir=models_dir, device=device)
    if name == "vitpose":
        from kinova_lfd.data_collection.human_demo.perception.hand.hamer import VitPoseHandDetector

        return VitPoseHandDetector(models_dir=models_dir, device=device)
    raise ValueError(f"Unsupported hand detector {name!r}. Use one of {HAND_DETECTORS}.")


def make_hand_estimator(cfg: HandEstimatorCfg) -> HandEstimator:
    """Construct the hand estimator named by ``cfg.name`` (backend modules are imported lazily)."""
    name = cfg.name.lower()
    if name == "hamer":
        from kinova_lfd.data_collection.human_demo.perception.hand.hamer import HamerCfg, HamerEstimator

        assert isinstance(cfg, HamerCfg), f"Hand estimator 'hamer' needs a HamerCfg, got {type(cfg).__name__}"
        return HamerEstimator(cfg)
    raise ValueError(f"Unsupported hand estimator {cfg.name!r}. Use one of {HAND_ESTIMATORS}.")
