"""Read human-demo episodes from the raw folder of a LeRobot dataset.

Layout written by ``record_human_demo.py``::

    <dataset_root>/meta/info.json
    <dataset_root>/raw/episode_000000/rgb/00000.png     # 8-bit color
    <dataset_root>/raw/episode_000000/depth/00000.png   # uint16 millimeters, aligned to color
    <dataset_root>/raw/episode_000000/cam_K.txt         # 3x3 intrinsics

Perception outputs live next to ``raw/``: ``<dataset_root>/segmentation/``, ``<dataset_root>/6d_pose/`` and
``<dataset_root>/hand_pose/``.

Written by Will Solow, 2026.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

SEGMENTATION_DIR = "segmentation"
POSE_DIR = "6d_pose"
HAND_DIR = "hand_pose"
_EPISODE_RE = re.compile(r"^episode_(\d+)$")


def episode_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def raw_episode_dir(dataset_root: str | Path, episode_index: int) -> Path:
    return Path(dataset_root) / "raw" / episode_name(episode_index)


def segmentation_path(dataset_root: str | Path, episode_index: int) -> Path:
    return Path(dataset_root) / SEGMENTATION_DIR / f"{episode_name(episode_index)}.npz"


def pose_path(dataset_root: str | Path, episode_index: int) -> Path:
    return Path(dataset_root) / POSE_DIR / f"{episode_name(episode_index)}.npz"


def hand_path(dataset_root: str | Path, episode_index: int) -> Path:
    return Path(dataset_root) / HAND_DIR / f"{episode_name(episode_index)}.npz"


def list_episodes(dataset_root: str | Path) -> list[int]:
    """Episode indices that have a complete raw folder (``rgb/``, ``depth/``, ``cam_K.txt``)."""
    raw = Path(dataset_root) / "raw"
    if not raw.is_dir():
        raise FileNotFoundError(f"{raw} does not exist; record with record_human_demo.py (raw output on).")
    episodes = []
    for child in raw.iterdir():
        match = _EPISODE_RE.match(child.name)
        if match and (child / "rgb").is_dir() and (child / "depth").is_dir() and (child / "cam_K.txt").is_file():
            episodes.append(int(match.group(1)))
    return sorted(episodes)


def read_fps(dataset_root: str | Path) -> float:
    with open(Path(dataset_root) / "meta" / "info.json") as f:
        return float(json.load(f)["fps"])


class RawEpisode:
    """Lazy access to one raw episode folder (frames are read from disk on demand)."""

    def __init__(self, episode_dir: str | Path) -> None:
        self.dir = Path(episode_dir)
        self.rgb_dir = self.dir / "rgb"
        self.depth_dir = self.dir / "depth"
        self.rgb_files = sorted(self.rgb_dir.glob("*.png"))
        depth_files = sorted(self.depth_dir.glob("*.png"))
        if not self.rgb_files:
            raise FileNotFoundError(f"No RGB frames in {self.rgb_dir}.")
        if [p.name for p in depth_files] != [p.name for p in self.rgb_files]:
            raise ValueError(
                f"{self.dir}: rgb/ and depth/ frames do not match ({len(self.rgb_files)} vs {len(depth_files)})."
            )
        self.depth_files = depth_files
        self.intrinsic_k = np.loadtxt(self.dir / "cam_K.txt").reshape(3, 3)
        match = _EPISODE_RE.match(self.dir.name)
        self.episode_index = int(match.group(1)) if match else -1

    def __len__(self) -> int:
        return len(self.rgb_files)

    @property
    def frame_index(self) -> np.ndarray:
        return np.array([int(p.stem) for p in self.rgb_files], dtype=np.int64)

    def image_size(self) -> tuple[int, int]:
        """``(H, W)`` of the frames."""
        h, w = self.rgb(0).shape[:2]
        return h, w

    def rgb(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB."""
        bgr = cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read {self.rgb_files[i]}.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth_m(self, i: int) -> np.ndarray:
        """(H, W) float32 depth in meters (0 = invalid)."""
        depth = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise OSError(f"Could not read uint16 depth from {self.depth_files[i]}.")
        return depth.astype(np.float32) / 1000.0

    def frames(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield ``(i, rgb, depth_m)`` for every frame."""
        for i in range(len(self)):
            yield i, self.rgb(i), self.depth_m(i)
