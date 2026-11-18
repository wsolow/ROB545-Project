"""Object tracking pipeline for human demonstrations: SAM3 segmentation -> FoundationPose 6D pose.

For each dataset root (a LeRobot dataset recorded by ``record_human_demo.py``) and each selected
episode, the pipeline:

1. segments every object (one text concept each) and writes ``<root>/segmentation/episode_XXXXXX.npz``;
2. estimates each object's 6D pose in the camera frame and writes ``<root>/6d_pose/episode_XXXXXX.npz``;
3. optionally writes a check video ``<root>/6d_pose/episode_XXXXXX_overlay.mp4``.

``<root>/6d_pose/meta.json`` records the objects and configs used. Both model stages run on
local servers (see ``segmentation/sam3.py`` and ``pose/foundationpose.py``).

Written by Will Solow, 2026.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import (
    POSE_DIR,
    RawEpisode,
    list_episodes,
    pose_path,
    raw_episode_dir,
    read_fps,
    segmentation_path,
)
from kinova_lfd.data_collection.human_demo.perception.pose import (
    FoundationPoseCfg,
    PoseEstimatorCfg,
    make_pose_estimator,
)
from kinova_lfd.data_collection.human_demo.perception.segmentation import Sam3Cfg, SegmenterCfg, make_segmenter
from kinova_lfd.data_collection.human_demo.perception.types import ObjectSpec, PoseResult, SegmentationResult

DEFAULT_OBJECTS_YAML = Path(__file__).resolve().parent / "objects.yaml"
_COLORS = [(255, 64, 160), (64, 200, 255), (255, 200, 0), (120, 255, 80), (200, 120, 255)]


@dataclass
class ObjectTrackingCfg:
    """Settings for :class:`ObjectTrackingPipeline`."""

    objects: list[ObjectSpec] = field(default_factory=list)
    segmenter: SegmenterCfg = field(default_factory=Sam3Cfg)
    pose: PoseEstimatorCfg = field(default_factory=FoundationPoseCfg)
    stages: tuple[str, ...] = ("segmentation", "pose")
    """Stages to run."""
    overwrite: bool = False
    """Recompute outputs that already exist (otherwise they are skipped)."""
    save_overlay: bool = False
    """Write an ``_overlay.mp4`` check video per episode."""


class ObjectTrackingPipeline:
    """Segment and track objects through human-demo episodes."""

    def __init__(self, cfg: ObjectTrackingCfg) -> None:
        if not cfg.objects:
            raise ValueError("ObjectTrackingCfg.objects is empty.")
        self.cfg = cfg
        self.segmenter = make_segmenter(cfg.segmenter) if "segmentation" in cfg.stages else None
        self.pose_estimator = make_pose_estimator(cfg.pose) if "pose" in cfg.stages else None

    def check_ready(self) -> None:
        """Fail fast if a needed server is not running."""
        for stage in (self.segmenter, self.pose_estimator):
            if stage is not None:
                info = stage.check_ready()
                print(f"[Perception] {type(stage).__name__} ready: {info}")

    def run(self, dataset_roots: list[str | Path], episodes: list[int] | None = None) -> dict[str, Any]:
        """Process ``episodes`` (default: all with a raw folder) of every dataset root."""
        self.check_ready()
        report: dict[str, Any] = {}
        for root in map(Path, dataset_roots):
            available = list_episodes(root)
            selected = available if episodes is None else [e for e in episodes if e in available]
            missing = [] if episodes is None else sorted(set(episodes) - set(available))
            if missing:
                print(f"[Perception] {root}: no raw folder for episodes {missing}; skipping them.")
            fps = read_fps(root)
            self._write_meta(root)
            report[str(root)] = {}
            for ep in selected:
                print(f"[Perception] `{root}` Episode {ep} ...")
                report[str(root)][ep] = self.run_episode(root, ep, fps)
        return report

    def run_episode(self, root: Path, episode_index: int, fps: float) -> dict[str, Any]:
        """Run the configured stages on one episode."""
        ep_dir = raw_episode_dir(root, episode_index)
        seg_file, pose_file = segmentation_path(root, episode_index), pose_path(root, episode_index)
        summary: dict[str, Any] = {}

        if self.segmenter is not None:
            if seg_file.exists() and not self.cfg.overwrite:
                summary["segmentation"] = "exists (skipped)"
            else:
                summary["segmentation"] = self.segmenter.segment_episode(ep_dir, self.cfg.objects, seg_file)
            print(f"  segmentation: {summary['segmentation']}")

        if self.pose_estimator is not None:
            if not seg_file.exists():
                raise FileNotFoundError(f"{seg_file} missing; run the segmentation stage first.")
            if pose_file.exists() and not self.cfg.overwrite:
                summary["pose"] = "exists (skipped)"
            else:
                summary["pose"] = self.pose_estimator.estimate_episode(
                    ep_dir, seg_file, self.cfg.objects, pose_file, fps
                )
            print(f"  pose: {summary['pose']}")

        if self.cfg.save_overlay and seg_file.exists():
            out = pose_file.with_name(pose_file.stem + "_overlay.mp4")
            self.render_overlay(ep_dir, seg_file, pose_file if pose_file.exists() else None, self.cfg.objects, out, fps)
            summary["overlay"] = str(out)
            print(f"  overlay: {out}")
        return summary

    def _write_meta(self, root: Path) -> None:
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame": "camera (T_C_O: object pose in the camera optical frame, meters)",
            "objects": [o.to_dict() | {"resolved_mesh": str(o.resolve_mesh())} for o in self.cfg.objects],
            "segmenter": asdict(self.cfg.segmenter),
            "pose": asdict(self.cfg.pose),
        }
        out = root / POSE_DIR / "meta.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2))

    def render_overlay(
        self,
        episode_dir: Path,
        seg_file: Path,
        pose_file: Path | None,
        objects: list[ObjectSpec],
        out_path: Path,
        fps: float,
    ) -> None:
        """Write a check video: SAM mask tint + box per object, plus pose axes (solid if valid, thin if not)."""
        import trimesh

        episode = RawEpisode(episode_dir)
        seg = SegmentationResult.load(seg_file)
        poses = PoseResult.load(pose_file) if pose_file is not None else None
        K = episode.intrinsic_k
        axis_len = {
            o.name: 0.75 * float(np.max(trimesh.load(str(o.resolve_mesh()), force="mesh").extents)) for o in objects
        }
        h, w = episode.image_size()
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for i in range(len(episode)):
                img = cv2.cvtColor(episode.rgb(i), cv2.COLOR_RGB2BGR)
                for k, name in enumerate(seg.object_names):
                    color = _COLORS[k % len(_COLORS)][::-1]
                    mask = seg.masks[k, i]
                    if mask.any():
                        img[mask] = (0.5 * img[mask] + 0.5 * np.array(color)).astype(np.uint8)
                        x1, y1, x2, y2 = seg.boxes[k, i].astype(int)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(img, name, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    if poses is not None and name in poses.object_names:
                        p = poses.object_names.index(name)
                        T = poses.T_C_O[p, i]
                        if np.all(np.isfinite(T)):
                            self._draw_axes(img, T, K, axis_len.get(name, 0.05), thick=2 if poses.valid[p, i] else 1)
                cv2.putText(img, f"frame {i}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writer.write(img)
        finally:
            writer.release()

    def _draw_axes(self, img: np.ndarray, T: np.ndarray, K: np.ndarray, length: float, thick: int) -> None:
        pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=float)
        cam = (T[:3, :3] @ pts.T).T + T[:3, 3]
        if np.any(cam[:, 2] <= 1e-3):
            return
        uv = (K @ cam.T).T
        uv = (uv[:, :2] / uv[:, 2:3]).round().astype(int)
        for j, color in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):  # x red, y green, z blue (BGR)
            cv2.line(img, tuple(uv[0]), tuple(uv[j]), color, thick, cv2.LINE_AA)
