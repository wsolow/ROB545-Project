"""Perception pipeline for human demonstrations: object segmentation, 6D object pose, and hand pose.

:class:`PerceptionPipeline` runs any non-empty subset of three stages on each selected episode of
each dataset root (a LeRobot dataset recorded by ``record_human_demo.py``):

1. ``segmentation``: segments every object (one SAM3 text concept each) and writes
   ``<root>/segmentation/episode_XXXXXX.npz``;
2. ``pose``: estimates each object's 6D pose in the camera frame (FoundationPose, from the episode's
   segmentation file) and writes ``<root>/6d_pose/episode_XXXXXX.npz``;
3. ``hand``: reconstructs the left and right hand (HaMeR, MANO) and writes
   ``<root>/hand_pose/episode_XXXXXX.npz``.

``<root>/6d_pose/meta.json`` records the objects and object configs, ``<root>/hand_pose/meta.json``
the hand config. With ``save_overlay``, check videos go to ``<root>/overlay/``:

- ``episode_XXXXXX_objects.mp4``: SAM masks, boxes, and pose axes (object stages);
- ``episode_XXXXXX_hands.mp4``: MANO hand masks and detection boxes (hand stage);
- ``episode_XXXXXX_combined.mp4``: both, when object and hand stages ran.

Every model runs on a local server (see ``segmentation/sam3.py``, ``pose/foundationpose.py``,
``hand/hamer.py``).

Written by Will Solow, 2026.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import (
    HAND_DIR,
    POSE_DIR,
    RawEpisode,
    hand_path,
    list_episodes,
    overlay_path,
    pose_path,
    raw_episode_dir,
    read_fps,
    segmentation_path,
)
from kinova_lfd.data_collection.human_demo.perception.hand import HamerCfg, HandEstimatorCfg, make_hand_estimator
from kinova_lfd.data_collection.human_demo.perception.pose import (
    FoundationPoseCfg,
    PoseEstimatorCfg,
    make_pose_estimator,
)
from kinova_lfd.data_collection.human_demo.perception.segmentation import Sam3Cfg, SegmenterCfg, make_segmenter
from kinova_lfd.data_collection.human_demo.perception.types import (
    MODELS_DIR,
    HandResult,
    ObjectSpec,
    PoseResult,
    SegmentationResult,
)

STAGES = ("segmentation", "pose", "hand")
OBJECT_STAGES = ("segmentation", "pose")
DEFAULT_MANO_DIR = MODELS_DIR / "hamer" / "data" / "mano"

_COLORS = [(255, 64, 160), (64, 200, 255), (255, 200, 0), (120, 255, 80), (200, 120, 255)]
"""RGB overlay color per object."""
_HAND_COLORS = {"left": (255, 140, 0), "right": (0, 200, 255)}
"""RGB overlay color per hand."""


class OverlayLayer(Protocol):
    """Draws one result type onto a frame of a check video."""

    def draw(self, img: np.ndarray, i: int) -> None:
        """Draw onto the BGR frame ``img`` (frame index ``i``) in place."""


class ObjectOverlay:
    """SAM mask tint + box per object, plus pose axes (solid if valid, thin if not)."""

    def __init__(self, seg_file: Path, pose_file: Path | None, objects: list[ObjectSpec], K: np.ndarray) -> None:
        import trimesh

        self.seg = SegmentationResult.load(seg_file)
        self.poses = PoseResult.load(pose_file) if pose_file is not None else None
        self.K = K
        self.axis_len = {
            o.name: 0.75 * float(np.max(trimesh.load(str(o.resolve_mesh()), force="mesh").extents)) for o in objects
        }

    def draw(self, img: np.ndarray, i: int) -> None:
        seg, poses = self.seg, self.poses
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
                    self._draw_axes(img, T, self.axis_len.get(name, 0.05), thick=2 if poses.valid[p, i] else 1)

    def _draw_axes(self, img: np.ndarray, T: np.ndarray, length: float, thick: int) -> None:
        pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=float)
        cam = (T[:3, :3] @ pts.T).T + T[:3, 3]
        if np.any(cam[:, 2] <= 1e-3):
            return
        uv = (self.K @ cam.T).T
        uv = (uv[:, :2] / uv[:, 2:3]).round().astype(int)
        for j, color in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):  # x red, y green, z blue (BGR)
            cv2.line(img, tuple(uv[0]), tuple(uv[j]), color, thick, cv2.LINE_AA)


class HandOverlay:
    """MANO hand mask tint + detection box and label per hand.

    The MANO meshes are posed for every detected frame up front; masks are rasterized per frame.
    """

    _mano_cache: dict[str, Any] = {}

    def __init__(self, hand_file: Path, mano_dir: Path = DEFAULT_MANO_DIR, alpha: float = 0.5) -> None:
        self.hands = HandResult.load(hand_file)
        self.alpha = alpha
        mano = self.mano_layer(mano_dir)
        self.faces = np.asarray(mano.faces, dtype=np.int64)
        # (side, frame) -> (778, 3) camera-frame vertices
        self.vertices: dict[tuple[int, int], np.ndarray] = {}
        for k in range(len(self.hands.hand_sides)):
            frames = np.flatnonzero(self.hands.detected[k] & np.isfinite(self.hands.cam_t[k]).all(axis=1))
            if frames.size:
                for i, verts in zip(frames, self.hand_vertices(mano, self.hands, k, frames)):
                    self.vertices[(k, int(i))] = verts

    @classmethod
    def mano_layer(cls, mano_dir: Path) -> Any:
        """Right-hand ``smplx.MANOLayer`` from ``mano_dir`` (``MANO_RIGHT.pkl``, chumpy-free), cached."""
        key = str(Path(mano_dir).resolve())
        if key not in cls._mano_cache:
            import smplx

            cls._mano_cache[key] = smplx.MANOLayer(model_path=key, is_rhand=True, use_pca=False)
        return cls._mano_cache[key]

    @staticmethod
    def hand_vertices(mano: Any, hands: HandResult, side: int, frames: np.ndarray) -> np.ndarray:
        """``(len(frames), 778, 3)`` camera-frame MANO vertices of one hand on ``frames``.

        Applies HaMeR's convention: the parameters describe a right hand, mirrored in x for the left.
        """
        import torch

        with torch.no_grad():
            out = mano(
                betas=torch.as_tensor(hands.betas[side, frames], dtype=torch.float32),
                global_orient=torch.as_tensor(hands.global_orient[side, frames], dtype=torch.float32)[:, None],
                hand_pose=torch.as_tensor(hands.hand_pose[side, frames], dtype=torch.float32),
            )
        verts = out.vertices.numpy().astype(np.float64)
        if hands.hand_sides[side] == "left":
            verts[..., 0] *= -1.0
        return verts + hands.cam_t[side, frames][:, None]

    def mask(self, side: int, i: int, image_hw: tuple[int, int]) -> np.ndarray | None:
        """``(H, W)`` bool mask of the projected MANO mesh, or ``None`` if the hand is not on frame ``i``."""
        verts = self.vertices.get((side, i))
        if verts is None or np.any(verts[:, 2] <= 1e-3):
            return None
        uv = verts @ self.hands.intrinsic_k.T
        uv = uv[:, :2] / uv[:, 2:3]
        tris = np.round(uv[self.faces]).astype(np.int32)  # (F, 3, 2)
        mask = np.zeros(image_hw, dtype=np.uint8)
        # One fill per triangle: a single fillPoly over all triangles cancels out where the front and
        # back surfaces of the mesh overlap.
        for tri in tris:
            cv2.fillConvexPoly(mask, tri, 1)
        return mask.astype(bool)

    def draw(self, img: np.ndarray, i: int) -> None:
        hands = self.hands
        for k, side in enumerate(hands.hand_sides):
            if not hands.detected[k, i]:
                continue
            color = _HAND_COLORS.get(side, (255, 255, 255))[::-1]
            mask = self.mask(k, i, img.shape[:2])
            if mask is not None and mask.any():
                img[mask] = ((1 - self.alpha) * img[mask] + self.alpha * np.array(color)).astype(np.uint8)
            x1, y1, x2, y2 = hands.boxes[k, i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label = f"{side} {hands.det_scores[k, i]:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def write_overlay_videos(episode: RawEpisode, videos: dict[Path, list[OverlayLayer]], fps: float) -> None:
    """Write one check video per entry of ``videos``, each frame drawn by its layers in order.

    Frames are read once and shared by all videos.
    """
    if not videos:
        return
    h, w = episode.image_size()
    writers = {}
    try:
        for path in videos:
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[path] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(len(episode)):
            frame = cv2.cvtColor(episode.rgb(i), cv2.COLOR_RGB2BGR)
            for path, layers in videos.items():
                img = frame.copy()
                for layer in layers:
                    layer.draw(img, i)
                cv2.putText(img, f"frame {i}", (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                writers[path].write(img)
    finally:
        for writer in writers.values():
            writer.release()


@dataclass
class PerceptionCfg:
    """Settings for :class:`PerceptionPipeline`."""

    stages: tuple[str, ...] = STAGES
    """Stages to run, any non-empty subset of ``STAGES``."""
    objects: list[ObjectSpec] = field(default_factory=list)
    """Objects to segment / track (required by the ``segmentation`` and ``pose`` stages)."""
    segmenter: SegmenterCfg = field(default_factory=Sam3Cfg)
    pose: PoseEstimatorCfg = field(default_factory=FoundationPoseCfg)
    hand: HandEstimatorCfg = field(default_factory=HamerCfg)
    save_overlay: bool = False
    """Write check videos to ``<root>/overlay/`` per episode."""
    mano_dir: Path = DEFAULT_MANO_DIR
    """Folder with ``MANO_RIGHT.pkl`` for the hand overlay (the HaMeR server makes it chumpy-free)."""
    hand_mask_alpha: float = 0.5
    """Opacity of the hand mask tint in the overlay."""

    @property
    def object_stages(self) -> bool:
        return any(s in self.stages for s in OBJECT_STAGES)


class PerceptionPipeline:
    """Segment and track objects and / or reconstruct hands through human-demo episodes."""

    def __init__(self, cfg: PerceptionCfg) -> None:
        unknown = sorted(set(cfg.stages) - set(STAGES))
        if not cfg.stages or unknown:
            raise ValueError(f"PerceptionCfg.stages must be a non-empty subset of {STAGES}, got {cfg.stages}.")
        if cfg.object_stages and not cfg.objects:
            raise ValueError(f"PerceptionCfg.objects is empty but object stages {OBJECT_STAGES} were requested.")
        self.cfg = cfg
        self.segmenter = make_segmenter(cfg.segmenter) if "segmentation" in cfg.stages else None
        self.pose_estimator = make_pose_estimator(cfg.pose) if "pose" in cfg.stages else None
        self.hand_estimator = make_hand_estimator(cfg.hand) if "hand" in cfg.stages else None

    def check_ready(self) -> None:
        """Fail fast if a needed server is not running."""
        for stage in (self.segmenter, self.pose_estimator, self.hand_estimator):
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
        """Run the configured stages on one episode, then optionally write the check videos."""
        ep_dir = raw_episode_dir(root, episode_index)
        seg_file, pose_file = segmentation_path(root, episode_index), pose_path(root, episode_index)
        hand_file = hand_path(root, episode_index)
        summary: dict[str, Any] = {}

        if self.segmenter is not None:
            summary["segmentation"] = self.segmenter.segment_episode(ep_dir, self.cfg.objects, seg_file)
            print(f"  segmentation: {summary['segmentation']}")

        if self.pose_estimator is not None:
            if not seg_file.exists():
                raise FileNotFoundError(f"{seg_file} missing; run the segmentation stage first.")
            summary["pose"] = self.pose_estimator.estimate_episode(ep_dir, seg_file, self.cfg.objects, pose_file, fps)
            print(f"  pose: {summary['pose']}")

        if self.hand_estimator is not None:
            summary["hand"] = self.hand_estimator.estimate_episode(ep_dir, hand_file, fps)
            print(f"  hand: {summary['hand']}")

        if self.cfg.save_overlay:
            summary["overlay"] = self.render_overlays(root, episode_index, fps)
            for path in summary["overlay"]:
                print(f"  overlay: {path}")
        return summary

    def render_overlays(self, root: Path, episode_index: int, fps: float) -> list[str]:
        """Write the objects / hands / combined check videos for the stages that have results on disk."""
        episode = RawEpisode(raw_episode_dir(root, episode_index))
        seg_file, pose_file = segmentation_path(root, episode_index), pose_path(root, episode_index)
        hand_file = hand_path(root, episode_index)

        object_layer = hand_layer = None
        if self.cfg.object_stages and seg_file.exists():
            object_layer = ObjectOverlay(
                seg_file, pose_file if pose_file.exists() else None, self.cfg.objects, episode.intrinsic_k
            )
        if "hand" in self.cfg.stages and hand_file.exists():
            hand_layer = HandOverlay(hand_file, self.cfg.mano_dir, self.cfg.hand_mask_alpha)
            if hand_layer.hands.num_frames != len(episode):
                raise ValueError(f"{hand_file} has {hand_layer.hands.num_frames} frames, episode has {len(episode)}.")

        videos: dict[Path, list[OverlayLayer]] = {}
        if object_layer is not None:
            videos[overlay_path(root, episode_index, "objects")] = [object_layer]
        if hand_layer is not None:
            videos[overlay_path(root, episode_index, "hands")] = [hand_layer]
        if object_layer is not None and hand_layer is not None:
            videos[overlay_path(root, episode_index, "combined")] = [object_layer, hand_layer]
        write_overlay_videos(episode, videos, fps)
        return [str(path) for path in videos]

    def _write_meta(self, root: Path) -> None:
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.cfg.object_stages:
            meta = {
                "created": created,
                "frame": "camera (T_C_O: object pose in the camera optical frame, meters)",
                "objects": [o.to_dict() | {"resolved_mesh": str(o.resolve_mesh())} for o in self.cfg.objects],
                "segmenter": asdict(self.cfg.segmenter),
                "pose": asdict(self.cfg.pose),
            }
            self._write_json(root / POSE_DIR / "meta.json", meta)
        if "hand" in self.cfg.stages:
            meta = {
                "created": created,
                "frame": "camera (keypoints_3d / cam_t in the camera optical frame, meters)",
                "hand_sides": ["left", "right"],
                "hand": asdict(self.cfg.hand),
            }
            self._write_json(root / HAND_DIR / "meta.json", meta)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
