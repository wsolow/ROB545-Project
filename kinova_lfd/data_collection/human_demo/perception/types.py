"""Shared types for the human-demo object perception pipeline.

- :class:`ObjectSpec`: one tracked object (name, SAM3 text prompt, mesh or box size, symmetry).
- :class:`SegmentationResult`: per-episode masks / boxes / scores for every object.
- :class:`PoseResult`: per-episode 6D pose trajectory ``T_C_O`` (camera frame) for every object.
- :class:`HandResult`: per-episode MANO reconstruction of the left and right hand (camera frame).

Every result type is saved as one ``.npz`` per episode.

Written by Will Solow, 2026.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache"
"""``kinova_lfd/.cache``: model weights and generated meshes (git-ignored)."""
MODELS_DIR = CACHE_DIR / "models"
MESH_CACHE_DIR = CACHE_DIR / "meshes"

SYMMETRIES = ("none", "box", "cube")
HAND_SIDES = ("left", "right")
"""Order of the hand axis in :class:`HandResult`."""


@dataclass
class ObjectSpec:
    """One object to segment and track.

    Exactly one of ``mesh_path`` / ``box_size_m`` must be set. ``box_size_m`` objects get a
    generated box mesh centered at the origin, in meters.
    """

    name: str
    """Short identifier used in the output files (e.g. ``pink_cube``)."""
    prompt: str
    """SAM3 text prompt (concept), e.g. ``"pink cube"``."""
    mesh_path: str | None = None
    box_size_m: tuple[float, float, float] | None = None
    symmetry: str | None = None
    """``"none"``, ``"box"`` (180 deg about each axis) or ``"cube"`` (24 rotations). Defaults to
    ``"cube"`` / ``"box"`` for box objects and ``"none"`` for meshes."""
    color_rgb: tuple[int, int, int] | None = None
    """Vertex color for generated box meshes (helps FoundationPose's render-and-compare)."""

    def __post_init__(self) -> None:
        if (self.mesh_path is None) == (self.box_size_m is None):
            raise ValueError(f"Object {self.name!r}: set exactly one of mesh_path / box_size_m.")
        if self.box_size_m is not None:
            size = tuple(float(s) for s in np.broadcast_to(np.asarray(self.box_size_m, dtype=float), (3,)))
            self.box_size_m = size  # type: ignore[assignment]
            if self.symmetry is None:
                self.symmetry = "cube" if np.allclose(size, size[0]) else "box"
        if self.symmetry is None:
            self.symmetry = "none"
        if self.symmetry not in SYMMETRIES:
            raise ValueError(f"Object {self.name!r}: symmetry must be one of {SYMMETRIES}, got {self.symmetry!r}.")

    def resolve_mesh(self, cache_dir: Path = MESH_CACHE_DIR) -> Path:
        """Return the mesh path, generating (and caching) the box mesh for box objects."""
        if self.mesh_path is not None:
            path = Path(self.mesh_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Object {self.name!r}: mesh {path} does not exist.")
            return path

        import trimesh

        assert self.box_size_m is not None
        sx, sy, sz = self.box_size_m
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = (cache_dir / f"box_{sx:.4f}x{sy:.4f}x{sz:.4f}_{self.color_tag}.obj").resolve()
        if not path.exists():
            mesh = trimesh.creation.box(extents=self.box_size_m)
            # FoundationPose samples model points from the vertices, so subdivide the 8-vertex box.
            vertices, faces = trimesh.remesh.subdivide_to_size(
                mesh.vertices, mesh.faces, max_edge=min(self.box_size_m) / 8
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            if self.color_rgb is not None:
                mesh.visual.vertex_colors = np.array([*self.color_rgb, 255], dtype=np.uint8)
            mesh.export(path)
        return path

    @property
    def color_tag(self) -> str:
        return "nocolor" if self.color_rgb is None else "{:02x}{:02x}{:02x}".format(*self.color_rgb)

    def symmetry_tfs(self) -> np.ndarray:
        """``(K, 4, 4)`` object-frame symmetry transforms (identity first)."""
        return symmetry_transforms(self.symmetry or "none")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "mesh_path": self.mesh_path,
            "box_size_m": list(self.box_size_m) if self.box_size_m is not None else None,
            "symmetry": self.symmetry,
            "color_rgb": list(self.color_rgb) if self.color_rgb is not None else None,
        }


def load_object_specs(path: str | Path) -> list[ObjectSpec]:
    """Load objects from a YAML file with a top-level ``objects:`` list."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("objects")
    if not entries:
        raise ValueError(f"{path}: expected a non-empty top-level 'objects:' list.")
    specs = [ObjectSpec(**entry) for entry in entries]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: object names must be unique, got {names}.")
    return specs


def symmetry_transforms(kind: str) -> np.ndarray:
    """Rotation symmetries of an origin-centered box, as ``(K, 4, 4)`` transforms.

    ``"cube"``: the 24 proper rotations of a cube (signed permutation matrices with det +1).
    ``"box"``: identity plus 180 deg about x, y and z. ``"none"``: identity only.
    """
    if kind == "none":
        rots = [np.eye(3)]
    elif kind == "box":
        rots = [np.diag(d) for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    elif kind == "cube":
        rots = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                R = np.zeros((3, 3))
                R[range(3), perm] = signs
                if np.isclose(np.linalg.det(R), 1.0):
                    rots.append(R)
        rots.sort(key=lambda R: not np.allclose(R, np.eye(3)))  # identity first
    else:
        raise ValueError(f"Unknown symmetry {kind!r}.")
    tfs = np.tile(np.eye(4), (len(rots), 1, 1))
    tfs[:, :3, :3] = np.stack(rots)
    return tfs


@dataclass
class SegmentationResult:
    """Segmentation of ``M`` objects over ``N`` frames of one episode."""

    object_names: list[str]
    prompts: list[str]
    masks: np.ndarray
    """(M, N, H, W) bool."""
    boxes: np.ndarray
    """(M, N, 4) float32 pixel ``x1, y1, x2, y2``; NaN where the object is not detected."""
    scores: np.ndarray
    """(M, N) float32 SAM3 score; 0 where the object is not detected."""
    instance_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    """(M,) SAM3 track id of the instance kept for each object (-1 if none)."""

    @property
    def num_frames(self) -> int:
        return int(self.masks.shape[1])

    def detected(self) -> np.ndarray:
        """(M, N) bool: frames where each object has a non-empty mask."""
        return self.masks.any(axis=(2, 3))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m, n, h, w = self.masks.shape
        np.savez_compressed(
            path,
            object_names=np.asarray(self.object_names),
            prompts=np.asarray(self.prompts),
            masks_packed=np.packbits(self.masks.reshape(m, n, -1), axis=-1),
            mask_shape=np.asarray([m, n, h, w]),
            boxes=self.boxes.astype(np.float32),
            scores=self.scores.astype(np.float32),
            instance_ids=np.asarray(self.instance_ids, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SegmentationResult":
        with np.load(path) as data:
            m, n, h, w = (int(v) for v in data["mask_shape"])
            masks = np.unpackbits(data["masks_packed"], axis=-1, count=h * w).reshape(m, n, h, w).astype(bool)
            return cls(
                object_names=[str(s) for s in data["object_names"]],
                prompts=[str(s) for s in data["prompts"]],
                masks=masks,
                boxes=data["boxes"],
                scores=data["scores"],
                instance_ids=data["instance_ids"],
            )


@dataclass
class PoseResult:
    """6D pose trajectories of ``M`` objects over ``N`` frames of one episode, in the camera frame."""

    object_names: list[str]
    T_C_O: np.ndarray
    """(M, N, 4, 4) float64 object pose in the camera frame; NaN before the first registration."""
    valid: np.ndarray
    """(M, N) bool: pose is backed by a SAM3 mask on that frame."""
    registered: np.ndarray
    """(M, N) bool: pose came from a (re-)registration rather than frame-to-frame tracking."""
    scores: np.ndarray
    """(M, N) float32 registration score on registered frames, NaN otherwise."""
    frame_index: np.ndarray
    """(N,) int64."""
    timestamps: np.ndarray
    """(N,) float64 seconds (``frame_index / fps``)."""
    intrinsic_k: np.ndarray
    """(3, 3) float64."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            object_names=np.asarray(self.object_names),
            T_C_O=self.T_C_O.astype(np.float64),
            valid=self.valid.astype(bool),
            registered=self.registered.astype(bool),
            scores=self.scores.astype(np.float32),
            frame_index=self.frame_index.astype(np.int64),
            timestamps=self.timestamps.astype(np.float64),
            intrinsic_k=self.intrinsic_k.astype(np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseResult":
        with np.load(path) as data:
            return cls(
                object_names=[str(s) for s in data["object_names"]],
                **{
                    key: data[key]
                    for key in ("T_C_O", "valid", "registered", "scores", "frame_index", "timestamps", "intrinsic_k")
                },
            )


@dataclass
class HandResult:
    """MANO reconstructions of the left and right hand over ``N`` frames of one episode.

    The first axis follows ``hand_sides`` (``HAND_SIDES``). Frames where a hand is not detected are
    NaN (``detected`` is False). The MANO parameters are in HaMeR's convention: they always describe
    a *right* MANO hand, and for the left hand they describe the horizontally mirrored hand (mirror
    the x axis of its MANO vertices / joints before adding ``cam_t``). ``keypoints_3d`` /
    ``keypoints_2d`` already have that mirroring applied and are the ones to use downstream.
    """

    hand_sides: list[str]
    detected: np.ndarray
    """(2, N) bool."""
    boxes: np.ndarray
    """(2, N, 4) float32 pixel ``x1, y1, x2, y2`` of the hand detection; NaN where undetected."""
    det_scores: np.ndarray
    """(2, N) float32 detector score; 0 where undetected."""
    global_orient: np.ndarray
    """(2, N, 3, 3) float32 MANO root rotation (camera frame, see class docstring for the left hand)."""
    hand_pose: np.ndarray
    """(2, N, 15, 3, 3) float32 MANO finger joint rotations."""
    betas: np.ndarray
    """(2, N, 10) float32 MANO shape."""
    cam_t: np.ndarray
    """(2, N, 3) float64 translation (m) from the MANO frame to the camera frame."""
    keypoints_3d: np.ndarray
    """(2, N, 21, 3) float64 hand joints (OpenPose order) in the camera frame, meters."""
    keypoints_2d: np.ndarray
    """(2, N, 21, 2) float32 ``keypoints_3d`` projected with ``intrinsic_k``, pixels."""
    frame_index: np.ndarray
    """(N,) int64."""
    timestamps: np.ndarray
    """(N,) float64 seconds (``frame_index / fps``)."""
    intrinsic_k: np.ndarray
    """(3, 3) float64."""
    detector: str = ""
    """Hand detector the server ran (e.g. ``yolov8``, ``vitpose``)."""

    _ARRAYS = (
        "detected",
        "boxes",
        "det_scores",
        "global_orient",
        "hand_pose",
        "betas",
        "cam_t",
        "keypoints_3d",
        "keypoints_2d",
        "frame_index",
        "timestamps",
        "intrinsic_k",
    )

    @property
    def num_frames(self) -> int:
        return int(self.detected.shape[1])

    def side(self, name: str) -> int:
        """Index of ``"left"`` / ``"right"`` along the hand axis."""
        return self.hand_sides.index(name)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            hand_sides=np.asarray(self.hand_sides),
            detector=np.asarray(self.detector),
            detected=self.detected.astype(bool),
            boxes=self.boxes.astype(np.float32),
            det_scores=self.det_scores.astype(np.float32),
            global_orient=self.global_orient.astype(np.float32),
            hand_pose=self.hand_pose.astype(np.float32),
            betas=self.betas.astype(np.float32),
            cam_t=self.cam_t.astype(np.float64),
            keypoints_3d=self.keypoints_3d.astype(np.float64),
            keypoints_2d=self.keypoints_2d.astype(np.float32),
            frame_index=self.frame_index.astype(np.int64),
            timestamps=self.timestamps.astype(np.float64),
            intrinsic_k=self.intrinsic_k.astype(np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HandResult":
        with np.load(path) as data:
            return cls(
                hand_sides=[str(s) for s in data["hand_sides"]],
                detector=str(data["detector"]),
                **{key: data[key] for key in cls._ARRAYS},
            )
