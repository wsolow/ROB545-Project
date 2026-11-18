"""FoundationPose model-based 6D pose estimation: model wrapper, FastAPI server, and client.

Start the server once (it keeps the scorer / refiner weights on the GPU)::

    python -m kinova_lfd.data_collection.human_demo.perception.pose.foundationpose --port 8002

Then ``FoundationPoseEstimator`` (``FoundationPoseCfg(server_url=...)``) sends it episodes plus their
SAM3 segmentation. FoundationPose's source lives in ``third_party/FoundationPose`` and loads its
weights from ``third_party/FoundationPose/weights/``; the server links that folder to
``kinova_lfd/.cache/models/foundationpose`` (``2023-10-28-18-33-37`` refiner, ``2024-01-11-20-02-45`` scorer).

Written by Will Solow, 2026.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import RawEpisode
from kinova_lfd.data_collection.human_demo.perception.pose.base import (
    PoseEstimator,
    PoseEstimatorCfg,
    PoseModel,
    TrackingCfg,
    estimate_object_track,
)
from kinova_lfd.data_collection.human_demo.perception.server_utils import (
    ServerClient,
    create_model_app,
    run_locked,
    serve,
)
from kinova_lfd.data_collection.human_demo.perception.types import (
    CACHE_DIR,
    MODELS_DIR,
    ObjectSpec,
    PoseResult,
    SegmentationResult,
)

DEFAULT_PORT = 8002
REPO_ROOT = Path(__file__).resolve().parents[5]
FOUNDATIONPOSE_DIR = REPO_ROOT / "third_party" / "FoundationPose"
DEFAULT_WEIGHTS_DIR = MODELS_DIR / "foundationpose"
REQUIRED_RUNS = ("2023-10-28-18-33-37", "2024-01-11-20-02-45")


@dataclass
class FoundationPoseCfg(PoseEstimatorCfg):
    """Client settings for the FoundationPose server."""

    name: str = "foundationpose"
    server_url: str = f"http://127.0.0.1:{DEFAULT_PORT}"
    timeout_s: float = 3600.0
    est_refine_iter: int = 5
    """Refiner iterations when registering."""
    track_refine_iter: int = 2
    """Refiner iterations when tracking."""
    tracking: TrackingCfg = field(default_factory=TrackingCfg)


@dataclass
class FoundationPoseServerCfg:
    """Model settings used when the server starts."""

    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    debug_dir: Path = CACHE_DIR / "foundationpose_debug"


def _link_weights(weights_dir: Path) -> None:
    """Point ``third_party/FoundationPose/weights`` (hard-coded by FoundationPose) at ``weights_dir``."""
    missing = [run for run in REQUIRED_RUNS if not (weights_dir / run / "model_best.pth").exists()]
    if missing:
        raise FileNotFoundError(f"FoundationPose weights {missing} not found under {weights_dir}.")
    link = FOUNDATIONPOSE_DIR / "weights"
    if link.is_symlink() or not link.exists():
        if link.is_symlink() and link.resolve() == weights_dir.resolve():
            return
        if link.is_symlink():
            link.unlink()
        link.symlink_to(weights_dir.resolve(), target_is_directory=True)
    elif link.resolve() != weights_dir.resolve():
        raise RuntimeError(f"{link} is a real folder; move its contents to {weights_dir} and delete it.")


class FoundationPoseModel(PoseModel):
    """In-process FoundationPose with one shared scorer / refiner and one estimator per object."""

    def __init__(self, cfg: FoundationPoseServerCfg, est_refine_iter: int = 5, track_refine_iter: int = 2) -> None:
        if not FOUNDATIONPOSE_DIR.is_dir():
            raise FileNotFoundError(f"{FOUNDATIONPOSE_DIR} not found; clone NVlabs/FoundationPose into third_party/.")
        _link_weights(Path(cfg.weights_dir))
        if str(FOUNDATIONPOSE_DIR) not in sys.path:
            sys.path.insert(0, str(FOUNDATIONPOSE_DIR))
        import estimater  # noqa: PLC0415  (FoundationPose's top-level module)
        import nvdiffrast.torch as dr

        logging.getLogger().setLevel(logging.WARNING)  # FoundationPose logs every step at INFO
        self.cfg = cfg
        self._fp = estimater
        self.scorer = estimater.ScorePredictor()
        self.refiner = estimater.PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self._estimators: dict[tuple[str, bytes], Any] = {}
        self._est: Any = None
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    def set_object(self, mesh_path: Path, symmetry_tfs: np.ndarray) -> None:
        import trimesh

        key = (str(mesh_path), np.asarray(symmetry_tfs, dtype=np.float32).tobytes())
        if key not in self._estimators:
            mesh = trimesh.load(str(mesh_path), force="mesh")
            self._estimators[key] = self._fp.FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                symmetry_tfs=symmetry_tfs,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
                debug=0,
                debug_dir=str(self.cfg.debug_dir),
            )
        self._est = self._estimators[key]
        self._est.pose_last = None

    def register(
        self, rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._est.scores = None
        pose = self._est.register(
            K=K, rgb=rgb, depth=depth_m, ob_mask=mask.astype(bool), iteration=self.est_refine_iter
        )
        scores = getattr(self._est, "scores", None)
        score = float(scores[0]) if scores is not None and len(scores) else float("nan")
        return np.asarray(pose, dtype=np.float64).reshape(4, 4), score

    def track(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
        pose = self._est.track_one(rgb=rgb, depth=depth_m, K=K, iteration=self.track_refine_iter)
        return np.asarray(pose, dtype=np.float64).reshape(4, 4)

    def set_pose(self, T_C_O: np.ndarray) -> None:
        import torch

        # FoundationPose tracks the pose of its internally centered mesh: T = P_centered @ tf_to_center.
        tf_to_center = self._est.get_tf_to_centered_mesh()
        T = torch.as_tensor(T_C_O, device=tf_to_center.device, dtype=torch.float)
        self._est.pose_last = T @ torch.linalg.inv(tf_to_center)

    def estimate_episode(
        self,
        episode_dir: Path,
        segmentation: SegmentationResult,
        objects: Sequence[dict[str, Any]],
        fps: float,
        tracking: TrackingCfg,
    ) -> tuple[PoseResult, dict[str, Any]]:
        """Estimate every object (``{name, mesh_path, symmetry_tfs}``) through the episode."""
        episode = RawEpisode(episode_dir)
        n = len(episode)
        if segmentation.num_frames != n:
            raise ValueError(f"Segmentation has {segmentation.num_frames} frames, episode has {n}.")
        m = len(objects)
        T_C_O = np.full((m, n, 4, 4), np.nan)
        valid = np.zeros((m, n), dtype=bool)
        registered = np.zeros((m, n), dtype=bool)
        scores = np.full((m, n), np.nan, dtype=np.float32)
        stats: dict[str, Any] = {}
        for k, obj in enumerate(objects):
            name = obj["name"]
            if name not in segmentation.object_names:
                raise ValueError(f"Object {name!r} is not in the segmentation ({segmentation.object_names}).")
            s = segmentation.object_names.index(name)
            symmetry_tfs = np.asarray(obj["symmetry_tfs"], dtype=np.float64).reshape(-1, 4, 4)
            self.set_object(Path(obj["mesh_path"]), symmetry_tfs)
            T_C_O[k], valid[k], registered[k], scores[k], stats[name] = estimate_object_track(
                self, episode, segmentation.masks[s], segmentation.boxes[s], symmetry_tfs, tracking
            )
        frame_index = episode.frame_index
        result = PoseResult(
            object_names=[obj["name"] for obj in objects],
            T_C_O=T_C_O,
            valid=valid,
            registered=registered,
            scores=scores,
            frame_index=frame_index,
            timestamps=frame_index / float(fps),
            intrinsic_k=episode.intrinsic_k,
        )
        return result, stats


def create_app(model: FoundationPoseModel) -> Any:
    """FastAPI app exposing ``POST /estimate`` for ``model``."""
    from pydantic import BaseModel

    class ObjectModel(BaseModel):
        name: str
        mesh_path: str
        symmetry_tfs: list[list[list[float]]]

    class EstimateRequest(BaseModel):
        episode_dir: str
        segmentation_path: str
        objects: list[ObjectModel]
        out_path: str
        fps: float
        est_refine_iter: int = 5
        track_refine_iter: int = 2
        tracking: dict[str, Any] = {}

    def info() -> dict[str, Any]:
        return {"model": "foundationpose", "weights_dir": str(model.cfg.weights_dir)}

    app, lock = create_model_app("FoundationPose 6D pose server", info)

    @app.post("/estimate")
    def estimate(request: EstimateRequest) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            model.est_refine_iter = request.est_refine_iter
            model.track_refine_iter = request.track_refine_iter
            result, stats = model.estimate_episode(
                Path(request.episode_dir),
                SegmentationResult.load(request.segmentation_path),
                [o.model_dump() for o in request.objects],
                request.fps,
                TrackingCfg(**request.tracking),
            )
            result.save(request.out_path)
            return {"out_path": request.out_path, "num_frames": int(result.T_C_O.shape[1]), "objects": stats}

        return run_locked(lock, work, "foundationpose")

    return app


class FoundationPoseEstimator(PoseEstimator):
    """Client that sends episodes to a running FoundationPose server."""

    cfg: FoundationPoseCfg

    def __init__(self, cfg: FoundationPoseCfg) -> None:
        super().__init__(cfg)
        self.client = ServerClient(cfg.server_url, timeout_s=cfg.timeout_s, name="FoundationPose")

    def check_ready(self) -> dict[str, Any]:
        return self.client.health()

    def estimate_episode(
        self,
        episode_dir: Path,
        segmentation_path: Path,
        objects: Sequence[ObjectSpec],
        out_path: Path,
        fps: float,
    ) -> dict[str, Any]:
        return self.client.post(
            "estimate",
            {
                "episode_dir": str(Path(episode_dir).resolve()),
                "segmentation_path": str(Path(segmentation_path).resolve()),
                "objects": [
                    {"name": o.name, "mesh_path": str(o.resolve_mesh()), "symmetry_tfs": o.symmetry_tfs().tolist()}
                    for o in objects
                ],
                "out_path": str(Path(out_path).resolve()),
                "fps": fps,
                "est_refine_iter": self.cfg.est_refine_iter,
                "track_refine_iter": self.cfg.track_refine_iter,
                "tracking": self.cfg.tracking.to_dict(),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose 6D pose server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR, help="Folder holding the two weight runs."
    )
    args = parser.parse_args()

    print(f"[FoundationPose] Loading weights from {args.weights_dir} ...")
    model = FoundationPoseModel(FoundationPoseServerCfg(weights_dir=args.weights_dir))
    print(f"[FoundationPose] Ready on http://{args.host}:{args.port}")
    serve(create_app(model), args.host, args.port)


if __name__ == "__main__":
    main()
