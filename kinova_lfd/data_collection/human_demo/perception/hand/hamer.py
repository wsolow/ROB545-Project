"""HaMeR MANO hand reconstruction: model wrapper, HaMeR's own hand detector, FastAPI server, and client.

Start the server once (it keeps the weights on the GPU)::

    python -m kinova_lfd.data_collection.human_demo.perception.hand.hamer --port 8003 --detector yolov8

Then ``HamerEstimator`` (``HamerCfg(server_url=...)``) sends it episode folders. On every frame the
server's hand detector finds hand boxes with their handedness, the most confident box of each hand
is kept, and HaMeR regresses MANO parameters plus a weak-perspective camera from the box crop. The
camera is converted to a camera-frame translation with the episode's real intrinsics.

Hand detectors (``--detector``, see ``base.HAND_DETECTORS``):

- ``vitpose``: HaMeR's original pipeline, :class:`VitPoseHandDetector` below (detectron2 ViTDet-H
  person boxes, then ViTPose-H whole-body keypoints, hand boxes from the hand keypoints). Needs
  ``detectron2`` and HaMeR's ``third-party/ViTPose`` (``mmcv==1.3.9``) installed.
- ``yolov8``: WiLoR's YOLOv8 hand detector (``yolov8.py``), much lighter.

HaMeR's source lives in ``third_party/hamer``. Its weights live in ``kinova_lfd/.cache/models/hamer``
(the contents of HaMeR's ``_DATA`` folder from ``hamer_demo_data.tar.gz``)::

    hamer_ckpts/checkpoints/hamer.ckpt, hamer_ckpts/model_config.yaml
    data/mano_mean_params.npz, data/mano/MANO_RIGHT.pkl   # MANO: download from mano.is.tue.mpg.de
    vitpose_ckpts/vitpose+_huge/wholebody.pth             # vitpose detector only
    vitdet/model_final_f05665.pkl                         # vitpose detector only, downloaded on first use
    yolov8_hand_detector.pt                               # yolov8 detector only

``MANO_RIGHT.pkl`` stores some arrays as ``chumpy`` objects, which do not import on Python 3.12 /
numpy 2; the server rewrites it once with plain numpy arrays (the original is kept as
``MANO_RIGHT_chumpy.pkl``).

Written by Will Solow, 2026.
"""

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import RawEpisode
from kinova_lfd.data_collection.human_demo.perception.hand.base import (
    HAND_DETECTORS,
    HandDetection,
    HandDetector,
    HandEstimator,
    HandEstimatorCfg,
    best_per_side,
    make_hand_detector,
)
from kinova_lfd.data_collection.human_demo.perception.server_utils import (
    DEFAULT_TIMEOUT_S,
    ServerClient,
    create_model_app,
    run_locked,
    serve,
)
from kinova_lfd.data_collection.human_demo.perception.types import HAND_SIDES, MODELS_DIR, HandResult

DEFAULT_PORT = 8003
REPO_ROOT = Path(__file__).resolve().parents[5]
HAMER_DIR = REPO_ROOT / "third_party" / "hamer"
VITPOSE_DIR = HAMER_DIR / "third-party" / "ViTPose"
DEFAULT_MODELS_DIR = MODELS_DIR / "hamer"
DEFAULT_DETECTOR = "yolov8"

VITDET_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)
VITPOSE_CONFIG = (
    VITPOSE_DIR
    / "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
)
NUM_JOINTS = 21
"""HaMeR joints (OpenPose hand order)."""


@dataclass
class HamerCfg(HandEstimatorCfg):
    """Client settings for the HaMeR server."""

    name: str = "hamer"
    server_url: str = f"http://127.0.0.1:{DEFAULT_PORT}"
    timeout_s: float = DEFAULT_TIMEOUT_S
    det_score_thresh: float = 0.5
    """Minimum detector score for a hand box to be used."""
    rescale_factor: float = 2.0
    """Padding applied to the hand box before cropping (HaMeR's demo default)."""
    batch_size: int = 8
    """Hand crops per HaMeR forward pass."""
    use_true_intrinsics: bool = True
    """Convert HaMeR's camera with ``cam_K.txt``; otherwise use HaMeR's assumed focal length and image center."""


@dataclass
class HamerServerCfg:
    """Model settings used when the server starts."""

    models_dir: Path = DEFAULT_MODELS_DIR
    detector: str = DEFAULT_DETECTOR
    """Hand detector, one of ``HAND_DETECTORS``."""
    device: str = "cuda"


# --------------------------------------------------------------------------- MANO (chumpy-free)


class _ChumpyStub:
    """Stand-in for chumpy objects while unpickling; keeps their state dict."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _ChumpyFreeUnpickler(pickle.Unpickler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.found_chumpy = False

    def find_class(self, module: str, name: str) -> Any:
        if module == "chumpy" or module.startswith("chumpy."):
            self.found_chumpy = True
            return _ChumpyStub
        return super().find_class(module, name)


def _chumpy_to_numpy(value: Any, key: str) -> Any:
    if not isinstance(value, _ChumpyStub):
        return value
    state = value.__dict__
    if "x" in state:  # terminal chumpy.Ch stores its value in .x
        return np.asarray(state["x"])
    arrays = [v for v in state.values() if isinstance(v, np.ndarray)]
    if len(arrays) == 1:
        return arrays[0]
    raise ValueError(f"MANO entry {key!r} is a chumpy expression that cannot be converted without chumpy.")


def ensure_mano_numpy(mano_path: Path) -> None:
    """Rewrite a MANO ``.pkl`` with plain numpy arrays if it contains chumpy objects (keeps a backup)."""
    with open(mano_path, "rb") as f:
        unpickler = _ChumpyFreeUnpickler(f, encoding="latin1")
        data = unpickler.load()
    if not unpickler.found_chumpy:
        return
    clean = {key: _chumpy_to_numpy(value, key) for key, value in data.items()}
    backup = mano_path.with_name(f"{mano_path.stem}_chumpy{mano_path.suffix}")
    if not backup.exists():
        mano_path.rename(backup)
    with open(mano_path, "wb") as f:
        pickle.dump(clean, f)
    print(f"[HaMeR] Rewrote {mano_path} without chumpy (original kept at {backup}).")


# --------------------------------------------------------------------------- HaMeR's hand detector


class VitPoseHandDetector(HandDetector):
    """HaMeR's original hand detector: ViTDet-H person boxes, then ViTPose-H whole-body keypoints.

    A hand box is the extent of that hand's confident keypoints; its score is their mean confidence.
    """

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        device: str = "cuda",
        person_score_thresh: float = 0.5,
        keypoint_thresh: float = 0.5,
        min_keypoints: int = 4,
    ) -> None:
        import torch
        from detectron2.config import LazyConfig
        from mmpose.apis import init_pose_model

        import hamer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

        models_dir = Path(models_dir)
        vitpose_ckpt = models_dir / "vitpose_ckpts" / "vitpose+_huge" / "wholebody.pth"
        if not vitpose_ckpt.exists():
            raise FileNotFoundError(f"ViTPose checkpoint {vitpose_ckpt} not found (from hamer_demo_data.tar.gz).")
        if not VITPOSE_CONFIG.exists():
            raise FileNotFoundError(f"{VITPOSE_CONFIG} not found; clone HaMeR with --recursive.")
        vitdet_ckpt = models_dir / "vitdet" / Path(VITDET_URL).name
        if not vitdet_ckpt.exists():
            print(f"[HaMeR] Downloading ViTDet-H weights to {vitdet_ckpt} ...")
            vitdet_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(VITDET_URL, str(vitdet_ckpt))

        cfg = LazyConfig.load(str(Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"))
        cfg.train.init_checkpoint = str(vitdet_ckpt)
        for i in range(3):
            cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.person_detector = DefaultPredictor_Lazy(cfg)
        self.pose_model = init_pose_model(str(VITPOSE_CONFIG), str(vitpose_ckpt), device=device)
        self.person_score_thresh = person_score_thresh
        self.keypoint_thresh = keypoint_thresh
        self.min_keypoints = min_keypoints

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        from mmpose.apis import inference_top_down_pose_model, process_mmdet_results

        bgr = np.ascontiguousarray(rgb[..., ::-1])
        instances = self.person_detector(bgr)["instances"]
        keep = (instances.pred_classes == 0) & (instances.scores > self.person_score_thresh)
        person_boxes = instances.pred_boxes.tensor[keep].cpu().numpy()
        person_scores = instances.scores[keep].cpu().numpy()
        if len(person_boxes) == 0:
            return []

        poses, _ = inference_top_down_pose_model(
            self.pose_model,
            bgr,
            person_results=process_mmdet_results([np.concatenate([person_boxes, person_scores[:, None]], axis=1)], 1),
            bbox_thr=self.person_score_thresh,
            format="xyxy",
        )
        detections = []
        for pose in poses:
            # COCO-WholeBody: the last 42 keypoints are the left hand then the right hand.
            for is_right, keyp in ((False, pose["keypoints"][-42:-21]), (True, pose["keypoints"][-21:])):
                valid = keyp[:, 2] > self.keypoint_thresh
                if valid.sum() < self.min_keypoints:
                    continue
                xy = keyp[valid, :2]
                box = np.array([*xy.min(axis=0), *xy.max(axis=0)], dtype=np.float32)
                detections.append(HandDetection(box=box, score=float(keyp[valid, 2].mean()), is_right=is_right))
        return detections


# --------------------------------------------------------------------------- HaMeR model


class HamerModel:
    """In-process HaMeR regressor plus a hand detector."""

    def __init__(self, cfg: HamerServerCfg) -> None:
        if not HAMER_DIR.is_dir():
            raise FileNotFoundError(f"{HAMER_DIR} not found; clone geopavlakos/hamer (--recursive) into third_party/.")
        models_dir = Path(cfg.models_dir)
        checkpoint = models_dir / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
        mano_path = models_dir / "data" / "mano" / "MANO_RIGHT.pkl"
        for path in (
            checkpoint,
            checkpoint.parents[1] / "model_config.yaml",
            models_dir / "data" / "mano_mean_params.npz",
        ):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; extract hamer_demo_data.tar.gz's _DATA/ into {models_dir}.")
        if not mano_path.exists():
            raise FileNotFoundError(f"{mano_path} not found; download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de.")
        ensure_mano_numpy(mano_path)

        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # hamer.utils imports pyrender
        if str(HAMER_DIR) not in sys.path:
            sys.path.insert(0, str(HAMER_DIR))
        import torch

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model, self.model_cfg = self._load_hamer(checkpoint, models_dir, cfg.device)
        self.detector = make_hand_detector(cfg.detector, models_dir, cfg.device)

    @staticmethod
    def _load_hamer(checkpoint: Path, models_dir: Path, device: str) -> tuple[Any, Any]:
        """``hamer.models.load_hamer`` with MANO paths pointed at ``models_dir`` and no renderer."""
        from hamer.configs import get_config
        from hamer.models import HAMER

        model_cfg = get_config(str(checkpoint.parents[1] / "model_config.yaml"), update_cachedir=False)
        model_cfg.defrost()
        # HaMeR's config paths are relative to its ./_DATA folder; resolve them in models_dir instead.
        for key in ("DATA_DIR", "MODEL_PATH", "MEAN_PARAMS", "JOINT_REGRESSOR_EXTRA"):
            value = model_cfg.MANO.get(key)
            if value and not os.path.isabs(value):
                rel = Path(value)
                if rel.parts and rel.parts[0] in (".", "_DATA"):
                    rel = Path(*rel.parts[1:])
                model_cfg.MANO[key] = str(models_dir / rel)
        if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        model_cfg.freeze()

        # The checkpoint stores a yacs config in its hyperparameters, so it needs weights_only=False.
        model = HAMER.load_from_checkpoint(
            str(checkpoint), strict=False, cfg=model_cfg, init_renderer=False, map_location="cpu", weights_only=False
        )
        return model.to(device).eval(), model_cfg

    def _crop(
        self, rgb: np.ndarray, box: np.ndarray, is_right: bool, rescale_factor: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """HaMeR's ``ViTDetDataset`` preprocessing for one box; returns ``(patch (3,S,S), center (2,), box_size)``.

        Left hands are mirrored so HaMeR always sees a right hand.
        """
        from hamer.datasets.utils import expand_to_aspect_ratio, generate_image_patch_cv2

        size = self.model_cfg.MODEL.IMAGE_SIZE
        center = (box[:2] + box[2:]) / 2.0
        box_size = float(
            expand_to_aspect_ratio(
                rescale_factor * (box[2:] - box[:2]), target_aspect_ratio=self.model_cfg.MODEL.BBOX_SHAPE
            ).max()
        )
        # Anti-alias before downsampling (skimage.gaussian in HaMeR; same kernel extent here).
        downsampling = box_size / size / 2.0
        img = rgb
        if downsampling > 1.1:
            sigma = (downsampling - 1) / 2
            ksize = 2 * int(np.ceil(4 * sigma)) + 1
            img = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        patch, _ = generate_image_patch_cv2(
            img.astype(np.float32),
            center[0],
            center[1],
            box_size,
            box_size,
            size,
            size,
            not is_right,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        mean = 255.0 * np.asarray(self.model_cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        std = 255.0 * np.asarray(self.model_cfg.MODEL.IMAGE_STD, dtype=np.float32)
        patch = ((patch - mean) / std).transpose(2, 0, 1)
        return patch.astype(np.float32), center.astype(np.float32), box_size

    def _regress(
        self, crops: list[dict[str, Any]], K: np.ndarray, img_hw: tuple[int, int], use_true_intrinsics: bool
    ) -> dict[str, np.ndarray]:
        """Run HaMeR on a batch of crops; returns camera-frame outputs per crop."""
        import torch

        batch = torch.from_numpy(np.stack([c["patch"] for c in crops])).to(self.device)
        with torch.no_grad():
            out = self.model({"img": batch})
        right = np.array([c["is_right"] for c in crops], dtype=np.float64)
        sign = 2.0 * right - 1.0  # +1 right, -1 left (undo the mirroring)
        pred_cam = out["pred_cam"].float().cpu().numpy().astype(np.float64)
        pred_cam[:, 1] *= sign

        h, w = img_hw
        if use_true_intrinsics:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx = fy = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * max(h, w)
            cx, cy = w / 2.0, h / 2.0
        center = np.stack([c["center"] for c in crops]).astype(np.float64)
        box_size = np.array([c["box_size"] for c in crops], dtype=np.float64)
        # hamer.utils.renderer.cam_crop_to_full, generalized to fx != fy and an off-center principal point.
        tz = 2.0 * 0.5 * (fx + fy) / (box_size * pred_cam[:, 0] + 1e-9)
        tx = (center[:, 0] - cx) * tz / fx + pred_cam[:, 1]
        ty = (center[:, 1] - cy) * tz / fy + pred_cam[:, 2]
        cam_t = np.stack([tx, ty, tz], axis=-1)

        kp = out["pred_keypoints_3d"].float().cpu().numpy().astype(np.float64)
        kp[:, :, 0] *= sign[:, None]
        kp_cam = kp + cam_t[:, None]
        uvw = kp_cam @ K.T
        kp_2d = uvw[..., :2] / uvw[..., 2:3]

        params = out["pred_mano_params"]
        n = len(crops)
        return {
            "cam_t": cam_t,
            "keypoints_3d": kp_cam,
            "keypoints_2d": kp_2d,
            "global_orient": params["global_orient"].float().reshape(n, 3, 3).cpu().numpy(),
            "hand_pose": params["hand_pose"].float().reshape(n, -1, 3, 3).cpu().numpy(),
            "betas": params["betas"].float().reshape(n, -1).cpu().numpy(),
        }

    def estimate_episode(
        self,
        episode_dir: Path,
        fps: float,
        det_score_thresh: float = 0.5,
        rescale_factor: float = 2.0,
        batch_size: int = 8,
        use_true_intrinsics: bool = True,
    ) -> tuple[HandResult, dict[str, Any]]:
        """Detect and reconstruct both hands on every frame of the episode."""
        import torch

        episode = RawEpisode(episode_dir)
        n = len(episode)
        h, w = episode.image_size()
        K = episode.intrinsic_k
        s = len(HAND_SIDES)
        num_betas = int(self.model_cfg.MANO.get("NUM_BETAS", 10))
        num_hand_joints = int(self.model_cfg.MANO.get("NUM_HAND_JOINTS", 15))
        detected = np.zeros((s, n), dtype=bool)
        boxes = np.full((s, n, 4), np.nan, dtype=np.float32)
        det_scores = np.zeros((s, n), dtype=np.float32)
        global_orient = np.full((s, n, 3, 3), np.nan, dtype=np.float32)
        hand_pose = np.full((s, n, num_hand_joints, 3, 3), np.nan, dtype=np.float32)
        betas = np.full((s, n, num_betas), np.nan, dtype=np.float32)
        cam_t = np.full((s, n, 3), np.nan)
        keypoints_3d = np.full((s, n, NUM_JOINTS, 3), np.nan)
        keypoints_2d = np.full((s, n, NUM_JOINTS, 2), np.nan, dtype=np.float32)

        pending: list[dict[str, Any]] = []

        def flush() -> None:
            if not pending:
                return
            out = self._regress(pending, K, (h, w), use_true_intrinsics)
            for j, crop in enumerate(pending):
                k, i = crop["side"], crop["frame"]
                global_orient[k, i] = out["global_orient"][j]
                hand_pose[k, i] = out["hand_pose"][j]
                betas[k, i] = out["betas"][j]
                cam_t[k, i] = out["cam_t"][j]
                keypoints_3d[k, i] = out["keypoints_3d"][j]
                keypoints_2d[k, i] = out["keypoints_2d"][j]
            pending.clear()

        try:
            for i in range(n):
                rgb = episode.rgb(i)
                for k, det in enumerate(best_per_side(self.detector.detect(rgb), det_score_thresh)):
                    if det is None:
                        continue
                    detected[k, i] = True
                    boxes[k, i] = det.box
                    det_scores[k, i] = det.score
                    patch, center, box_size = self._crop(rgb, det.box, det.is_right, rescale_factor)
                    pending.append(
                        {
                            "side": k,
                            "frame": i,
                            "is_right": det.is_right,
                            "patch": patch,
                            "center": center,
                            "box_size": box_size,
                        }
                    )
                if len(pending) >= batch_size:
                    flush()
            flush()
        finally:
            torch.cuda.empty_cache()

        frame_index = episode.frame_index
        result = HandResult(
            hand_sides=list(HAND_SIDES),
            detected=detected,
            boxes=boxes,
            det_scores=det_scores,
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            cam_t=cam_t,
            keypoints_3d=keypoints_3d,
            keypoints_2d=keypoints_2d,
            frame_index=frame_index,
            timestamps=frame_index / float(fps),
            intrinsic_k=K,
            detector=self.cfg.detector,
        )
        stats = {
            side: {
                "detected_frames": int(detected[k].sum()),
                "mean_det_score": round(float(det_scores[k][detected[k]].mean()), 4) if detected[k].any() else 0.0,
            }
            for k, side in enumerate(HAND_SIDES)
        }
        return result, stats


def create_app(model: HamerModel) -> Any:
    """FastAPI app exposing ``POST /estimate`` for ``model``."""
    from pydantic import BaseModel

    class EstimateRequest(BaseModel):
        episode_dir: str
        out_path: str
        fps: float
        det_score_thresh: float = 0.5
        rescale_factor: float = 2.0
        batch_size: int = 8
        use_true_intrinsics: bool = True

    def info() -> dict[str, Any]:
        return {"model": "hamer", "detector": model.cfg.detector, "models_dir": str(model.cfg.models_dir)}

    app, lock = create_model_app("HaMeR hand pose server", info)

    @app.post("/estimate")
    def estimate(request: EstimateRequest) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            result, stats = model.estimate_episode(
                Path(request.episode_dir),
                request.fps,
                det_score_thresh=request.det_score_thresh,
                rescale_factor=request.rescale_factor,
                batch_size=request.batch_size,
                use_true_intrinsics=request.use_true_intrinsics,
            )
            result.save(request.out_path)
            return {"out_path": request.out_path, "num_frames": result.num_frames, "hands": stats}

        return run_locked(lock, work, "hamer")

    return app


class HamerEstimator(HandEstimator):
    """Client that sends episodes to a running HaMeR server."""

    cfg: HamerCfg

    def __init__(self, cfg: HamerCfg) -> None:
        super().__init__(cfg)
        self.client = ServerClient(cfg.server_url, timeout_s=cfg.timeout_s, name="HaMeR")

    def check_ready(self) -> dict[str, Any]:
        return self.client.health()

    def estimate_episode(self, episode_dir: Path, out_path: Path, fps: float) -> dict[str, Any]:
        return self.client.post(
            "estimate",
            {
                "episode_dir": str(Path(episode_dir).resolve()),
                "out_path": str(Path(out_path).resolve()),
                "fps": fps,
                "det_score_thresh": self.cfg.det_score_thresh,
                "rescale_factor": self.cfg.rescale_factor,
                "batch_size": self.cfg.batch_size,
                "use_true_intrinsics": self.cfg.use_true_intrinsics,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="HaMeR hand pose server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Folder holding HaMeR's _DATA contents."
    )
    parser.add_argument("--detector", type=str, default=DEFAULT_DETECTOR, choices=HAND_DETECTORS, help="Hand detector.")
    args = parser.parse_args()

    print(f"[HaMeR] Loading weights from {args.models_dir} (detector: {args.detector}) ...")
    model = HamerModel(HamerServerCfg(models_dir=args.models_dir, detector=args.detector))
    print(f"[HaMeR] Ready on http://{args.host}:{args.port}")
    serve(create_app(model), args.host, args.port)


if __name__ == "__main__":
    main()
