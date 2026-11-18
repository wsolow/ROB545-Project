"""SAM3 text-prompted video segmentation: model wrapper, FastAPI server, and client.

Start the server once (it keeps the weights on the GPU)::

    python -m kinova_lfd.data_collection.human_demo.perception.segmentation.sam3 --port 8001

Then ``Sam3Segmenter`` (``Sam3Cfg(server_url=...)``) sends it episode folders to segment. Each
object is one text concept; SAM3 detects and tracks every instance of the concept through the
video and the instance with the highest average score is kept.

Written by Will Solow, 2026.
"""

import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.data_collection.human_demo.perception.episode_io import RawEpisode
from kinova_lfd.data_collection.human_demo.perception.segmentation.base import Segmenter, SegmenterCfg
from kinova_lfd.data_collection.human_demo.perception.server_utils import (
    DEFAULT_TIMEOUT_S,
    ServerClient,
    create_model_app,
    run_locked,
    serve,
)
from kinova_lfd.data_collection.human_demo.perception.types import MODELS_DIR, ObjectSpec, SegmentationResult

DEFAULT_PORT = 8001
DEFAULT_CHECKPOINT = MODELS_DIR / "sam3" / "sam3.pt"


@dataclass
class Sam3Cfg(SegmenterCfg):
    """Client settings for the SAM3 server."""

    name: str = "sam3"
    server_url: str = f"http://127.0.0.1:{DEFAULT_PORT}"
    timeout_s: float = DEFAULT_TIMEOUT_S
    output_prob_thresh: float = 0.5
    """Minimum SAM3 score for an instance to be reported on a frame."""
    prompt_frame_index: int = 0
    """Frame the text prompt is added on; propagation runs forward and backward from it."""


@dataclass
class Sam3ServerCfg:
    """Model settings used when the server starts."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    offload_video_to_cpu: bool = False
    """Keep the decoded video frames on the CPU (saves GPU memory on long episodes)."""


class Sam3Model:
    """In-process SAM3 video predictor."""

    def __init__(self, cfg: Sam3ServerCfg) -> None:
        from sam3.model_builder import build_sam3_video_predictor

        if not Path(cfg.checkpoint_path).exists():
            raise FileNotFoundError(f"SAM3 checkpoint {cfg.checkpoint_path} not found.")
        self.cfg = cfg
        self.predictor = build_sam3_video_predictor(checkpoint_path=str(cfg.checkpoint_path))

    def segment(
        self,
        episode_dir: Path,
        objects: Sequence[tuple[str, str]],
        output_prob_thresh: float = 0.5,
        prompt_frame_index: int = 0,
    ) -> tuple[SegmentationResult, dict[str, Any]]:
        """Segment ``(name, prompt)`` objects through the episode; returns the result and per-object stats."""
        import torch

        # SAM3's tracker enters a bf16 autocast context once, at construction, and relies on it for
        # the whole process. Autocast state is per thread and the server handles requests on worker
        # threads, so re-enter the same context here.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self._segment(episode_dir, objects, output_prob_thresh, prompt_frame_index)

    def _segment(
        self,
        episode_dir: Path,
        objects: Sequence[tuple[str, str]],
        output_prob_thresh: float,
        prompt_frame_index: int,
    ) -> tuple[SegmentationResult, dict[str, Any]]:
        import torch

        episode = RawEpisode(episode_dir)
        n = len(episode)
        h, w = episode.image_size()
        m = len(objects)
        masks = np.zeros((m, n, h, w), dtype=bool)
        boxes = np.full((m, n, 4), np.nan, dtype=np.float32)
        scores = np.zeros((m, n), dtype=np.float32)
        instance_ids = np.full(m, -1, dtype=np.int64)
        stats: dict[str, Any] = {}

        session_id = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(episode.rgb_dir),
                "offload_video_to_cpu": self.cfg.offload_video_to_cpu,
            }
        )["session_id"]
        try:
            for k, (name, prompt) in enumerate(objects):
                self.predictor.handle_request({"type": "reset_session", "session_id": session_id})
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": prompt_frame_index,
                        "text": prompt,
                        "output_prob_thresh": output_prob_thresh,
                    }
                )
                # obj_id -> {frame: packed mask}, obj_id -> {frame: score}
                inst_masks: dict[int, dict[int, np.ndarray]] = {}
                inst_scores: dict[int, dict[int, float]] = {}
                for response in self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "both",
                        "start_frame_index": prompt_frame_index,
                        "output_prob_thresh": output_prob_thresh,
                    }
                ):
                    frame = int(response["frame_index"])
                    out = response["outputs"]
                    for obj_id, prob, mask in zip(out["out_obj_ids"], out["out_probs"], out["out_binary_masks"]):
                        inst_masks.setdefault(int(obj_id), {})[frame] = np.packbits(mask.reshape(-1))
                        inst_scores.setdefault(int(obj_id), {})[frame] = float(prob)

                # Keep the instance with the highest average score over all frames (absent = 0), which
                # favors an instance that is both confident and present for most of the episode.
                mean_scores = {obj_id: sum(s.values()) / n for obj_id, s in inst_scores.items()}
                stats[name] = {"prompt": prompt, "instances_found": len(mean_scores), "detected_frames": 0}
                if not mean_scores:
                    continue
                best = max(mean_scores, key=mean_scores.__getitem__)
                instance_ids[k] = best
                for frame, packed in inst_masks[best].items():
                    mask = np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)
                    if not mask.any():
                        continue
                    masks[k, frame] = mask
                    ys, xs = np.nonzero(mask)
                    boxes[k, frame] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                    scores[k, frame] = inst_scores[best][frame]
                stats[name]["detected_frames"] = int(masks[k].any(axis=(1, 2)).sum())
                stats[name]["mean_score"] = round(mean_scores[best], 4)
        finally:
            self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            gc.collect()
            torch.cuda.empty_cache()

        result = SegmentationResult(
            object_names=[name for name, _ in objects],
            prompts=[prompt for _, prompt in objects],
            masks=masks,
            boxes=boxes,
            scores=scores,
            instance_ids=instance_ids,
        )
        return result, stats


def create_app(model: Sam3Model) -> Any:
    """FastAPI app exposing ``POST /segment`` for ``model``."""
    from pydantic import BaseModel

    class ObjectPrompt(BaseModel):
        name: str
        prompt: str

    class SegmentRequest(BaseModel):
        episode_dir: str
        objects: list[ObjectPrompt]
        out_path: str
        output_prob_thresh: float = 0.5
        prompt_frame_index: int = 0

    def info() -> dict[str, Any]:
        return {"model": "sam3", "checkpoint": str(model.cfg.checkpoint_path)}

    app, lock = create_model_app("SAM3 segmentation server", info)

    @app.post("/segment")
    def segment(request: SegmentRequest) -> dict[str, Any]:
        def work() -> dict[str, Any]:
            result, stats = model.segment(
                Path(request.episode_dir),
                [(o.name, o.prompt) for o in request.objects],
                output_prob_thresh=request.output_prob_thresh,
                prompt_frame_index=request.prompt_frame_index,
            )
            result.save(request.out_path)
            return {"out_path": request.out_path, "num_frames": result.num_frames, "objects": stats}

        return run_locked(lock, work, "sam3")

    return app


class Sam3Segmenter(Segmenter):
    """Client that sends episodes to a running SAM3 server."""

    cfg: Sam3Cfg

    def __init__(self, cfg: Sam3Cfg) -> None:
        super().__init__(cfg)
        self.client = ServerClient(cfg.server_url, timeout_s=cfg.timeout_s, name="SAM3")

    def check_ready(self) -> dict[str, Any]:
        return self.client.health()

    def segment_episode(self, episode_dir: Path, objects: Sequence[ObjectSpec], out_path: Path) -> dict[str, Any]:
        return self.client.post(
            "segment",
            {
                "episode_dir": str(Path(episode_dir).resolve()),
                "objects": [{"name": o.name, "prompt": o.prompt} for o in objects],
                "out_path": str(Path(out_path).resolve()),
                "output_prob_thresh": self.cfg.output_prob_thresh,
                "prompt_frame_index": self.cfg.prompt_frame_index,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 segmentation server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM3 checkpoint (.pt).")
    parser.add_argument("--offload-video-to-cpu", action="store_true", help="Keep decoded frames on the CPU.")
    args = parser.parse_args()

    print(f"[SAM3] Loading {args.checkpoint} ...")
    model = Sam3Model(Sam3ServerCfg(checkpoint_path=args.checkpoint, offload_video_to_cpu=args.offload_video_to_cpu))
    print(f"[SAM3] Ready on http://{args.host}:{args.port}")
    serve(create_app(model), args.host, args.port)


if __name__ == "__main__":
    main()
