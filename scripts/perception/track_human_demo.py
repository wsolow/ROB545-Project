"""track_human_demo.py.

Run perception on human demonstration episodes recorded with ``scripts/record/record_human_demo.py``:
object segmentation (SAM3, text prompts), 6D object pose (FoundationPose), and hand pose (HaMeR,
MANO). Pick any subset with ``--stages`` (default: all three). Outputs go next to each dataset's
``raw/`` folder::

    <dataset>/segmentation/episode_XXXXXX.npz        masks, boxes, scores per object
    <dataset>/6d_pose/episode_XXXXXX.npz             T_C_O (camera frame), valid, registered, scores
    <dataset>/6d_pose/meta.json                      objects + object configs
    <dataset>/hand_pose/episode_XXXXXX.npz           MANO params, keypoints (camera frame), boxes per hand
    <dataset>/hand_pose/meta.json                    hand config
    <dataset>/overlay/episode_XXXXXX_objects.mp4     object check video   (object stages)
    <dataset>/overlay/episode_XXXXXX_hands.mp4       hand check video     (hand stage)
    <dataset>/overlay/episode_XXXXXX_combined.mp4    both overlays        (object + hand stages)

Start the model servers for the stages you run first (each keeps its weights on the GPU)::

    python -m kinova_lfd.data_collection.human_demo.perception.segmentation.sam3 --port 8001
    python -m kinova_lfd.data_collection.human_demo.perception.pose.foundationpose --port 8002
    python -m kinova_lfd.data_collection.human_demo.perception.hand.hamer --port 8003 --detector yolov8

Examples::

    # everything
    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/perception/track_human_demo.py
      --dataset _data/human_demo --episodes 0 1
    # hands only
    python scripts/perception/track_human_demo.py --dataset _data/human_demo --stages hand --no-overlay
    # object segmentation + hands (no 6D pose)
    python scripts/perception/track_human_demo.py --dataset _data/human_demo --stages segmentation hand

Written by Will Solow, 2026.
"""

import argparse
import sys

from kinova_lfd.data_collection.human_demo.perception import STAGES, PerceptionCfg, PerceptionPipeline
from kinova_lfd.data_collection.human_demo.perception.hand import HamerCfg
from kinova_lfd.data_collection.human_demo.perception.pipeline import OBJECT_STAGES
from kinova_lfd.data_collection.human_demo.perception.pose import FoundationPoseCfg
from kinova_lfd.data_collection.human_demo.perception.segmentation import Sam3Cfg
from kinova_lfd.data_collection.human_demo.perception.types import load_object_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track objects and hands in human demonstration datasets.")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Dataset root(s), e.g. _data/human_demo.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices (default: all).")
    parser.add_argument("--stages", type=str, nargs="+", default=list(STAGES), choices=STAGES, help="Stages to run.")
    parser.add_argument(
        "--objects",
        type=str,
        default="objects.yaml",
        help="Objects YAML file (used by the segmentation / pose stages).",
    )
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write check videos to <dataset>/overlay/ per episode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = tuple(s for s in STAGES if s in args.stages)  # canonical order, no duplicates
    cfg = PerceptionCfg(
        stages=stages,
        objects=load_object_specs(args.objects) if any(s in stages for s in OBJECT_STAGES) else [],
        segmenter=Sam3Cfg(),
        pose=FoundationPoseCfg(),
        hand=HamerCfg(),
        save_overlay=args.overlay,
    )
    PerceptionPipeline(cfg).run(args.dataset, args.episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
