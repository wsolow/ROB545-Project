"""track_object_pose.py.

Segment objects (SAM3, text prompts) and estimate their 6D pose (FoundationPose) through human
demonstration episodes recorded with ``scripts/record/record_human_demo.py``. Outputs go next to
each dataset's ``raw/`` folder::

    <dataset>/segmentation/episode_XXXXXX.npz   masks, boxes, scores per object
    <dataset>/6d_pose/episode_XXXXXX.npz        T_C_O (camera frame), valid, registered, scores
    <dataset>/6d_pose/meta.json                 objects + configs

Start the two model servers first (each keeps its weights on the GPU between runs)::

    python -m kinova_lfd.data_collection.human_demo.perception.segmentation.sam3 --port 8001
    python -m kinova_lfd.data_collection.human_demo.perception.pose.foundationpose --port 8002

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/perception/track_object_pose.py
      --dataset _data/human_demo --episodes 0 1 --overlay

Written by Will Solow, 2026.
"""

import argparse
import sys

from kinova_lfd.data_collection.human_demo.perception import (
    STAGES,
    ObjectTrackingCfg,
    ObjectTrackingPipeline,
)
from kinova_lfd.data_collection.human_demo.perception.pose import FoundationPoseCfg
from kinova_lfd.data_collection.human_demo.perception.segmentation import Sam3Cfg
from kinova_lfd.data_collection.human_demo.perception.types import load_object_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track object 6D poses in human demonstration datasets.")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Dataset root(s), e.g. _data/human_demo.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episode indices (default: all).")
    parser.add_argument("--objects", type=str, default="objects.yaml", help="Objects YAML file.")
    parser.add_argument("--stages", type=str, nargs="+", default=list(STAGES), choices=STAGES, help="Stages to run.")

    parser.add_argument("--overlay", action="store_true", help="Write an _overlay.mp4 check video per episode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ObjectTrackingCfg(
        objects=load_object_specs(args.objects),
        segmenter=Sam3Cfg(),
        pose=FoundationPoseCfg(),
        stages=tuple(args.stages),
        save_overlay=args.overlay,
    )
    ObjectTrackingPipeline(cfg).run(args.dataset, args.episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
