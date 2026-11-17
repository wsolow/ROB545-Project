"""record_human_demo.py.

Record human demonstration videos with the RealSense camera. Each saved episode is written
both to a LeRobot dataset (RGB video + intrinsics, no actions) under ``_data/<dataset_name>/``
and to a raw PNG folder ``_data/<dataset_name>/raw/episode_XXXXXX/{rgb,depth,cam_K.txt}``
in the ``collect_rgbd_demo.py`` layout.

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/record/record_human_demo.py
      --dataset-name human_pick_pink_block --task "Pick the pink block"
      --fps 30 --num-episodes 5

Written by Will Solow, 2026.
"""

import argparse
import sys

import cv2  # ruff: ignore[unused-import]. Needed for windows to launch

from kinova_lfd.data_collection.configs import Gen3DatasetConfig, find_available_dataset_name
from kinova_lfd.data_collection.human_demo.record_human_demo import HumanDemoRecordConfig, record_human_demo
from kinova_lfd.envs.base.cameras import RealsenseCameraCfg

REALSENSE_FPS = (6, 15, 30, 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record human demonstration videos with LeRobot.")
    parser.add_argument("--dataset-name", type=str, default="human_demo", help="Folder under _data/.")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo id (default local/<name>).")
    parser.add_argument("--task", type=str, default="Human demonstration", help="Task description.")
    parser.add_argument("--fps", type=int, default=30, help="Recording rate (also the camera stream rate).")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--episode-time", type=float, default=60.0, help="Seconds per episode.")
    parser.add_argument("--reset-time", type=float, default=10.0, help="Seconds between episodes (] skips).")
    parser.add_argument("--width", type=int, default=640, help="Camera stream width (px).")
    parser.add_argument("--height", type=int, default=480, help="Camera stream height (px).")
    parser.add_argument("--no-raw", action="store_true", help="Skip the raw rgb/depth PNG folder.")
    parser.add_argument("--no-view", action="store_true", help="Do not show the live camera window.")
    parser.add_argument("--no-video", action="store_true", help="Store images instead of video.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_name = args.dataset_name
    if not args.resume:
        dataset_name = find_available_dataset_name(dataset_name)
        if dataset_name != args.dataset_name:
            print(f"Dataset {args.dataset_name!r} already exists; using {dataset_name!r}.")
    repo_id = args.repo_id or f"local/{dataset_name}"

    # RealSense only streams at fixed rates; record other rates by sampling a 30 Hz stream.
    camera_fps = args.fps if args.fps in REALSENSE_FPS else 30
    if camera_fps != args.fps:
        print(f"RealSense cannot stream at {args.fps} Hz; streaming at {camera_fps} Hz and recording at {args.fps} Hz.")

    cfg = HumanDemoRecordConfig(
        camera=RealsenseCameraCfg(width=args.width, height=args.height, fps=camera_fps),
        dataset=Gen3DatasetConfig(
            dataset_name=dataset_name,
            repo_id=repo_id,
            single_task=args.task,
            fps=args.fps,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            num_episodes=args.num_episodes,
            video=not args.no_video,
            resume=args.resume,
        ),
        save_raw=not args.no_raw,
        view=not args.no_view,
    )

    dataset = record_human_demo(cfg)
    print(f"Saved {dataset.num_episodes} episodes to {cfg.dataset.resolved_root().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
