"""record_gen3.py.

Record Gen3 demonstrations (MuJoCo or Kortex TCP) with the repo's keyboard or Xbox
teleop devices. Episodes are saved under ``_data/<dataset_name>/``.

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/record/record_gen3.py
      --robot mj_gen3 --teleop xbox
      --dataset-name pick_pink_block --task "Pick the pink block"
      --fps 20 --num-episodes 5

Written by Will Solow, 2026.
"""

import argparse
import sys

from kinova_lfd.data_collection import (
    TELEOP_DEVICES,
    Gen3DatasetConfig,
    Gen3RecordConfig,
    find_available_dataset_name,
    make_robot_config,
    make_teleop_config,
    record,
)

CAMERA_CHOICES = {
    "both": (True, True),
    "wrist": (True, False),
    "tabletop": (False, True),
    "none": (False, False),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Gen3 demos with LeRobot.")
    parser.add_argument(
        "--robot",
        type=str,
        default="mj_gen3",
        choices=("mj_gen3", "kortex_tcp_gen3"),
        help="Robot backend.",
    )
    parser.add_argument(
        "--teleop",
        type=str,
        default="keyboard",
        choices=TELEOP_DEVICES,
        help="Teleop device (keyboard = Se3Keyboard, xbox = XboxController); both 6-DoF + gripper.",
    )
    parser.add_argument("--dataset-name", type=str, default="gen3_demo", help="Folder under _data/.")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo id (default local/<name>).")
    parser.add_argument("--task", type=str, default="Gen3 demonstration", help="Task description.")
    parser.add_argument("--fps", type=int, default=20, help="Recording / control rate.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--episode-time", type=float, default=60.0, help="Seconds per episode.")
    parser.add_argument("--reset-time", type=float, default=10.0, help="Unrecorded reset seconds.")
    parser.add_argument(
        "--pos-sensitivity", type=float, default=0.15, help="Linear speed (m/s) at full key press / stick."
    )
    parser.add_argument(
        "--rot-sensitivity", type=float, default=0.15, help="Angular speed (rad/s) at full key press / stick."
    )
    parser.add_argument("--device", type=str, default="cuda", help="MuJoCo sim device (mj_gen3 only).")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.10", help="Kortex robot IP (kortex only).")
    parser.add_argument(
        "--cameras",
        type=str,
        default="both",
        choices=tuple(CAMERA_CHOICES),
        help="Cameras recorded in the dataset observation.",
    )
    parser.add_argument("--no-video", action="store_true", help="Store images instead of video.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset.")
    parser.add_argument("--display-data", action="store_true", help="Log to Rerun.")
    parser.add_argument("--play-sounds", action="store_true", help="Speak episode prompts.")
    parser.add_argument(
        "--view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the live viewer (MuJoCo window for mj_gen3, camera window for kortex_tcp_gen3).",
    )
    parser.add_argument(
        "--view-wrist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the wrist RGB preview window.",
    )
    parser.add_argument(
        "--view-tabletop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the tabletop RGB preview window.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    use_wrist, use_tabletop = CAMERA_CHOICES[args.cameras]
    robot_overrides: dict = {
        "use_wrist_camera": use_wrist,
        "use_tabletop_camera": use_tabletop,
        "dt": 1.0 / float(args.fps),
    }
    if args.robot == "mj_gen3":
        robot_overrides["device"] = args.device
    else:
        robot_overrides["robot_ip"] = args.robot_ip

    dataset_name = args.dataset_name
    if not args.resume:
        dataset_name = find_available_dataset_name(dataset_name)
        if dataset_name != args.dataset_name:
            print(f"Dataset {args.dataset_name!r} already exists; using {dataset_name!r}.")
    repo_id = args.repo_id or f"local/{dataset_name}"

    cfg = Gen3RecordConfig(
        robot=make_robot_config(args.robot, **robot_overrides),
        teleop=make_teleop_config(
            args.teleop,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        ),
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
        display_data=args.display_data,
        play_sounds=args.play_sounds,
        view=args.view,
        view_wrist=args.view_wrist,
        view_tabletop=args.view_tabletop,
    )

    dataset = record(cfg)
    print(f"Saved {dataset.num_episodes} episodes to {cfg.dataset.resolved_root().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
