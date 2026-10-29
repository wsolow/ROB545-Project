"""record_gen3.py.

Record Gen3 demonstrations (MuJoCo or Kortex TCP) with Gen3 keyboard or stock
LeRobot teleops. Episodes are saved under ``_data/<dataset_name>/``.

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/record_gen3.py
      --robot mj_gen3 --teleop gen3_keyboard
      --dataset-name pick_pink_block --task "Pick the pink block"
      --fps 20 --num-episodes 5

Written by Will Solow, 2026.
"""

from __future__ import annotations

import argparse
import sys

from kinova_lfd.data_collection import (
    Gen3DatasetConfig,
    Gen3RecordConfig,
    make_robot_config,
    make_teleop_config,
    record,
    register_gen3_robots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Gen3 demos with LeRobot teleops.")
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
        default="gen3_keyboard",
        choices=("gen3_keyboard", "keyboard_ee", "gamepad"),
        help="Teleoperator (gen3_keyboard = 6-DoF letter keys; keyboard_ee/gamepad = stock LeRobot).",
    )
    parser.add_argument("--dataset-name", type=str, default="gen3_demo", help="Folder under _data/.")
    parser.add_argument("--repo-id", type=str, default=None, help="Dataset repo id (default local/<name>).")
    parser.add_argument("--task", type=str, default="Gen3 demonstration", help="Task description.")
    parser.add_argument("--fps", type=int, default=20, help="Recording / control rate.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--episode-time", type=float, default=60.0, help="Seconds per episode.")
    parser.add_argument("--reset-time", type=float, default=10.0, help="Unrecorded reset seconds.")
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=0.15,
        help="Scale stock keyboard_ee/gamepad deltas to twist (ignored for gen3_keyboard).",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=0.15,
        help="Linear twist magnitude for gen3_keyboard (m/s-ish).",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=0.15,
        help="Angular twist magnitude for gen3_keyboard (rad/s-ish).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="MuJoCo sim device.")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.10", help="Kortex robot IP.")
    parser.add_argument("--no-video", action="store_true", help="Store images instead of video.")
    parser.add_argument("--no-cameras", action="store_true", help="Disable wrist/tabletop in obs.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing dataset.")
    parser.add_argument("--display-data", action="store_true", help="Log to Rerun.")
    parser.add_argument("--play-sounds", action="store_true", help="Speak episode prompts.")
    parser.add_argument(
        "--view-mujoco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open MjViewer while recording (mj_gen3 only).",
    )
    parser.add_argument(
        "--view-wrist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show wrist RGB OpenCV window in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--view-tabletop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show tabletop RGB OpenCV window in the MuJoCo viewer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    register_gen3_robots()

    use_cams = not args.no_cameras
    robot_overrides: dict = {
        "use_wrist_camera": use_cams,
        "use_tabletop_camera": use_cams,
        "action_spec": "TWIST_TCP",
    }
    if args.robot == "mj_gen3":
        # One env.step advances ~4 ms of sim; substep so wall-clock fps ≈ realtime.
        mj_step_dt = 0.004
        robot_overrides["device"] = args.device
        robot_overrides["physics_steps_per_action"] = max(1, round((1.0 / float(args.fps)) / mj_step_dt))
    else:
        robot_overrides["robot_ip"] = args.robot_ip
        robot_overrides["dt"] = 1.0 / float(args.fps)

    teleop_overrides: dict = {}
    if args.teleop == "gen3_keyboard":
        teleop_overrides["pos_sensitivity"] = args.pos_sensitivity
        teleop_overrides["rot_sensitivity"] = args.rot_sensitivity
    else:
        teleop_overrides["use_gripper"] = True

    dataset_name = args.dataset_name
    repo_id = args.repo_id or f"local/{dataset_name}"

    cfg = Gen3RecordConfig(
        robot=make_robot_config(args.robot, **robot_overrides),
        teleop=make_teleop_config(args.teleop, **teleop_overrides),
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
        linear_scale=args.linear_scale,
        display_data=args.display_data,
        play_sounds=args.play_sounds,
        resume=args.resume,
        view_mujoco=args.view_mujoco,
        view_wrist=args.view_wrist,
        view_tabletop=args.view_tabletop,
    )

    dataset = record(cfg)
    print(f"Saved {dataset.num_episodes} episodes to {cfg.dataset.resolved_root().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
