"""Record successful Gen3 lift-cube episodes from an RSL-RL policy."""

from __future__ import annotations

import argparse
import sys

from kinova_lfd.data_collection import (
    Gen3DatasetConfig,
    Gen3RecordConfig,
    find_available_dataset_name,
    make_policy_config,
    make_robot_config,
    record,
    register_gen3_robots,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the policy recording CLI."""
    parser = argparse.ArgumentParser(description="Record successful Gen3 lift-cube policy episodes.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="_logs/rsl_rl/mj_gen3/2026-11-04_12-50-24/model_700.pt",
        help="RSL-RL checkpoint path.",
    )
    parser.add_argument("--dataset-name", type=str, default="gen3_policy_demo")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--task-description", type=str, default="Lift the cube to the goal")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of successful episodes to save.")
    parser.add_argument("--episode-time", type=float, default=5.8)
    parser.add_argument("--reset-time", type=float, default=0.0)
    parser.add_argument("--success-delta", type=float, default=0.1, help="Maximum cube-goal distance in meters.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--view-mujoco", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--view-wrist", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--view-tabletop", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    """Run policy-driven recording."""
    args = build_parser().parse_args()
    register_gen3_robots()
    dataset_name = find_available_dataset_name(args.dataset_name)
    if dataset_name != args.dataset_name:
        print(f"Dataset {args.dataset_name!r} already exists; using {dataset_name!r}.")
    robot = make_robot_config(
        "mj_gen3",
        task_name="Mj-Gen3-LiftCube-v0",
        device=args.device,
        enable_cameras=True,
        use_wrist_camera=True,
        use_tabletop_camera=True,
        action_spec="JOINTS_POS",
        physics_steps_per_action=1,
    )
    cfg = Gen3RecordConfig(
        robot=robot,
        teleop=None,
        policy=make_policy_config(
            args.checkpoint,
            task="Mj-Gen3-LiftCube-v0",
            device=args.device,
            success_delta=args.success_delta,
        ),
        dataset=Gen3DatasetConfig(
            dataset_name=dataset_name,
            repo_id=args.repo_id or f"local/{dataset_name}",
            single_task=args.task_description,
            fps=args.fps,
            episode_time_s=args.episode_time,
            reset_time_s=args.reset_time,
            num_episodes=args.num_episodes,
            video=not args.no_video,
        ),
        view_mujoco=args.view_mujoco,
        view_wrist=args.view_wrist,
        view_tabletop=args.view_tabletop,
    )
    dataset = record(cfg)
    print(f"Saved {dataset.num_episodes} successful episodes to {cfg.dataset.resolved_root().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
