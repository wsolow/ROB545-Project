"""play_diffusion.py.

Roll out a trained LeRobot Diffusion Policy in MuJoCo with the same env timing it was recorded with.

The training dataset's metadata sets the playback: ``dt = 1 / fps`` (one env.step per recorded
frame, e.g. 50 ms for 20 fps ``record_gen3`` data), and the action names pick the action spec and
sim task (``tcp.*`` twist -> ``Mj-Gen3-v0``; ``joint_*.pos`` -> ``Mj-Gen3-LiftCube-v0``).

Uses overlapping realtime action chunks: the control loop keeps stepping while the next diffusion
inference runs asynchronously (control at the dataset fps / 20 Hz replan / 0.5 overlap by default).

Example::

    python3 scripts/imitation/play_diffusion.py \\
      --checkpoint outputs/train/diffusion_gen3_policy_demo_1/checkpoints/001000/pretrained_model \\
      --episodes 3 --max-steps 300 --device cuda --inference-steps 16

Written by Will Solow, 2026.
"""

import argparse
import sys
import time

import torch
from lerobot.policies.utils import build_inference_frame, make_robot_action

from kinova_lfd.data_collection import make_robot, make_robot_config
from kinova_lfd.data_collection.policy import cube_goal_success
from kinova_lfd.imitation.play import (
    load_dataset_info,
    load_diffusion_policy,
    load_play_features,
    playback_settings,
    policy_image_keys,
)
from kinova_lfd.imitation.realtime import RealtimeDiffusionController, pace_control_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a Diffusion Policy in MuJoCo at its recording timing.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained_model/ or a checkpoint step dir containing it.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="LeRobot dataset used for training (feature names). Default: root from train_config.json.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for the policy.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=290,
        help="Max control steps per episode (5.8 s at 50 fps, 14.5 s at 20 fps).",
    )
    parser.add_argument(
        "--success-delta",
        type=float,
        default=0.1,
        help="Cube-goal distance (m) treated as success; <=0 disables early stop.",
    )
    parser.add_argument("--task", type=str, default="Lift the cube to the goal", help="Language task string.")
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Wall-clock control rate (default: the dataset fps, matching one env.step per dt).",
    )

    parser.add_argument(
        "--replan-hz",
        type=float,
        default=20.0,
        help="Target diffusion replan rate (default 20). Chunk length = 2*control_hz/replan_hz.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fraction of the chunk to execute before starting the next async replan (default 0.5).",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=16,
        help="Denoising steps per replan (default 16; checkpoint train default is often 100).",
    )

    return parser


def _episode_succeeded(env: object, success_delta: float) -> bool:
    if success_delta <= 0:
        return False
    cube_pose = getattr(env, "last_cube_pose_w", getattr(env, "cube_pose_w", None))
    goal_pose = getattr(env, "last_cube_goal_pose_w", getattr(env, "cube_goal_pose_w", None))
    if cube_pose is None or goal_pose is None:
        return False
    return cube_goal_success(cube_pose, goal_pose, success_delta)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    policy, preprocessor, postprocessor, pretrained_dir = load_diffusion_policy(
        args.checkpoint,
        device=str(device),
        inference_steps=args.inference_steps,
    )
    ds_features = load_play_features(
        pretrained_dir=pretrained_dir,
        policy=policy,
        dataset=args.dataset,
    )
    image_keys = policy_image_keys(policy)
    use_wrist = any(k.endswith(".wrist") for k in image_keys)
    use_tabletop = any(k.endswith(".tabletop") for k in image_keys)

    # Reproduce the recording timing: one env.step per recorded frame (dt = 1 / dataset fps).
    task_name, action_spec, dt = playback_settings(
        load_dataset_info(pretrained_dir=pretrained_dir, dataset=args.dataset)
    )
    control_hz = args.control_hz if args.control_hz is not None else 1.0 / dt
    if abs(control_hz * dt - 1.0) > 1e-6:
        print(
            f"[play] warning: --control-hz {control_hz:g} differs from the dataset rate {1.0 / dt:g} Hz; "
            "the sim still advances dt per step, so playback runs faster/slower than real time."
        )

    controller = RealtimeDiffusionController(
        policy,
        control_hz=control_hz,
        replan_hz=args.replan_hz,
        overlap=args.overlap,
        verbose=True,
    )

    robot_overrides: dict = {}
    if task_name == "Mj-Gen3-v0":
        # Same env settings as record_gen3: episodes are delimited by this script (env.reset per
        # episode), not by the task time limit.
        robot_overrides["episode_length_s"] = 1e9
        robot_overrides["interpolate_joint_targets"] = False

    robot_cfg = make_robot_config(
        "mj_gen3",
        task_name=task_name,
        device=str(device),
        enable_cameras=True,
        use_wrist_camera=use_wrist,
        use_tabletop_camera=use_tabletop,
        action_spec=action_spec,
        dt=dt,
        **robot_overrides,
    )
    robot = make_robot(robot_cfg)
    robot.connect()

    if robot.env is not None:
        from kinova_lfd.envs.base.viewers.mujoco_viewer import MjViewer

        mj_viewer = MjViewer(
            robot.env,
            policy=None,
            view_wrist=True,
            view_tabletop=True,
            ignore_mjlab_keys=True,
        )
        mj_viewer.open_display()
        print("[play] MuJoCo viewer opened (syncs each control step).")

    print(
        f"\nPlaying diffusion policy (realtime chunks):\n"
        f"  checkpoint:      {pretrained_dir}\n"
        f"  cameras:         {[k.removeprefix('observation.images.') for k in image_keys]}\n"
        f"  device:          {device}\n"
        f"  sim:             {task_name}, {action_spec}, dt={dt * 1000:g} ms "
        f"(decimation {robot.env.cfg.decimation} x {robot.env.physics_dt * 1000:g} ms)\n"
        f"  control/replan:  {control_hz:g} Hz / {args.replan_hz:g} Hz "
        f"(chunk={controller.chunk_len}, replan_after={controller.replan_after})\n"
        f"  inference_steps: {policy.diffusion.num_inference_steps}\n"
        f"  episodes:        {args.episodes} x {args.max_steps} steps\n"
    )

    successes = 0
    try:
        for ep in range(args.episodes):
            assert robot.env is not None
            robot.env.reset()
            controller.reset()

            success = False
            for step in range(args.max_steps):
                step_start = time.perf_counter()
                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=ds_features,
                    device=device,
                    task=args.task,
                    robot_type=robot.name,
                )
                obs_proc = preprocessor(obs_frame)
                action = controller.step(obs_proc)
                action = postprocessor(action)
                robot_action = make_robot_action(action, ds_features)
                robot.send_action(robot_action)

                if mj_viewer is not None:
                    mj_viewer.sync_display()

                if _episode_succeeded(robot.env, args.success_delta):
                    success = True
                    break

                pace_control_loop(step_start, control_hz)

            successes += int(success)
            print(
                f"Episode {ep + 1}/{args.episodes}: "
                f"{'SUCCESS' if success else 'done'} "
                f"(step {step + 1}/{args.max_steps})"
            )
    finally:
        controller.reset()
        if mj_viewer is not None:
            mj_viewer.close_display()
        robot.disconnect()

    print(f"Done. Successes: {successes}/{args.episodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
