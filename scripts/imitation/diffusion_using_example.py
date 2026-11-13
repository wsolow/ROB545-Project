import argparse
import sys
import time

import torch
from lerobot.policies.utils import build_inference_frame, make_robot_action

from kinova_lfd.data_collection import make_robot, make_robot_config, register_gen3_robots
from kinova_lfd.imitation.play import load_diffusion_policy, load_play_features, policy_image_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a Diffusion Policy on Mj-Gen3-LiftCube-v0.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/diffusion_gen3_policy_demo_3/checkpoints/080000/pretrained_model",
        help="Path to a pretrained_model directory or checkpoint step directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Local LeRobot dataset used for training; defaults to checkpoint train_config.json.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for the policy.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--task", default="Lift the cube to the goal")
    parser.add_argument("--inference-steps", type=int, default=16)
    parser.add_argument(
        "--view-mujoco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open and synchronize the MuJoCo viewer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    register_gen3_robots()
    model, preprocessor, postprocessor, pretrained_dir = load_diffusion_policy(
        args.checkpoint,
        device=str(device),
        inference_steps=args.inference_steps,
    )
    dataset_features = load_play_features(
        pretrained_dir=pretrained_dir,
        policy=model,
        dataset=args.dataset,
    )

    image_keys = policy_image_keys(model)
    if image_keys != ["observation.images.wrist"]:
        raise ValueError(f"This example expects the wrist-only Gen3 checkpoint, got image keys: {image_keys}")

    robot_config = make_robot_config(
        "mj_gen3",
        task_name="Mj-Gen3-LiftCube-v0",
        device=str(device),
        enable_cameras=True,
        use_wrist_camera=True,
        use_tabletop_camera=False,
        action_spec="JOINTS_POS",
        physics_steps_per_action=1,
    )
    robot = make_robot(robot_config)
    robot.connect()

    viewer = None
    if args.view_mujoco and robot.env is not None:
        from kinova_lfd.envs.base.viewers.mujoco_viewer import MjViewer

        viewer = MjViewer(robot.env, policy=None, view_wrist=True, view_tabletop=False, ignore_mjlab_keys=True)
        viewer.open_display()

    try:
        print(f"Playing {pretrained_dir} on Mj-Gen3-LiftCube-v0 with {image_keys}, device={device}")
        for episode in range(args.episodes):
            assert robot.env is not None
            robot.env.reset()
            model.reset()

            for step in range(args.max_steps):
                step_start = time.perf_counter()
                observation_frame = build_inference_frame(
                    observation=robot.get_observation(),
                    ds_features=dataset_features,
                    device=device,
                    task=args.task,
                    robot_type="mj_gen3",
                )
                action = model.select_action(preprocessor(observation_frame))
                action = postprocessor(action)
                robot.send_action(make_robot_action(action, dataset_features))

                if viewer is not None:
                    viewer.sync_display()

                elapsed = time.perf_counter() - step_start
                if elapsed < 1.0 / 50.0:
                    time.sleep(1.0 / 50.0 - elapsed)

            print(f"Episode {episode + 1}/{args.episodes} complete")
    finally:
        if viewer is not None:
            viewer.close_display()
        robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
