"""train_diffusion.py.

Train a LeRobot Diffusion Policy on a local dataset under ``_data/``.

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/train_diffusion.py \\
      --dataset gen3_demo --device cuda \\
      --cameras wrist --resize-h 96 --resize-w 128 --batch-size 4

    # State-only policy (joint positions + gripper + EE pose from observation.state; no images)
    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/train_diffusion.py \\
      --dataset gen3_demo --device cuda --cameras none

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/train_diffusion.py \\
      --dataset gen3_demo --validate-only

Written by Will Solow, 2026.
"""

import argparse
import sys
from pathlib import Path

from kinova_lfd.imitation.dataset import (
    format_dataset_summary,
    list_dataset_image_keys,
    normalize_image_keys,
    resolve_local_dataset,
    validate_lerobot_dataset,
)
from kinova_lfd.imitation.diffusion_config import filter_cameras_during_train, make_diffusion_train_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LeRobot Diffusion Policy on a local _data/ dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name under _data/ (e.g. gen3_demo) or path to a LeRobot root.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for the policy.")
    parser.add_argument("--steps", type=int, default=80000, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4 for 12GB GPUs).")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint every N steps.")
    parser.add_argument("--seed", type=int, default=1000, help="Training seed.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--log-freq", type=int, default=200, help="Log every N steps.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint root (default: outputs/train/diffusion_<name>).",
    )
    parser.add_argument("--job-name", type=str, default=None, help="Run / job name.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist"],
        help=(
            "Cameras to train on (short names or full keys). Default: wrist. Use 'all' for every cam, "
            "or 'none' for a state-only policy conditioned on observation.state alone."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_id, root = resolve_local_dataset(args.dataset)
    summary = validate_lerobot_dataset(root, repo_id=repo_id)
    print(format_dataset_summary(summary))

    available = list_dataset_image_keys(root)
    selected = normalize_image_keys(args.cameras, available=available)
    state_only = not selected
    print(f"  training cameras: {selected or 'none (state-only)'}")

    resize_shape = (96, 128)

    cfg = make_diffusion_train_config(
        repo_id=repo_id,
        root=root,
        output_dir=args.output_dir,
        job_name=args.job_name,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        seed=args.seed,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        wandb_enable=False,
        image_keys=selected,
        resize_shape=resize_shape,
        ema_enable=False,
        use_separate_rgb_encoder_per_camera=False,
        return_uint8=True,
        gradient_checkpointing=False,
    )

    if state_only:
        state_dim = cfg.policy.robot_state_feature.shape[0]
        inputs = f"  cameras:  none (state-only, observation.state dim {state_dim})\n"
    else:
        inputs = f"  cameras:  {selected}\n  resize:   {resize_shape}\n"
    print(
        f"\nStarting diffusion training:\n"
        f"  dataset:  {repo_id} @ {root}\n"
        f"{inputs}"
        f"  output:   {cfg.output_dir}\n"
        f"  steps:    {cfg.steps}  batch: {cfg.batch_size}  device: {cfg.policy.device}\n"
        f"  ema:      {None if cfg.ema is None else cfg.ema.type}  "
        f"grad_ckpt: {cfg.policy.gradient_checkpointing}\n"
    )

    from lerobot.scripts.lerobot_train import train

    # With no cameras selected this strips every image feature, so no video is decoded.
    with filter_cameras_during_train(selected):
        train(cfg)

    print(f"Done. Checkpoints under {Path(cfg.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
