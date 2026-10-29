"""train_diffusion.py.

Train a LeRobot Diffusion Policy on a local dataset under ``_data/``.

Example::

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/train_diffusion.py \\
      --dataset gen3_demo --device cuda \\
      --cameras wrist --resize-h 96 --resize-w 128 --batch-size 4

    PYTHONPATH=lerobot/src:$PYTHONPATH python scripts/train_diffusion.py \\
      --dataset gen3_demo --validate-only

Written by Will Solow, 2026.
"""

from __future__ import annotations

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
from kinova_lfd.imitation.diffusion_config import make_diffusion_train_config
from kinova_lfd.imitation.train_patch import filter_cameras_during_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LeRobot Diffusion Policy on a local _data/ dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name under _data/ (e.g. gen3_demo) or path to a LeRobot root.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the dataset, then exit without training.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for the policy.")
    parser.add_argument("--steps", type=int, default=80_000, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default 4 for 12GB GPUs).")
    parser.add_argument("--save-freq", type=int, default=20_000, help="Checkpoint every N steps.")
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
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist"],
        help="Cameras to train on (short names or full keys). Default: wrist. Use 'all' for every cam.",
    )
    parser.add_argument("--resize-h", type=int, default=96, help="Resize height for vision backbone.")
    parser.add_argument("--resize-w", type=int, default=128, help="Resize width for vision backbone.")
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Keep native image resolution (uses more VRAM).",
    )
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable EMA weights (uses ~2x model VRAM).",
    )
    parser.add_argument(
        "--shared-rgb-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share one ResNet across cameras (default on).",
    )
    parser.add_argument(
        "--return-uint8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode images as uint8 for cheaper DataLoader IPC (default on).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint Unet blocks to save activation memory (default on).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_id, root = resolve_local_dataset(args.dataset)
    summary = validate_lerobot_dataset(root, repo_id=repo_id)
    print(format_dataset_summary(summary))

    available = list_dataset_image_keys(root)
    selected = normalize_image_keys(args.cameras, available=available)
    print(f"  training cameras: {selected}")

    if args.validate_only:
        return 0

    resize_shape = None if args.no_resize else (args.resize_h, args.resize_w)

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
        wandb_enable=args.wandb,
        image_keys=args.cameras,
        resize_shape=resize_shape,
        ema_enable=args.ema,
        use_separate_rgb_encoder_per_camera=not args.shared_rgb_encoder,
        return_uint8=args.return_uint8,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    print(
        f"\nStarting diffusion training:\n"
        f"  dataset:  {repo_id} @ {root}\n"
        f"  cameras:  {selected}\n"
        f"  resize:   {resize_shape}\n"
        f"  output:   {cfg.output_dir}\n"
        f"  steps:    {cfg.steps}  batch: {cfg.batch_size}  device: {cfg.policy.device}\n"
        f"  ema:      {cfg.ema.enable}  grad_ckpt: {cfg.policy.gradient_checkpointing}\n"
    )

    from lerobot.scripts.lerobot_train import train

    with filter_cameras_during_train(selected):
        train(cfg)

    print(f"Done. Checkpoints under {Path(cfg.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
