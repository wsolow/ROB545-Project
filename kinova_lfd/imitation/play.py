"""Load a local LeRobot Diffusion checkpoint for MuJoCo / robot rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline

from .dataset import filter_dataset_features, resolve_local_dataset


def resolve_pretrained_dir(path: str | Path) -> Path:
    """Resolve a checkpoint path to the ``pretrained_model`` directory.

    Accepts either the ``pretrained_model`` folder itself or a step directory
    that contains ``pretrained_model/`` (and usually ``training_state/``).
    """
    raw = Path(path).expanduser().resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {raw}")

    if (raw / "model.safetensors").is_file() and (raw / "config.json").is_file():
        return raw

    nested = raw / "pretrained_model"
    if (nested / "model.safetensors").is_file() and (nested / "config.json").is_file():
        return nested.resolve()

    raise FileNotFoundError(
        f"No pretrained model found at {raw}. Expected model.safetensors + config.json "
        "either in this directory or under pretrained_model/."
    )


def load_diffusion_policy(
    checkpoint: str | Path,
    device: str = "cuda",
    inference_steps: int | None = None,
) -> tuple[
    DiffusionPolicy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline,
    Path,
]:
    """Load ``DiffusionPolicy`` + pre/post processors from a local checkpoint.

    Returns ``(policy, preprocessor, postprocessor, pretrained_dir)``.

    ``inference_steps`` overrides the denoising step count used at rollout time
    (smaller = faster replans; checkpoint default is often 100 DDPM steps).
    """
    pretrained_dir = resolve_pretrained_dir(checkpoint)
    policy = DiffusionPolicy.from_pretrained(str(pretrained_dir))
    policy.config.device = device
    policy.to(device)
    policy.eval()
    if inference_steps is not None:
        if inference_steps < 1:
            raise ValueError(f"inference_steps must be >= 1, got {inference_steps}")
        policy.diffusion.num_inference_steps = int(inference_steps)
        policy.config.num_inference_steps = int(inference_steps)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, pretrained_dir


def dataset_root_from_train_config(pretrained_dir: Path) -> Path | None:
    """Return the local dataset root stored in ``train_config.json``, if present."""
    train_cfg_path = pretrained_dir / "train_config.json"
    if not train_cfg_path.is_file():
        return None
    try:
        train_cfg = json.loads(train_cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    root = (train_cfg.get("dataset") or {}).get("root")
    if not root:
        return None
    path = Path(root).expanduser()
    return path if (path / "meta" / "info.json").is_file() else None


def load_play_features(
    *,
    pretrained_dir: Path,
    policy: DiffusionPolicy,
    dataset: str | Path | None = None,
) -> dict[str, Any]:
    """Load dataset feature metadata filtered to the policy's cameras + action/state.

    Prefers ``--dataset`` when given; otherwise uses ``train_config.json`` root.
    Image keys not present in ``policy.config.input_features`` are dropped so
    tabletop (etc.) is not required at inference when training was wrist-only.
    """
    if dataset is not None:
        _, root = resolve_local_dataset(dataset)
    else:
        root = dataset_root_from_train_config(pretrained_dir)
        if root is None:
            raise FileNotFoundError(
                "Could not resolve dataset features. Pass --dataset pointing at the "
                "LeRobot dataset used for training (e.g. _data/gen3_policy_demo_1)."
            )

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features") or {}
    image_keys = policy_image_keys(policy)
    return filter_dataset_features(features, image_keys)


def policy_image_keys(policy: DiffusionPolicy) -> list[str]:
    """Full ``observation.images.*`` keys expected by the policy."""
    return [k for k in (policy.config.input_features or {}) if k.startswith("observation.images.")]
