"""Build LeRobot Diffusion ``TrainPipelineConfig`` for local Gen3 datasets."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.configs import FeatureType
from lerobot.configs.default import DatasetConfig, EMAConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.diffusion import DiffusionConfig
from lerobot.utils.feature_utils import dataset_to_policy_features

from .dataset import filter_dataset_features, list_dataset_image_keys, normalize_image_keys


def make_diffusion_train_config(
    *,
    repo_id: str,
    root: Path | str,
    output_dir: Path | str | None = None,
    job_name: str | None = None,
    device: str = "cuda",
    steps: int = 80_000,
    batch_size: int = 4,
    save_freq: int = 20_000,
    seed: int = 1000,
    num_workers: int = 2,
    log_freq: int = 200,
    wandb_enable: bool = False,
    image_keys: list[str] | None = None,
    resize_shape: tuple[int, int] | None = (96, 128),
    ema_enable: bool = False,
    use_separate_rgb_encoder_per_camera: bool = False,
    return_uint8: bool = True,
    gradient_checkpointing: bool = True,
) -> TrainPipelineConfig:
    """Create an offline Diffusion training config (no env eval, no Hub push).

    ``image_keys`` defaults to ``["wrist"]``. Features are filtered so the policy
    only sees selected cameras; pair with :func:`filter_cameras_during_train` so
    unused videos are not decoded.
    """
    root = Path(root)
    name = root.name
    if output_dir is None:
        output_dir = Path("outputs/train") / f"diffusion_{name}"
    else:
        output_dir = Path(output_dir)
    if job_name is None:
        job_name = f"diffusion_{name}"

    available = list_dataset_image_keys(root)
    selected = normalize_image_keys(image_keys or ["wrist"], available=available)

    info = json.loads((root / "meta" / "info.json").read_text())
    filtered = filter_dataset_features(info["features"], selected)
    policy_features = dataset_to_policy_features(filtered)
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}
    if not any(ft.type is FeatureType.VISUAL for ft in input_features.values()):
        raise ValueError(f"No visual features left after camera filter {selected}.")

    policy = DiffusionConfig(
        device=device,
        push_to_hub=False,
        input_features=input_features,
        output_features=output_features,
        resize_shape=resize_shape,
        use_separate_rgb_encoder_per_camera=use_separate_rgb_encoder_per_camera,
        gradient_checkpointing=gradient_checkpointing,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=repo_id,
            root=str(root.resolve()),
            return_uint8=return_uint8,
        ),
        env=None,
        policy=policy,
        output_dir=output_dir,
        job_name=job_name,
        seed=seed,
        num_workers=num_workers,
        batch_size=batch_size,
        steps=steps,
        env_eval_freq=0,
        log_freq=log_freq,
        save_freq=save_freq,
        ema=EMAConfig(enable=ema_enable),
        wandb=WandBConfig(enable=wandb_enable),
    )
    # Stash selected cameras for the train script / patch (not a LeRobot field).
    cfg._kinova_image_keys = selected  # type: ignore[attr-defined]
    return cfg
