"""Build LeRobot Diffusion ``TrainPipelineConfig`` for local Gen3 datasets."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lerobot.configs import FeatureType
from lerobot.configs.default import DatasetConfig, TrackioConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.ema import WarmupEMAConfig
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

    ``image_keys`` defaults to ``["wrist"]`` when ``None``; ``[]`` or ``["none"]`` trains a
    state-only policy conditioned on ``observation.state`` alone (``resize_shape`` is then
    ignored). Features are filtered so the policy only sees selected cameras; pair with
    :func:`filter_cameras_during_train` so unused videos are not decoded.

    ``wandb_enable`` maps to LeRobot's ``TrackioConfig`` (WandB was replaced upstream).
    ``ema_enable`` selects the default warmup EMA schedule when True, else disables EMA.
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
    if image_keys is None:
        image_keys = ["wrist"]
    selected = normalize_image_keys(image_keys or ["none"], available=available)

    info = json.loads((root / "meta" / "info.json").read_text())
    filtered = filter_dataset_features(info["features"], selected)
    policy_features = dataset_to_policy_features(filtered)
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}
    if selected and not any(ft.type is FeatureType.VISUAL for ft in input_features.values()):
        raise ValueError(f"No visual features left after camera filter {selected}.")
    if not any(ft.type is FeatureType.STATE for ft in input_features.values()):
        raise ValueError("Dataset has no observation.state; a diffusion policy needs it.")

    policy = DiffusionConfig(
        device=device,
        push_to_hub=False,
        input_features=input_features,
        output_features=output_features,
        resize_shape=resize_shape if selected else None,
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
        ema=WarmupEMAConfig() if ema_enable else None,
        trackio=TrackioConfig(enable=wandb_enable),
    )
    return cfg


@contextmanager
def filter_cameras_during_train(image_keys: list[str]) -> Iterator[None]:
    """Temporarily strip non-selected image features from loaded dataset metadata.

    LeRobot's dataset reader decodes every ``meta.video_keys`` entry for every sample,
    even cameras the policy does not use. Removing unused ``observation.images.*`` keys
    from in-memory metadata (without rewriting disk) skips decoding those videos for the
    duration of ``train()``. Policy inputs are already limited to the selected cameras by
    :func:`make_diffusion_train_config`; this only saves data-loading work.

    Patches the private :meth:`LeRobotDatasetMetadata._load_metadata` so both
    ``make_dataset`` and ``LeRobotDataset`` see the filtered feature set. Recheck this
    whenever the ``lerobot/`` checkout is updated: if the method is renamed or no longer
    called, the filter silently stops applying and unused cameras are decoded again.
    """
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    keep = set(image_keys)
    original = LeRobotDatasetMetadata._load_metadata

    def _filtered_load(self) -> None:
        original(self)
        features = self.info.features
        for key in list(features):
            if key.startswith("observation.images.") and key not in keep:
                del features[key]

    LeRobotDatasetMetadata._load_metadata = _filtered_load  # type: ignore[method-assign]
    try:
        yield
    finally:
        LeRobotDatasetMetadata._load_metadata = original  # type: ignore[method-assign]


def _is_state_only(config: DiffusionConfig) -> bool:
    """Whether ``config`` conditions on ``observation.state`` alone (no images, no env state)."""
    return not config.image_features and config.env_state_feature is None and config.robot_state_feature is not None


def _allow_state_only_diffusion() -> None:
    """Let LeRobot's diffusion policy train and load as a state-only policy (no cameras).

    The diffusion network itself supports conditioning on ``observation.state`` alone:
    no RGB encoder is built and the global conditioning is just the state vector. Two
    LeRobot checks still demand an image or an ``observation.environment_state`` input:

    - ``DiffusionConfig.validate_features`` (run by the config and again by
      ``DiffusionPolicy.__init__`` when a checkpoint loads). The wrapper skips it for the
      state-only case; every other config goes through LeRobot's original unchanged.
    - A bare ``assert`` in ``DiffusionModel.compute_loss`` (training only). For state-only
      configs the wrapper adds an empty ``(B, n_obs_steps, 0)`` environment-state tensor to
      the batch so the assert passes. ``_prepare_global_conditioning`` only reads that key
      when the config declares an environment state, so the placeholder is never used.

    Installed when this module is imported (``kinova_lfd.imitation`` imports it), covering
    training and checkpoint loading. Recheck this whenever the ``lerobot/`` checkout is
    updated: if these methods are renamed or their checks change, state-only training may
    fail again or other validation may be skipped.
    """
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

    if getattr(DiffusionConfig.validate_features, "_kinova_state_only", False):
        return
    original_validate = DiffusionConfig.validate_features
    original_compute_loss = DiffusionModel.compute_loss

    def validate_features(self: DiffusionConfig) -> None:
        if _is_state_only(self):
            return
        original_validate(self)

    def compute_loss(self: DiffusionModel, batch: dict) -> Any:
        if _is_state_only(self.config) and OBS_IMAGES not in batch and OBS_ENV_STATE not in batch:
            state = batch[OBS_STATE]
            batch = {**batch, OBS_ENV_STATE: state.new_zeros((*state.shape[:2], 0))}
        return original_compute_loss(self, batch)

    validate_features._kinova_state_only = True  # type: ignore[attr-defined]
    DiffusionConfig.validate_features = validate_features  # type: ignore[method-assign]
    DiffusionModel.compute_loss = compute_loss  # type: ignore[method-assign]


_allow_state_only_diffusion()
