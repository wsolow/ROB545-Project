"""Patches so unused cameras are not decoded during LeRobot training."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def filter_cameras_during_train(image_keys: list[str]) -> Iterator[None]:
    """Temporarily strip non-selected image features from loaded dataset metadata.

    LeRobot's dataset reader decodes every ``meta.video_keys`` entry. Removing
    unused ``observation.images.*`` keys from in-memory metadata (without
    rewriting disk) skips those videos for the duration of ``train()``.

    Patches :meth:`LeRobotDatasetMetadata._load_metadata` so both
    ``make_dataset`` and ``LeRobotDataset`` see the filtered feature set.
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
