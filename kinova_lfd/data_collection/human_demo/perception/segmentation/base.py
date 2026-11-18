"""Segmentation interface, config, and factory.

A :class:`Segmenter` segments every object (one text concept each) through one episode and
writes a :class:`SegmentationResult` ``.npz``. Backends are chosen by ``SegmenterCfg.name``.

Written by Will Solow, 2026.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinova_lfd.data_collection.human_demo.perception.types import ObjectSpec

SEGMENTERS = ("sam3",)


@dataclass
class SegmenterCfg:
    """Settings shared by every segmenter."""

    name: str = ""
    """Backend to create, one of ``SEGMENTERS``."""


class Segmenter(ABC):
    """Segment objects through a whole episode (video-level, so trackers keep their memory)."""

    def __init__(self, cfg: SegmenterCfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def segment_episode(self, episode_dir: Path, objects: Sequence[ObjectSpec], out_path: Path) -> dict[str, Any]:
        """Segment ``objects`` in the raw episode at ``episode_dir`` and save the result to ``out_path``.

        Returns:
            A summary dict (frames, per-object detected frame counts, seconds, ...).

        """
        raise NotImplementedError

    def check_ready(self) -> dict[str, Any]:
        """Raise if the backend is unavailable (e.g. its server is down); return backend info."""
        return {}

    def close(self) -> None:
        """Release resources."""


def make_segmenter(cfg: SegmenterCfg) -> Segmenter:
    """Construct the segmenter named by ``cfg.name`` (backend modules are imported lazily)."""
    name = cfg.name.lower()
    if name == "sam3":
        from kinova_lfd.data_collection.human_demo.perception.segmentation.sam3 import Sam3Cfg, Sam3Segmenter

        assert isinstance(cfg, Sam3Cfg), f"Segmenter 'sam3' needs a Sam3Cfg, got {type(cfg).__name__}"
        return Sam3Segmenter(cfg)
    raise ValueError(f"Unsupported segmenter {cfg.name!r}. Use one of {SEGMENTERS}.")
