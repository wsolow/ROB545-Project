"""Resolve and validate local LeRobot datasets under ``_data/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset

_DEFAULT_DATA_ROOT = Path("_data")
_IMAGE_PREFIX = "observation.images."


def list_dataset_image_keys(root: Path | str) -> list[str]:
    """Return full ``observation.images.*`` feature keys from ``meta/info.json``."""
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return sorted(k for k in info.get("features", {}) if k.startswith(_IMAGE_PREFIX))


def normalize_image_keys(
    names: list[str] | None,
    *,
    available: list[str] | None = None,
) -> list[str]:
    """Normalize camera names to full feature keys.

    Accepts short names (``wrist``), full keys, or ``["all"]`` to keep every
    available camera. Raises if a requested camera is missing from ``available``.
    """
    if available is None:
        available = []
    available_set = set(available)

    if not names or names == ["all"]:
        if not available:
            raise ValueError("No cameras available in the dataset.")
        return list(available)

    resolved: list[str] = []
    for name in names:
        if name == "all":
            return list(available)
        key = name if name.startswith(_IMAGE_PREFIX) else f"{_IMAGE_PREFIX}{name}"
        if available_set and key not in available_set:
            short = sorted(k.removeprefix(_IMAGE_PREFIX) for k in available)
            raise ValueError(f"Camera {name!r} not in dataset. Available: {short}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def filter_dataset_features(features: dict[str, Any], image_keys: list[str]) -> dict[str, Any]:
    """Drop ``observation.images.*`` keys not listed in ``image_keys``."""
    keep = set(image_keys)
    out: dict[str, Any] = {}
    for key, value in features.items():
        if key.startswith(_IMAGE_PREFIX) and key not in keep:
            continue
        out[key] = value
    return out


def resolve_local_dataset(name_or_path: str | Path) -> tuple[str, Path]:
    """Map a dataset name or path to ``(repo_id, root)``.

    Accepts ``gen3_demo``, ``_data/gen3_demo``, or an absolute path to a LeRobot
    dataset root. ``repo_id`` defaults to ``local/<dirname>`` (matching
    ``record_gen3.py``).
    """
    raw = Path(name_or_path).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute() or raw.exists():
        candidates.append(raw.resolve())
    else:
        candidates.append((_DEFAULT_DATA_ROOT / raw).resolve())
        candidates.append(raw.resolve())

    root: Path | None = None
    for cand in candidates:
        if (cand / "meta" / "info.json").is_file():
            root = cand
            break
    if root is None:
        tried = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(f"No LeRobot dataset (missing meta/info.json) for {name_or_path!r}. Tried: {tried}")

    repo_id = f"local/{root.name}"
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
        # Some recordings may store a hub-style id; prefer it when present.
        stored = info.get("repo_id")
        if isinstance(stored, str) and stored.strip():
            repo_id = stored.strip()
    except (OSError, json.JSONDecodeError):
        pass

    return repo_id, root


def validate_lerobot_dataset(root: Path | str, *, repo_id: str | None = None) -> dict[str, Any]:
    """Load a local dataset and assert the features needed for diffusion BC.

    Returns a summary dict (episodes, frames, fps, feature shapes) for logging.
    """
    root = Path(root)
    if repo_id is None:
        repo_id, root = resolve_local_dataset(root)

    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing meta/info.json under {root}")

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    features = dataset.meta.features
    feature_keys = set(features.keys())

    missing: list[str] = []
    if "action" not in feature_keys:
        missing.append("action")
    if "observation.state" not in feature_keys:
        missing.append("observation.state")
    image_keys = sorted(k for k in feature_keys if k.startswith("observation.images."))
    if not image_keys:
        missing.append("observation.images.* (at least one camera)")
    if missing:
        raise ValueError(f"Dataset at {root} is missing required features: {missing}")

    sample = dataset[0]
    for key in ("action", "observation.state", *image_keys):
        if key not in sample:
            raise ValueError(f"Dataset __getitem__ missing key {key!r}")

    summary: dict[str, Any] = {
        "repo_id": repo_id,
        "root": str(root.resolve()),
        "codebase_version": (
            getattr(dataset.meta, "info", {}).get("codebase_version") if hasattr(dataset.meta, "info") else None
        ),
        "fps": dataset.fps,
        "total_episodes": dataset.num_episodes,
        "total_frames": dataset.num_frames,
        "robot_type": getattr(dataset.meta, "robot_type", None),
        "features": {
            key: {
                "dtype": features[key].get("dtype"),
                "shape": list(features[key].get("shape", [])),
            }
            for key in ("action", "observation.state", *image_keys)
        },
        "image_keys": image_keys,
        "sample_shapes": {
            "action": tuple(sample["action"].shape),
            "observation.state": tuple(sample["observation.state"].shape),
            **{k: tuple(sample[k].shape) for k in image_keys},
        },
    }

    # Prefer info.json version when meta wrapper differs.
    info = json.loads((root / "meta" / "info.json").read_text())
    summary["codebase_version"] = info.get("codebase_version")
    summary["robot_type"] = info.get("robot_type", summary["robot_type"])

    return summary


def format_dataset_summary(summary: dict[str, Any]) -> str:
    """Pretty-print a :func:`validate_lerobot_dataset` summary."""
    lines = [
        f"Dataset OK: {summary['repo_id']}",
        f"  root: {summary['root']}",
        f"  version: {summary.get('codebase_version')}  robot: {summary.get('robot_type')}",
        f"  episodes: {summary['total_episodes']}  frames: {summary['total_frames']}  fps: {summary['fps']}",
        "  features:",
    ]
    for key, meta in summary["features"].items():
        sample_shape = summary["sample_shapes"].get(key)
        lines.append(f"    {key}: dtype={meta['dtype']} meta_shape={meta['shape']} sample={sample_shape}")
    return "\n".join(lines)
