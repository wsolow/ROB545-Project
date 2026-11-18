"""Shared helpers for keyboard teleop + diffusion-policy handoff scripts."""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.robots import Robot

from kinova_lfd.imitation.play import (
    load_dataset_info,
    load_diffusion_policy,
    load_play_features,
    playback_settings,
    policy_image_keys,
)
from kinova_lfd.imitation.realtime import RealtimeDiffusionController

_IMAGE_PREFIX = "observation.images."


@dataclass
class DiffusionBundle:
    """Everything loaded from a diffusion checkpoint that a rollout needs."""

    policy: DiffusionPolicy
    preprocessor: PolicyProcessorPipeline
    postprocessor: PolicyProcessorPipeline
    ds_features: dict[str, Any]
    image_keys: list[str]
    task_name: str
    action_spec: str
    dt: float
    pretrained_dir: Path
    device: torch.device


def add_diffusion_args(parser: argparse.ArgumentParser) -> None:
    """Add diffusion-policy related arguments to the argparser."""
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
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=16,
        help="Denoising steps per replan (checkpoint train default is often 100).",
    )
    parser.add_argument(
        "--replan-hz",
        type=float,
        default=2.0,
        help="Target diffusion replan rate. Chunk length = 2*control_hz/replan_hz.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fraction of the chunk to execute before starting the next async replan.",
    )
    parser.add_argument(
        "--round-gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Round the policy's gripper command to 0/1 at 0.5 (demos record exactly 0 or 1).",
    )

    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help="Hand control back to teleop after this many policy steps (0 = no limit).",
    )


def resolve_device(device: str) -> torch.device:
    """Fall back to CPU when CUDA is requested but unavailable."""
    return torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")


def load_diffusion_bundle(args: argparse.Namespace) -> DiffusionBundle:
    """Load the policy, processors, dataset features, and recording timing from a checkpoint."""
    device = resolve_device(args.device)
    policy, preprocessor, postprocessor, pretrained_dir = load_diffusion_policy(
        args.checkpoint,
        device=str(device),
        inference_steps=args.inference_steps,
    )
    ds_features = load_play_features(pretrained_dir=pretrained_dir, policy=policy, dataset=args.dataset)
    task_name, action_spec, dt = playback_settings(
        load_dataset_info(pretrained_dir=pretrained_dir, dataset=args.dataset)
    )
    return DiffusionBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        ds_features=ds_features,
        image_keys=policy_image_keys(policy),
        task_name=task_name,
        action_spec=action_spec,
        dt=dt,
        pretrained_dir=pretrained_dir,
        device=device,
    )


def require_twist_tcp(bundle: DiffusionBundle) -> None:
    """The teleop adapter only sends TCP twists; reject checkpoints trained on other action specs."""
    if bundle.action_spec != "TWIST_TCP":
        raise ValueError(
            f"Checkpoint was trained with action_spec={bundle.action_spec}; "
            "the diffusion teleop adapter only supports TWIST_TCP policies."
        )


def policy_camera_flags(image_keys: list[str]) -> tuple[bool, bool]:
    """Return ``(use_wrist, use_tabletop)`` from the policy's image keys."""
    use_wrist = any(k.endswith(".wrist") for k in image_keys)
    use_tabletop = any(k.endswith(".tabletop") for k in image_keys)
    return use_wrist, use_tabletop


def _camera_hw(ds_features: dict[str, Any], name: str) -> tuple[int, int] | None:
    feature = ds_features.get(f"{_IMAGE_PREFIX}{name}")
    if feature is None:
        return None
    shape = list(feature["shape"])
    names = list(feature.get("names") or [])
    if names[:2] == ["channels", "height"] or (len(shape) == 3 and shape[0] == 3 and shape[-1] != 3):
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def camera_size_overrides(ds_features: dict[str, Any], robot_kind: str) -> dict[str, int]:
    """Robot-config camera sizes so ``get_observation()`` resizes to the trained image shapes.

    ``robot_kind`` is ``"mj"`` or ``"kortex"``. The Kortex config has separate wrist fields
    (tabletop uses ``camera_*``); the MuJoCo config only has a shared ``camera_*`` size.
    """
    wrist = _camera_hw(ds_features, "wrist")
    tabletop = _camera_hw(ds_features, "tabletop")
    overrides: dict[str, int] = {}

    if robot_kind == "kortex":
        if wrist is not None:
            overrides["wrist_camera_height"], overrides["wrist_camera_width"] = wrist
        if tabletop is not None:
            overrides["camera_height"], overrides["camera_width"] = tabletop
        return overrides

    if robot_kind == "mj":
        sizes = {hw for hw in (wrist, tabletop) if hw is not None}
        if len(sizes) > 1:
            raise ValueError(
                f"Policy cameras have different sizes (wrist={wrist}, tabletop={tabletop}), "
                "but MjGen3RobotConfig only supports one shared camera size."
            )
        if sizes:
            overrides["camera_height"], overrides["camera_width"] = sizes.pop()
        return overrides

    raise ValueError(f"Unknown robot_kind={robot_kind!r}; expected 'mj' or 'kortex'.")


def build_realtime_controller(
    bundle: DiffusionBundle,
    args: argparse.Namespace,
    control_hz: float,
) -> RealtimeDiffusionController:
    """Construct the overlapping-chunk diffusion controller from the shared CLI arguments."""
    return RealtimeDiffusionController(
        bundle.policy,
        control_hz=control_hz,
        replan_hz=args.replan_hz,
        overlap=args.overlap,
        postprocessor=bundle.postprocessor,
        round_gripper=args.round_gripper,
    )


def cube_goal_done_fn(success_delta: float) -> Callable[[Any], bool] | None:
    """Return a ``done_fn(env)`` that fires when the cube reaches its goal, or ``None`` if disabled."""
    if success_delta <= 0:
        return None

    from kinova_lfd.data_collection.policy import cube_goal_success

    def _done(env: Any) -> bool:
        cube_pose = getattr(env, "last_cube_pose_w", getattr(env, "cube_pose_w", None))
        goal_pose = getattr(env, "last_cube_goal_pose_w", getattr(env, "cube_goal_pose_w", None))
        if cube_pose is None or goal_pose is None:
            return False
        return bool(cube_goal_success(cube_pose, goal_pose, success_delta))

    return _done


def wait_for_camera(robot: Robot, camera: str = "wrist", timeout: float = 10.0) -> None:
    """Block until ``robot.get_observation()`` contains ``camera``."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if camera in robot.get_observation():
            return
        time.sleep(0.05)
    raise RuntimeError(f"No {camera} camera frame within {timeout:.1f}s; check the camera stream.")
