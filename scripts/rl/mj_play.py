"""mj_play.py.

Script for visualizing a trained policy with Mujoco

Usage:
    python scripts/rl/mj_play.py Mj-Gen3-LiftCube-v0 --agent.experiment-name mj_gen3
    python scripts/rl/mj_play.py Mj-Gen3-LiftCube-v0 --checkpoint-file _logs/rsl_rl/mj_gen3/<run>/model_100.pt

Written by Will Solow and Jeff Jewett, 2026
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from kinova_lfd.envs.base.viewers.mujoco_viewer import MjViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.rl.cli import EnvCfgT, RunConfig, parse_cli
from kinova_lfd.rl.vecenv import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@dataclass
class PlayConfig(RunConfig[EnvCfgT]):
    """Visualize a trained RSL-RL policy."""

    checkpoint_file: str | None = None
    """Explicit checkpoint file. If unset, ``agent.load_run`` / ``agent.load_checkpoint`` select one."""
    viewer: str = "auto"
    """Mujoco viewer backend (``auto`` picks ``native`` when a display is available)."""


def main() -> None:
    """Play with RSL-RL agent."""
    task, cfg = parse_cli(
        PlayConfig,
        description="Visualize a trained RSL-RL policy.",
        env_overrides={"enable_cameras": True},
    )
    env_cfg, agent_cfg = cfg.env, cfg.agent
    env_cfg.seed = agent_cfg.seed

    # Specify directory for logging experiments
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if cfg.checkpoint_file:
        resume_path = Path(cfg.checkpoint_file).resolve()
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = str(resume_path.parent)

    env = RslRlVecEnvWrapper(create_env(task, env_cfg), clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=None, device=env_cfg.device)
    runner.load(str(resume_path), map_location=env_cfg.device)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    runner.export_policy_to_jit(str(resume_path.parent / "exported"), filename="agent.pt")

    # Launch Mujoco viewer
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        MjViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
