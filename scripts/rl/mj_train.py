"""mj_train.py.

Script for training agents in Mujoco Warp.

Usage:
    python scripts/rl/mj_train.py Mj-Gen3-LiftCube-v0 --env.scene.num-envs 4096 --agent.max-iterations 2000

Written by Will Solow and Jeff Jewett, 2026
"""

import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml, get_checkpoint_path

from kinova_lfd.envs.factory import create_env
from kinova_lfd.rl.cli import EnvCfgT, RunConfig, parse_cli
from kinova_lfd.rl.vecenv import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig(RunConfig[EnvCfgT]):
    """Train an RL agent with RSL-RL."""


def main() -> None:
    """Train with RSL-RL agent."""
    task, cfg = parse_cli(TrainConfig, description="Train an RL agent with RSL-RL.")
    env_cfg, agent_cfg = cfg.env, cfg.agent

    # A seed of -1 samples a random seed; the env always uses the agent seed.
    if agent_cfg.seed == -1:
        agent_cfg.seed = random.randint(0, 10000)
    env_cfg.seed = agent_cfg.seed

    # Specify directory for logging experiments
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Exact experiment name requested from command line: {run_dir}")
    if agent_cfg.run_name:
        run_dir += f"_{agent_cfg.run_name}"
    log_dir = log_root_path / run_dir
    env_cfg.log_dir = str(log_dir)

    # Resolve the resume path before the new run directory exists (it would otherwise be picked as latest).
    resume_path = (
        get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint) if agent_cfg.resume else None
    )

    # Dump the configuration into log-directory
    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

    env = RslRlVecEnvWrapper(create_env(task, env_cfg), clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=env_cfg.device)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
