"""mj_train.py.

Script for training agents in Mujoco Warp.

Written by Will Solow and Jeff Jewett, 2026
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.utils.os import dump_yaml, get_checkpoint_path

from kinova_lfd.envs.base.mujoco.configclass import class_to_dict
from kinova_lfd.envs.factory import create_env
from kinova_lfd.rl.config import add_rsl_rl_args, update_rsl_rl_cfg
from kinova_lfd.rl.hydra import hydra_task_config
from kinova_lfd.rl.vecenv import RslRlVecEnvWrapper

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--device", type=str, default="cuda", help="Name of GPU device")
parser.add_argument("--skill", action="store_true", help="If to use a a skill-based RL environment")

add_rsl_rl_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # Override configurations with non-hydra CLI arguments
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Apply environment overrides to the local config before direct construction.
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Set the environment seed
    env_cfg.seed = agent_cfg.seed

    # Specify directory for logging experiments
    log_root_path = (Path("_logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Exact experiment name requested from command line: {run_dir}")
    if agent_cfg.run_name:
        run_dir += f"_{agent_cfg.run_name}"
    log_dir = log_root_path / run_dir

    # Set the log directory for the environment
    env_cfg.log_dir = str(log_dir)

    env = create_env(args_cli.task, env_cfg)

    # Save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner from RSL-RL
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner = MjlabOnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=str(log_dir), device=args_cli.device)

    # Load checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    # Dump the configuration into log-directory
    dump_yaml(log_dir / "params" / "env.yaml", class_to_dict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", class_to_dict(agent_cfg))

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
