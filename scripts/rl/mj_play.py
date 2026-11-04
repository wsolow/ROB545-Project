"""mj_play.py.

Script for visualizing a trained policy with Mujoco

Written by Will Solow and Jeff Jewett, 2026
"""

import argparse
import os
import sys

import torch

from kinova_lfd.envs.base.mujoco.viewer import MjViewer
from kinova_lfd.envs.factory import create_env
from kinova_lfd.rl.cfg import cli_args
from kinova_lfd.rl.cfg.rsl_rl_cfg import RslRlBaseRunnerCfg
from kinova_lfd.rl.rsl_rl.runners import OnPolicyRunner
from kinova_lfd.rl.rsl_wrapper import RslRlVecEnvWrapper
from kinova_lfd.rl.util.export import export_policy_as_jit
from kinova_lfd.rl.util.hydra import hydra_task_config
from kinova_lfd.rl.util.parse_cfg import get_checkpoint_path

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--device", type=str, default="cuda", choices={"cuda", "cpu"}, help="Device to run on: cuda/cpu")
parser.add_argument("--viewer", type=str, default="auto", help="Mujoco viewer backend to use/")
parser.add_argument(
    "--enable_cameras",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If to enable cameras in the environment.",
)
cli_args.add_rsl_rl_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    agent_cfg.device = args_cli.device

    # Apply environment overrides to the local config before direct construction.
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Set the environment seed
    env_cfg.seed = agent_cfg.seed
    # env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.enable_cameras = args_cli.enable_cameras

    # Specify directory for logging experiments
    log_root_path = os.path.join("_logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    env = create_env(args_cli.task, env_cfg)

    # Wrap around environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if hasattr(runner.alg.policy, "actor_obs_normalizer"):
        normalizer = runner.alg.policy.actor_obs_normalizer
    elif hasattr(runner.alg.policy, "student_obs_normalizer"):
        normalizer = runner.alg.policy.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, normalizer=normalizer, path=export_model_dir, filename="agent.pt")
    # Launch Mujoco viewer
    if args_cli.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
        del has_display
    else:
        resolved_viewer = args_cli.viewer

    if resolved_viewer == "native":
        MjViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
