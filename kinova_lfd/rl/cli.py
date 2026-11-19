"""tyro command-line parsing shared by the RL scripts (mjlab-style).

Usage: ``python script.py <TASK> [--agent module:attr] [--env.<field> VALUE] [--agent.<field> VALUE] ...``
"""

import dataclasses
import sys
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import mjlab
import tyro
from mjlab.rl import RslRlOnPolicyRunnerCfg

from kinova_lfd.envs.base.mujoco.mj_env import MjDirectRlEnvCfg
from kinova_lfd.envs.factory import MJ_TASKS, get_env_cfg
from kinova_lfd.rl.config import load_agent_cfg

EnvCfgT = TypeVar("EnvCfgT", bound=MjDirectRlEnvCfg)
RunCfgT = TypeVar("RunCfgT", bound="RunConfig")


@dataclass
class RunConfig(Generic[EnvCfgT]):
    """Fields shared by every RL script. Scripts subclass this to add their own options."""

    env: EnvCfgT
    """Environment configuration (defaults come from the chosen task)."""
    agent: RslRlOnPolicyRunnerCfg
    """RSL-RL runner configuration."""
    log_root: str = "_logs/rsl_rl"
    """Root directory under which ``<experiment_name>/<run>`` folders are written."""


@dataclass
class _Selection:
    task: tyro.conf.Positional[Literal[MJ_TASKS]]  # type: ignore[valid-type]
    """Task to run."""
    agent_entry_point: str = "default"
    """Agent config as ``module:attribute`` (a config object or a factory), or ``default``."""


def parse_cli(
    config_cls: type[RunCfgT], description: str, env_overrides: dict[str, Any] | None = None
) -> tuple[str, RunCfgT]:
    """Parse ``<TASK>`` first, then parse the full ``config_cls`` CLI using that task's configs as defaults.

    Args:
        config_cls: A :class:`RunConfig` subclass describing the script's arguments.
        description: Help text for the script.
        env_overrides: Script-specific defaults applied to the env config before CLI overrides.

    Returns:
        The task name and the populated config.

    """
    # Without a task, show the task-selection help/error; with one, defer --help to the full config parser.
    has_task = any(arg in MJ_TASKS for arg in sys.argv[1:])
    selection, remaining_args = tyro.cli(
        _Selection, add_help=not has_task, return_unknown_args=True, description=description, config=mjlab.TYRO_FLAGS
    )
    env_cfg = get_env_cfg(selection.task)
    if env_overrides:
        env_cfg = dataclasses.replace(env_cfg, **env_overrides)
    default = config_cls(env=env_cfg, agent=load_agent_cfg(selection.agent_entry_point))
    cfg = tyro.cli(
        config_cls[type(env_cfg)],  # type: ignore[index]
        args=remaining_args,
        default=default,
        prog=f"{sys.argv[0]} {selection.task}",
        description=description,
        config=mjlab.TYRO_FLAGS,
    )
    return selection.task, cfg
