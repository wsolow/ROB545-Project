# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""

import functools
import importlib
from collections.abc import Callable
from typing import Any

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from kinova_lfd.envs.factory import get_env_cfg
from kinova_lfd.rl.cfg.default_rsl_rl import make_default_rsl_rl_cfg

from .dict import replace_slices_with_strings, replace_strings_with_slices
from .spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str | None, output_dir: str = "./_outputs"
) -> tuple[Any, dict]:
    """Register local task configurations to the Hydra configuration store.

    Task discovery is deliberately local: Gymnasium registration is not required.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        A tuple containing the parsed environment and agent configuration objects.

    """
    env_cfg = get_env_cfg(task_name)
    agent_cfg = _load_agent_cfg(agent_cfg_entry_point)
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    agent_cfg_dict = agent_cfg if isinstance(agent_cfg, dict) or agent_cfg is None else agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict, "hydra": {"run": {"dir": output_dir}}}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    # store the configuration to Hydra
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def _load_agent_cfg(agent_cfg_entry_point: str | None) -> Any:
    """Load an explicit config path or return the package-wide default."""
    if agent_cfg_entry_point in (None, "", "default", "rsl_rl_cfg_entry_point"):
        return make_default_rsl_rl_cfg()
    if ":" not in agent_cfg_entry_point:
        raise ValueError(
            "Explicit agent configuration must use the format 'module:attribute' "
            f"or one of the default sentinels. Received: {agent_cfg_entry_point}"
        )
    module_name, attribute_name = agent_cfg_entry_point.split(":", maxsplit=1)
    cfg = getattr(importlib.import_module(module_name), attribute_name)
    return cfg() if callable(cfg) else cfg


def hydra_task_config(task_name: str, agent_cfg_entry_point: str | None) -> Callable:
    """Handle with a Decorator the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            # define the new Hydra main function
            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator
