# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

import importlib
import inspect
import os
import pathlib
import re
from collections import defaultdict

import gymnasium as gym
import yaml

from kinova_lfd.envs.base.mujoco.configclass import class_to_dict


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports both YAML and Python configuration files.

    It expects the configuration to be registered in the gym registry as:

    .. code-block:: python

        gym.register(
            id="My-Awesome-Task-v0",
            ...
            kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
        )


    The parsed configuration object for above example can be obtained as:

    .. code-block:: python

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")


    Args:
        task_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
        If the entry point is a Python class, it is instantiated and returned.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.

    """
    task_name = task_name.split(":")[-1]
    try:
        task_spec = gym.spec(task_name)
    except gym.error.Error as exc:
        # Local tasks intentionally do not need Gym registration.
        if entry_point_key == "env_cfg_entry_point":
            from kinova_lfd.envs.factory import get_env_cfg

            return get_env_cfg(task_name)
        if entry_point_key == "rsl_rl_cfg_entry_point":
            from kinova_lfd.rl.cfg.default_rsl_rl import make_default_rsl_rl_cfg

            return make_default_rsl_rl_cfg()
        raise ValueError(
            f"Task '{task_name}' is not Gym-registered and has no local configuration for '{entry_point_key}'."
        ) from exc

    # obtain the configuration entry point
    cfg_entry_point = task_spec.kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        # get existing agents and algorithms
        agents = collections.defaultdict(list)
        for k in task_spec.kwargs:
            if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
                spec = (
                    k.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
        # raise error
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if pathlib.Path(cfg_entry_point).exists():
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if callable(cfg_cls) else cfg_cls
    return cfg


def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Returns:
        The path to the model checkpoint.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        run_path = os.path.join(runs[-1], *other_dirs) if other_dirs is not None else runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Save data into a YAML file safely.

    The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.

    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not pathlib.Path(os.path.dirname(filename)).exists():
        pathlib.Path(os.path.dirname(filename)).mkdir(exist_ok=True, parents=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with pathlib.Path(filename).open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)
