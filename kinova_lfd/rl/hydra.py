"""Hydra configuration utilities for local tasks.

Adapted from Isaac Lab (BSD-3-Clause, Copyright (c) 2022-2026, The Isaac Lab Project Developers).
"""

import functools
import importlib
import json
import re
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from kinova_lfd.envs.base.mujoco.configclass import class_to_dict, update_class_from_dict
from kinova_lfd.envs.factory import get_env_cfg
from kinova_lfd.rl.config import make_default_rsl_rl_cfg

_SPACE_ATTRS = ("observation_space", "action_space", "state_space")
_SPACE_DICT_ATTRS = ("observation_spaces", "action_spaces")


def hydra_task_config(task_name: str, agent_cfg_entry_point: str | None) -> Callable:
    """Decorate a ``main(env_cfg, agent_cfg)`` function so both configs can be overridden from the Hydra CLI.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: ``"module:attribute"`` of the agent config, or a default sentinel.

    """
    task_name = task_name.split(":")[-1]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> None:
            env_cfg, agent_cfg = _register_task_to_hydra(task_name, agent_cfg_entry_point)

            @hydra.main(config_path=None, config_name=task_name, version_base="1.3")
            def hydra_main(hydra_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg) -> None:
                cfg = _replace_strings_with_slices(OmegaConf.to_container(hydra_cfg, resolve=True))
                env_cfg.from_dict(cfg["env"])
                env_cfg = _deserialize_env_cfg_spaces(env_cfg)
                if isinstance(agent_cfg, dict):
                    agent_cfg = cfg["agent"]
                else:
                    update_class_from_dict(agent_cfg, cfg["agent"])
                func(env_cfg, agent_cfg, *args, **kwargs)

            hydra_main()

        return wrapper

    return decorator


def load_agent_cfg(agent_cfg_entry_point: str | None) -> Any:
    """Load an explicit ``module:attribute`` agent config, or return the package-wide default."""
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


def _register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str | None, output_dir: str = "./_outputs"
) -> tuple[Any, Any]:
    """Store the task's env/agent configs in the Hydra config store and return the config objects."""
    env_cfg = _serialize_env_cfg_spaces(get_env_cfg(task_name))
    agent_cfg = load_agent_cfg(agent_cfg_entry_point)
    agent_cfg_dict = agent_cfg if isinstance(agent_cfg, dict) else class_to_dict(agent_cfg)
    cfg_dict = {"env": env_cfg.to_dict(), "agent": agent_cfg_dict, "hydra": {"run": {"dir": output_dir}}}
    ConfigStore.instance().store(name=task_name, node=_replace_slices_with_strings(cfg_dict))
    return env_cfg, agent_cfg


# ==================== OmegaConf compatibility (no slice / gym.Space support) ====================


def _replace_slices_with_strings(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _replace_slices_with_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_slices_with_strings(v) for v in data]
    if isinstance(data, slice):
        return f"slice({data.start},{data.stop},{data.step})"
    return data


def _replace_strings_with_slices(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _replace_strings_with_slices(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_strings_with_slices(v) for v in data]
    if isinstance(data, str) and (match := re.fullmatch(r"slice\((.*),(.*),(.*)\)", data)):
        return slice(*(None if s == "None" else int(s) for s in match.groups()))
    return data


def _serialize_env_cfg_spaces(env_cfg: object) -> object:
    for attr in _SPACE_ATTRS:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, _serialize_space(getattr(env_cfg, attr)))
    for attr in _SPACE_DICT_ATTRS:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: _serialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg


def _deserialize_env_cfg_spaces(env_cfg: object) -> object:
    for attr in _SPACE_ATTRS:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, _deserialize_space(getattr(env_cfg, attr)))
    for attr in _SPACE_DICT_ATTRS:
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, {k: _deserialize_space(v) for k, v in getattr(env_cfg, attr).items()})
    return env_cfg


def _serialize_space(space: Any) -> str:  # noqa: PLR0911
    """Serialize a gymnasium space or python space specification as JSON."""
    if isinstance(space, gym.spaces.Discrete):
        return json.dumps({"type": "gymnasium", "space": "Discrete", "n": int(space.n)})
    if isinstance(space, gym.spaces.Box):
        return json.dumps(
            {"type": "gymnasium", "space": "Box", "low": space.low.tolist(), "high": space.high.tolist(),
             "shape": space.shape}
        )  # fmt: skip
    if isinstance(space, gym.spaces.MultiDiscrete):
        return json.dumps({"type": "gymnasium", "space": "MultiDiscrete", "nvec": space.nvec.tolist()})
    if isinstance(space, gym.spaces.Tuple):
        return json.dumps({"type": "gymnasium", "space": "Tuple", "spaces": [_serialize_space(s) for s in space]})
    if isinstance(space, gym.spaces.Dict):
        spaces = {k: _serialize_space(v) for k, v in space.spaces.items()}
        return json.dumps({"type": "gymnasium", "space": "Dict", "spaces": spaces})
    if isinstance(space, int) or (isinstance(space, list) and all(isinstance(x, int) for x in space)):
        return json.dumps({"type": "python", "space": "Box", "value": space})
    if isinstance(space, set) and len(space) == 1:
        return json.dumps({"type": "python", "space": "Discrete", "value": next(iter(space))})
    if isinstance(space, list) and all(isinstance(x, set) and len(x) == 1 for x in space):
        return json.dumps({"type": "python", "space": "MultiDiscrete", "value": [next(iter(x)) for x in space]})
    if isinstance(space, tuple):
        return json.dumps({"type": "python", "space": "Tuple", "value": [_serialize_space(x) for x in space]})
    if isinstance(space, dict):
        return json.dumps(
            {"type": "python", "space": "Dict", "value": {k: _serialize_space(v) for k, v in space.items()}}
        )
    raise ValueError(f"Unsupported space ({space})")


def _deserialize_space(string: str) -> Any:  # noqa: PLR0911
    """Inverse of :func:`_serialize_space`."""
    obj = json.loads(string)
    kind, space = obj["type"], obj["space"]
    if kind == "gymnasium":
        if space == "Discrete":
            return gym.spaces.Discrete(n=obj["n"])
        if space == "Box":
            return gym.spaces.Box(low=np.array(obj["low"]), high=np.array(obj["high"]), shape=obj["shape"])
        if space == "MultiDiscrete":
            return gym.spaces.MultiDiscrete(nvec=np.array(obj["nvec"]))
        if space == "Tuple":
            return gym.spaces.Tuple(spaces=tuple(map(_deserialize_space, obj["spaces"])))
        if space == "Dict":
            return gym.spaces.Dict(spaces={k: _deserialize_space(v) for k, v in obj["spaces"].items()})
    elif kind == "python":
        value = obj["value"]
        if space == "Discrete":
            return {value}
        if space == "Box":
            return value
        if space == "MultiDiscrete":
            return [{x} for x in value]
        if space == "Tuple":
            return tuple(map(_deserialize_space, value))
        if space == "Dict":
            return {k: _deserialize_space(v) for k, v in value.items()}
    raise ValueError(f"Unsupported space ({obj})")
