"""Default RSL-RL agent configuration, built on ``mjlab.rl`` config dataclasses."""

import importlib

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def make_default_rsl_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create the default PPO runner configuration shared by local tasks."""
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=24,
        max_iterations=2500,
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
        clip_actions=None,
        save_interval=50,
        experiment_name="mj_gen3",
        logger="tensorboard",
        upload_model=False,
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=8,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.001,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
    )


def load_agent_cfg(agent_cfg_entry_point: str | None = None) -> RslRlOnPolicyRunnerCfg:
    """Load an agent config from ``"module:attribute"`` (a config or a factory), or return the default."""
    if agent_cfg_entry_point in (None, "", "default", "rsl_rl_cfg_entry_point"):
        return make_default_rsl_rl_cfg()
    if ":" not in agent_cfg_entry_point:
        raise ValueError(
            f"Agent configuration must use the format 'module:attribute' or 'default'. Received: {agent_cfg_entry_point}"
        )
    module_name, attribute_name = agent_cfg_entry_point.split(":", maxsplit=1)
    cfg = getattr(importlib.import_module(module_name), attribute_name)
    return cfg() if callable(cfg) else cfg
