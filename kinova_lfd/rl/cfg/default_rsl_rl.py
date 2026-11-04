"""Default RSL-RL training configuration shared by local tasks."""

from kinova_lfd.rl.cfg.rsl_rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def make_default_rsl_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create a complete default PPO runner configuration."""
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        device="cuda:0",
        num_steps_per_env=24,
        max_iterations=1500,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        clip_actions=None,
        save_interval=100,
        experiment_name="mj_gen3",
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        ),
    )
