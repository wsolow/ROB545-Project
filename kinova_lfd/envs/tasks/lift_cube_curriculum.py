"""Curriculum scheduling for the MuJoCo lift-cube task."""

from typing import Any

from kinova_lfd.envs.base.mujoco.configclass import configclass


@configclass
class LiftCubeCurriculumCfg:
    """Configuration for the lift-cube reward curriculum."""

    start_step: int = 0
    end_step: int = 100_000

    # Cube reward coefficients. All defaults remain below the original
    # coefficients in MjGen3LiftCubeEnvCfg.
    cube_reach_reward_scale_start: float = 0.5
    cube_reach_reward_scale_end: float = 0.75
    cube_lift_reward_scale_start: float = 5.0
    cube_lift_reward_scale_end: float = 10.0
    cube_goal_reward_scale_start: float = 5.0
    cube_goal_reward_scale_end: float = 10.0
    cube_goal_fine_grained_scale_start: float = 1.0
    cube_goal_fine_grained_scale_end: float = 3.0

    # These are negative penalties. Increasing their magnitude means moving
    # from the start value toward the more-negative end value.
    action_rate_reward_scale_start: float = -0.0001
    action_rate_reward_scale_end: float = -0.001
    joint_vel_reward_scale_start: float = -0.0001
    joint_vel_reward_scale_end: float = -0.001

    def __post_init__(self) -> None:
        """Validate the configured global-step interval."""
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")


class LiftCubeCurriculumManager:
    """Smoothly schedule lift-cube reward coefficients over global steps."""

    def __init__(self, cfg: LiftCubeCurriculumCfg) -> None:
        """Create a manager using the supplied curriculum configuration."""
        self.cfg = cfg

    def _progress(self, step: int) -> float:
        """Return smoothstep progress in the configured schedule interval."""
        linear_progress = (step - self.cfg.start_step) / (self.cfg.end_step - self.cfg.start_step)
        linear_progress = min(max(linear_progress, 0.0), 1.0)
        return linear_progress * linear_progress * (3.0 - 2.0 * linear_progress)

    def values(self, step: int) -> dict[str, float]:
        """Return the scheduled reward coefficients for one global step."""
        progress = self._progress(step)
        return {name: start + progress * (end - start) for name, start, end in self._scheduled_values()}

    def _scheduled_values(self) -> tuple[tuple[str, float, float], ...]:
        return (
            (
                "cube_reach_reward_scale",
                self.cfg.cube_reach_reward_scale_start,
                self.cfg.cube_reach_reward_scale_end,
            ),
            (
                "cube_lift_reward_scale",
                self.cfg.cube_lift_reward_scale_start,
                self.cfg.cube_lift_reward_scale_end,
            ),
            (
                "cube_goal_reward_scale",
                self.cfg.cube_goal_reward_scale_start,
                self.cfg.cube_goal_reward_scale_end,
            ),
            (
                "cube_goal_fine_grained_scale",
                self.cfg.cube_goal_fine_grained_scale_start,
                self.cfg.cube_goal_fine_grained_scale_end,
            ),
            (
                "action_rate_reward_scale",
                self.cfg.action_rate_reward_scale_start,
                self.cfg.action_rate_reward_scale_end,
            ),
            (
                "joint_vel_reward_scale",
                self.cfg.joint_vel_reward_scale_start,
                self.cfg.joint_vel_reward_scale_end,
            ),
        )

    def log_values(self, step: int) -> dict[str, Any]:
        """Return scheduled values and progress in a logging-friendly format."""
        return {"curriculum_step": step, "curriculum_progress": self._progress(step), **self.values(step)}
