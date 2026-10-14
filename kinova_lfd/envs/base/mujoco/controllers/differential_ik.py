import torch
from mjlab.envs.mdp.actions import DifferentialIKActionCfg

from skillet.core.math import quat_from_euler_xyz
from skillet.envs.mujoco import MjDirectRlEnv


class MjDifferentialIk:
    def __init__(self, env: MjDirectRlEnv):
        self._env = env
        self._ik_action_cfg = DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=("joint.*",),
            frame_name="pinch_site",
            frame_type="site",
            posture_weight=0.05,
            joint_limit_weight=0.1,
            damping=0.01,
            use_relative_mode=False,
            max_dq=0.2,
            orientation_weight=100.0,
        )
        self._ik_action = self._ik_action_cfg.build(self._env)
        self._q_dot_prev = None

    def compute_joint_vel(self, actions: torch.Tensor, smoothing: bool = True) -> torch.Tensor:
        """Compute the new joint velocity with damped least squares."""
        quat_actions = quat_from_euler_xyz(actions[:, 3], actions[:, 4], actions[:, 5])
        self._ik_action.process_actions(torch.cat((actions[:, 0:3], quat_actions), dim=1))
        dq = self._ik_action.compute_dq()
        return self._velocity_smoothing(dq) if smoothing else dq

    def _velocity_smoothing(self, q_dot: torch.Tensor) -> torch.Tensor:
        """Smooth joint velocities computed from DiffIK.

        Args:
            q_dot: A tensor of shape (N, num_joints) for delta joint positions.

        Returns:
            A tensor of shape (N, num_joints) of smoothed joint positions.

        """
        self.max_delta = 0.1
        self.max_vel = 0.25
        self.alpha = 0.3

        # Normalize joints
        max_comp = torch.max(torch.abs(q_dot))
        if max_comp > self.max_delta:
            q_dot = q_dot * (self.max_delta / max_comp)

        q_dot = torch.clip(q_dot, -self.max_vel, self.max_vel)

        # Low pass filtering
        if self._q_dot_prev is None:
            self._q_dot_prev = q_dot
        q_dot = self.alpha * q_dot + (1 - self.alpha) * self._q_dot_prev
        self._q_dot_prev = q_dot

        return self._q_dot_prev
