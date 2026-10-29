import torch
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.utils.lab_api.math import apply_delta_pose

from kinova_lfd.core.tensor_math import quat_apply, quat_from_euler_xyz
from kinova_lfd.envs.base.gym import GymEnv


class MjDifferentialIk:
    def __init__(
        self,
        env: GymEnv,
        *,
        tau: float = 0.03,
        a_max_lin: float = 3.0,
        a_max_ang: float = 8.0,
        max_lead_pos: float = 0.03,
    ):
        self._env = env
        self._ik_action_cfg = DifferentialIKActionCfg(
            entity_name="robot",
            actuator_names=("joint.*",),
            frame_name="pinch_site",
            frame_type="site",
            posture_weight=0.05,
            joint_limit_weight=0.1,
            damping=0.05,
            use_relative_mode=False,
            max_dq=0.1,
            orientation_weight=10.0,
        )
        self._ik_action = self._ik_action_cfg.build(self._env)
        self._q_dot_prev = None
        self._goal_pos = None
        self._goal_quat = None
        self._twist_f = None
        self.active = False
        # Twist low-pass time constant (s) and accel limits (m/s^2, rad/s^2).
        self.tau = float(tau)
        self.a_max_lin = float(a_max_lin)
        self.a_max_ang = float(a_max_ang)
        self.a_max = None  # lazily built (6,) tensor on first filter call
        self.max_lead_pos = float(max_lead_pos)

    def reset(self) -> None:
        """Drop the integrated tool goal and the velocity filter."""
        self._goal_pos = self._goal_quat = self._twist_f = self._q_dot_prev = None
        self.active = False

    def _filter_twist(self, twist):
        dt = self._env.dt
        if self._twist_f is None:
            self._twist_f = torch.zeros_like(twist)
            self.a_max = torch.tensor(
                [self.a_max_lin] * 3 + [self.a_max_ang] * 3,
                device=twist.device,
                dtype=twist.dtype,
            )
        a = dt / (self.tau + dt)
        step = a * (twist - self._twist_f)
        step = torch.clamp(step, -self.a_max * dt, self.a_max * dt)
        self._twist_f = self._twist_f + step
        return self._twist_f

    def compute_joint_vel(self, actions: torch.Tensor, smoothing: bool = True) -> torch.Tensor:
        """Compute the new joint velocity with damped least squares."""
        quat_actions = quat_from_euler_xyz(actions[:, 3], actions[:, 4], actions[:, 5])
        self._ik_action.process_actions(torch.cat((actions[:, 0:3], quat_actions), dim=1))
        dq = self._solve_dq()
        return self._velocity_smoothing(dq) if smoothing else dq

    def compute_joint_pos(self, twist, smoothing=False):
        twist_f = self._filter_twist(twist[:, :6])
        self.active = bool((twist_f.abs() > 1e-4).any())
        if not self.active:
            self._twist_f.zero_()
            self._goal_pos = self._goal_quat = None
            return torch.zeros((twist.shape[0], self._ik_action._num_joints), device=twist.device, dtype=twist.dtype)

        frame_pos, frame_quat = self._ik_action._get_frame_pose()
        if self._goal_pos is None:
            self._goal_pos, self._goal_quat = frame_pos.clone(), frame_quat.clone()

        dt = self._env.dt
        # rotate by the GOAL orientation so the path is open-loop
        lin_w = quat_apply(self._goal_quat, twist_f[:, 0:3])
        ang_w = quat_apply(self._goal_quat, twist_f[:, 3:6])
        delta = torch.cat((lin_w, ang_w), dim=1) * dt
        self._goal_pos, self._goal_quat = apply_delta_pose(self._goal_pos, self._goal_quat, delta)

        # anti-windup: limit how far the goal can lead the measured pose
        err = self._goal_pos - frame_pos
        scale = (self.max_lead_pos / (err.norm(dim=1, keepdim=True) + 1e-9)).clamp(max=1.0)
        self._goal_pos = frame_pos + err * scale

        self._ik_action.process_actions(torch.cat((self._goal_pos, self._goal_quat), dim=1))
        return self._solve_dq()

    def _solve_dq(self) -> torch.Tensor:
        """Solve for a joint step, then scale the whole vector to ``max_dq``.

        mjlab clamps each joint on its own. Along world Y the shoulder and wrist
        have to move together; clipping only the larger one cancels the translation.
        """
        limit = self._ik_action.cfg.max_dq
        self._ik_action.cfg.max_dq = 1.0e6
        dq = self._ik_action.compute_dq()
        self._ik_action.cfg.max_dq = limit
        peak = dq.abs().amax(dim=-1, keepdim=True).clamp(min=1.0e-9)
        return dq * (limit / peak).clamp(max=1.0)

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
