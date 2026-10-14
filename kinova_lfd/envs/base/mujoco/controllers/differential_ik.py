import torch
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.utils.lab_api.math import apply_delta_pose

from kinova_lfd.core.tensor_math import quat_apply, quat_from_euler_xyz
from kinova_lfd.envs.base.gym import GymEnv


class MjDifferentialIk:
    def __init__(self, env: GymEnv):
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
        self._goal_pos = None
        self._goal_quat = None

    def reset(self) -> None:
        """Drop the integrated tool goal and the velocity filter."""
        self._q_dot_prev = None
        self._goal_pos = None
        self._goal_quat = None

    def compute_joint_vel(self, actions: torch.Tensor, smoothing: bool = True) -> torch.Tensor:
        """Compute the new joint velocity with damped least squares."""
        quat_actions = quat_from_euler_xyz(actions[:, 3], actions[:, 4], actions[:, 5])
        self._ik_action.process_actions(torch.cat((actions[:, 0:3], quat_actions), dim=1))
        dq = self._ik_action.compute_dq()
        return self._velocity_smoothing(dq) if smoothing else dq

    def compute_joint_pos(self, twist: torch.Tensor, smoothing: bool = True) -> torch.Tensor:
        """Resolve a tool-frame twist into a joint displacement.

        ``twist`` is expressed in the TCP frame, matching ``advance()``. While a
        command is held, it is rotated into the world frame and integrated into a
        tool goal. The same damped least-squares IK as :meth:`compute_joint_vel`
        tracks that goal. A zero twist returns zeros.

        Args:
            twist: TCP linear velocity (m/s) and angular velocity (rad/s).
                Shape is (N, 6).
            smoothing: Apply the same joint-step filter as :meth:`compute_joint_vel`.

        Returns:
            Joint displacement of shape (N, num_joints). Zero when the twist is zero.

        """
        # A zero twist is a hold. Drop the integrated goal so the next command
        # starts from the pose the arm is actually holding.
        if not torch.any(twist[:, :6].abs() > 1e-6):
            self._goal_pos = None
            self._goal_quat = None
            return torch.zeros(
                (twist.shape[0], self._ik_action._num_joints),
                device=twist.device,
                dtype=twist.dtype,
            )

        frame_pos, frame_quat = self._ik_action._get_frame_pose()
        if self._goal_pos is None:
            self._goal_pos = frame_pos.clone()
            self._goal_quat = frame_quat.clone()
        dt = self._env.dt
        # Keyboard and base-frame devices both publish the command in the TCP frame.
        lin_w = quat_apply(frame_quat, twist[:, 0:3])
        ang_w = quat_apply(frame_quat, twist[:, 3:6])
        delta = torch.cat((lin_w, ang_w), dim=1) * dt
        self._goal_pos, self._goal_quat = apply_delta_pose(self._goal_pos, self._goal_quat, delta)
        self._ik_action.process_actions(torch.cat((self._goal_pos, self._goal_quat), dim=1))
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
