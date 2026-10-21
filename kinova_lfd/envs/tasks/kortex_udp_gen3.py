"""gen3_kortex_udp.py.

Gen3 task on top of the 30 Hz UDP environment.

Written by Will Solow, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.kortex import KortexUDPEnv, KortexUDPEnvCfg


@dataclass
class Gen3KortexUdpEnvCfg(KortexUDPEnvCfg):
    """Kinova Gen3 7-DoF configuration for the UDP bridge."""

    joint_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])

    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.12, 0.0, 0.7071, -0.7071, 0.0])

    gripper_joint_names: list[str] = field(default_factory=lambda: ["robotiq_85_left_inner_knuckle_joint"])

    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
    )


class Gen3KortexUdpEnv(KortexUDPEnv):
    """Gen3 arm whose setpoints are joint positions or tool twists, plus the gripper."""

    def __init__(
        self,
        cfg: Gen3KortexUdpEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start the UDP bridge and allocate Gen3 observation buffers."""
        # Set up Kortex UDP bridge

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.joint_names = np.asarray(self.cfg.arm_joint_names + self.cfg.gripper_joint_names)
        self.arm_joint_names = np.asarray(self.cfg.arm_joint_names)
        self.gripper_joint_names = np.asarray(self.cfg.gripper_joint_names)

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            float("-inf"), float("inf"), shape=(2 * len(self.joint_names),)
        )
        self.single_action_space = gym.spaces.Box(float("-inf"), float("inf"), shape=(len(self.cfg.joint_ids),))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space)

        self._current_joint_positions = np.zeros(shape=len(self.joint_names))
        self._current_joint_velocities = np.zeros(shape=len(self.joint_names))
        self._current_joint_efforts = np.zeros(shape=len(self.joint_names))
        self._current_prev_actions = np.zeros(shape=(len(self.joint_names),))

        if self.cfg.use_tabletop_camera:
            from kinova_lfd.envs.base.kortex.realsense import RealsenseCamera

            print("[INFO][RealSense] Instancing TableTop Camera...")
            self._rs_tabletop_camera = RealsenseCamera()
            print("[INFO][RealSense] Finished Instancing TableTop Camera")
        if self.cfg.use_wrist_camera:
            from kinova_lfd.envs.base.kortex.realsense import WristCamera

            print("[INFO][Kortex] Instancing Wrist Camera...")
            self._rs_wrist_camera = WristCamera(self.cfg.robot_ip)
            self._rs_wrist_camera.start()
            print("[INFO][Kortex] Finished Instancing Wrist Camera")

    def _pre_process_action(self, actions: np.ndarray, action_spec: ActionSpec | None = None) -> np.ndarray:
        """Store the action and return it unchanged."""
        self.actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        self._current_prev_actions = self.actions.copy()
        return self.actions

    def _publish_action_to_kortex(
        self,
        actions: np.ndarray,
        action_spec: ActionSpec | None = None,
        duration: float = 1,
    ) -> None:
        """Forward a joint-position or twist setpoint to the C++ bridge.

        The bridge reads this packet on the next 1 ms ticks. Resolving it into
        actuator positions is left to ``JointTrajectoryController`` or ``TwistController``.

        Args:
            actions: Setpoint, with the gripper opening in the last element.
            action_spec: ``JOINTS_POS``, ``TWIST_TCP``, or ``None`` for joint positions.
            duration: Requested joint-trajectory time in seconds. The controller
                lengthens it when velocity or acceleration limits would be exceeded.

        """
        if actions.size < 1:
            raise ValueError("action must include a gripper opening")
        gripper = float(actions[-1])
        if action_spec in (None, ActionSpec.JOINTS_POS):
            if actions.size < 8:
                raise ValueError("joint position actions are 7 joints plus a gripper opening")
            self.kortex.send_joint_position(actions[:7], gripper, duration)
            return
        if action_spec == ActionSpec.TWIST_TCP:
            if actions.size < 7:
                raise ValueError("twist actions are 6 twist values plus a gripper opening")
            self.kortex.send_twist(actions[:6], gripper)
            return
        raise ValueError(f"KortexUDPEnv accepts JOINTS_POS and TWIST_TCP, received {action_spec!r}")

    def _get_dones(self) -> tuple[bool, bool]:
        """Truncate when the episode step limit is reached."""
        truncated = bool(self.episode_length_buf >= self.max_episode_length - 1)
        return False, truncated

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return joint position and velocity."""
        return {
            "policy": np.concatenate(
                (self._current_joint_positions, self._current_joint_velocities),
                axis=0,
                dtype=np.float32,
            )
        }

    def _get_rewards(self) -> np.ndarray:
        """Return a zero reward. Task rewards are filled in later."""
        return np.array([0.0])

    def _get_latest_rgbd(self) -> dict[str, Any]:
        """Return the latest tabletop RGB-D frame.

        Returns:
            RGB in ``(3, H, W)``, depth in meters, intrinsics, and a timestamp.

        """
        latest = self._rs_tabletop_camera._get_latest_rgbd_raw()
        latest["rgb"] = latest["rgb"].transpose((2, 0, 1))
        depth = np.expand_dims(latest["depth"], axis=0)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        latest["depth"] = depth
        return latest
