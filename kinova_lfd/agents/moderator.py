from enum import IntEnum

import numpy as np

from kinova_lfd.agents.data_collector import DataLogger
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.teleop.devices import DeviceBase

HOME_TCP_CART = np.asarray([0.25, 0.0, 0.3, np.pi, 0.0, np.pi / 2, 0.0])


class ActionStatusCodes(IntEnum):
    """The codes for the status of robot actions."""

    QUIT = 0
    """The experiment should exit."""
    RUNNING = 1
    """The experiment is running."""
    STOP = 2
    """The experiment is stopped."""
    HOME = 3
    """The experiment is resetting to safe position."""
    RESUME = 4
    """The experiment is resuming."""
    ADMITTANCE = 5
    """If to toggle admittance mode"""


class LfDModerator:
    def __init__(self, env: GymEnv, teleop_device: DeviceBase, logger: DataLogger):
        self._action_status = ActionStatusCodes.RUNNING

        self._env = env
        self._teleop_device = teleop_device
        self._logger = logger

        self._teleop_device.start()
        self._logger.start(
            self._env,
            {
                "tool_pose_b": "_robot_tool_pose_b",
                "wrist_rgb": "_wrist_rgb",
                "joint_positions": "_joint_positions",
                "joint_velocities": "_joint_velocities",
                "actions": "_prev_actions",
            },
        )

    def poll(self, action: np.ndarray) -> tuple[np.ndarray, ActionSpec]:
        """Poll the correct action on a skill."""
        self._state_update()

        action_spec = (
            ActionSpec.CARTESIAN_ADMITTANCE
            if self._action_status == ActionStatusCodes.ADMITTANCE
            else ActionSpec.TWIST_TCP
        )
        if self._action_status == ActionStatusCodes.QUIT:
            action_spec = ActionSpec.TWIST_TCP
            action = np.asarray([0, 0, 0, 0, 0, 0, 0])
        elif self._action_status == ActionStatusCodes.HOME:
            action_spec = ActionSpec.MOVE_TCP
            action = HOME_TCP_CART
        elif self._action_status == ActionStatusCodes.STOP:
            action_spec = ActionSpec.TWIST_TCP
            action = np.asarray([0, 0, 0, 0, 0, 0, 0])
        elif self._action_status == ActionStatusCodes.RESUME:
            pass
        elif self._action_status == ActionStatusCodes.ADMITTANCE:
            action_spec = ActionSpec.CARTESIAN_ADMITTANCE
        return action, action_spec

    def _state_update(self) -> None:

        def _quit():
            if self._teleop_device.quit:
                print("[INFO][MODERATOR] Quitting.")
                self._action_status = ActionStatusCodes.QUIT
                return True
            return False

        def _resume():
            if self._teleop_device.resume:
                print("[INFO][MODERATOR] Resuming the experiment.")
                self._action_status = ActionStatusCodes.RUNNING
                return True
            return False

        def _home():
            if self._teleop_device.home:
                print("[INFO][MODERATOR] Returning to home position.")
                self._action_status = ActionStatusCodes.HOME
                return True
            return False

        def _stop():
            if self._teleop_device.stop_exp:
                print("[INFO][MODERATOR] Stopping the robot.")
                self._action_status = ActionStatusCodes.STOP
                return True
            return False

        def _admittance_run():
            if self._teleop_device.admittance and self._action_status == ActionStatusCodes.RUNNING:
                print("[INFO][MODERATOR] Changing to Admittance Mode")
                self._action_status = ActionStatusCodes.ADMITTANCE
                return True
            if not self._teleop_device.admittance and self._action_status == ActionStatusCodes.ADMITTANCE:
                print("[INFO][MODERATOR] Changing to Twist Mode")
                self._action_status = ActionStatusCodes.RUNNING
                return True
            return False

        if self._action_status == ActionStatusCodes.QUIT:
            pass
        if self._action_status == ActionStatusCodes.HOME:
            if _quit():
                return
            if _resume():
                return
            if _stop():
                return
        if self._action_status in [ActionStatusCodes.RUNNING, ActionStatusCodes.ADMITTANCE]:
            if _quit():
                return
            if _stop():
                return
            if _home():
                return
            if _admittance_run():
                return
        if self._action_status == ActionStatusCodes.STOP:
            if _quit():
                return
            if _home():
                return
            if _resume():
                return

        self._logger.log = self._teleop_device.log
        self._logger.save_log = self._teleop_device.save

    def run_teleop_loop(self) -> None:
        """Run the teleop in the environment."""
        terminated = False
        import time

        while not terminated:
            actions = self._teleop_device.advance(self._env._robot_tool_pose_b)

            action, action_spec = self.poll(actions)
            if self._action_status == ActionStatusCodes.QUIT:
                terminated = True
                self._teleop_device.stop()
                self._env.close()
                break
            self._env.step(action, action_spec=action_spec)
