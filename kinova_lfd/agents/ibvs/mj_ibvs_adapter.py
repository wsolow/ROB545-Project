"""mj_ibvs_adapter.py.

MjViewer policy that switches between keyboard teleop and an IBVS grasp in MuJoCo.

Same keys and mode logic as :class:`KortexIbvsTeleopAdapter`; only the MuJoCo-specific
parts are overridden:

- ``env._robot_tool_pose_b`` is a ``(num_envs, 7)`` torch tensor whose quaternion is
  already wxyz (mjlab / MuJoCo convention), so it is passed to IBVSController as-is.
  Do **not** run it through ``np_convert_quat(..., to="wxyz")``: that treats wxyz as
  xyzw, scrambles the orientation, and makes the DESCEND/LIFT PID (world -> TCP twist)
  move sideways once the tool has any yaw.
- Home defaults to the TCP pose at construction (call after ``env.reset()``), since
  ``HOME_TCP_CART`` is expressed in the real robot's base frame.

Written by Will Solow, 2026.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from kinova_lfd.agents.ibvs import IbvsTeleopAdapter
from kinova_lfd.agents.ibvs.util import rgbd_to_numpy
from kinova_lfd.agents.moderator import HOME_TCP_CART
from kinova_lfd.controllers.ibvs import GraspPhase, IBVSController
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.teleop.devices import Se3Keyboard


class MjIbvsPolicyAdapter(IbvsTeleopAdapter):
    """Keyboard teleop + on-demand IBVS grasp for MuJoCo Gen3 envs (see module docstring)."""

    def __init__(
        self,
        env,
        teleop: Se3Keyboard,
        controller_factory: Callable[[np.ndarray, np.ndarray], IBVSController],
        controller: IBVSController | None = None,
        setpoint_dir: str | Path = "exp/ibvs_setpoints/mj",
        home_tcp: np.ndarray | None = HOME_TCP_CART,
    ) -> None:
        """Initialize the adapter.

        Args:
            env: MuJoCo Gen3 environment (already reset).
            teleop: Keyboard device used for teleoperation.
            controller_factory: ``(setpoint_rgb, K) -> IBVSController``, used when a new
                setpoint is captured with B.
            controller: Initial controller, or ``None`` until a setpoint is captured.
            setpoint_dir: Where captured setpoints are written.

        """
        super().__init__(
            env,
            teleop,
            controller_factory=controller_factory,
            controller=controller,
            setpoint_dir=setpoint_dir,
            home_tcp=home_tcp,
        )

    def _teleop_action(self) -> tuple[np.ndarray, ActionSpec]:
        action = self.teleop.advance(self.env._robot_tool_pose_b)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float64).reshape(-1), ActionSpec.TWIST_TCP

    def _ibvs_action(self) -> tuple[np.ndarray, ActionSpec]:
        assert self.controller is not None
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.controller.gripper_cmd]), ActionSpec.TWIST_TCP

        rgb, depth, _ = rgbd_to_numpy(rgbd)
        pose = self.env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        tcp_pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:7].copy()
        twist = self.controller(rgb, depth, tcp_pose=tcp_pose, dt=self.env.step_dt)
        twist = np.asarray(twist, dtype=np.float64)

        if self.controller.phase is GraspPhase.DONE:
            self._return_to_teleop("Grasp complete")

        return twist, ActionSpec.TWIST_TCP
