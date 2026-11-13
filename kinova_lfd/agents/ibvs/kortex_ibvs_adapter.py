from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from kinova_lfd.agents.ibvs.ibvs_adapter import IbvsTeleopAdapter
from kinova_lfd.agents.ibvs.util import rgbd_to_numpy
from kinova_lfd.agents.moderator import HOME_TCP_CART
from kinova_lfd.controllers import IBVSController
from kinova_lfd.controllers.ibvs import GraspPhase
from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.teleop.devices import Se3Keyboard


class KortexIbvsTeleopAdapter(IbvsTeleopAdapter):
    """KortexViewer policy that switches between keyboard teleop and an IBVS grasp.

    Key presses arrive on the pynput thread and are only latched there. All mode
    changes happen at the start of :meth:`__call__` on the control thread, so the
    controller and keyboard state are never modified mid-step.

    Outside of IBVS the keyboard's gripper value is the gripper command. When IBVS
    hands control back (grasp done or interrupted) the controller's gripper command is
    copied into the keyboard so the gripper does not snap open and drop the block.
    """

    def __init__(
        self,
        env,
        teleop: Se3Keyboard,
        controller_factory: Callable[[np.ndarray, np.ndarray], IBVSController],
        controller: IBVSController | None = None,
        setpoint_dir: str | Path = "exp/ibvs_setpoints",
        home_tcp: np.ndarray = HOME_TCP_CART,
    ) -> None:
        """Initialize the adapter.

        Args:
            env: Kortex TCP Gen3 environment.
            teleop: Keyboard device used for teleoperation.
            controller_factory: ``(setpoint_rgb, K) -> IBVSController``, used when a new
                setpoint is captured with B.
            controller: Initial controller, or ``None`` until a setpoint is captured.
            setpoint_dir: Where captured setpoints are written.
            home_tcp: Home pose ``[x, y, z, rx, ry, rz, gripper]`` (the gripper entry is
                replaced with the current gripper command).

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
        action = np.asarray(self.teleop.advance(self.env._robot_tool_pose_b), dtype=np.float64)
        return action, ActionSpec.TWIST_TCP

    def _ibvs_action(self) -> tuple[np.ndarray, ActionSpec]:
        assert self.controller is not None
        rgbd = self.env._get_latest_rgbd_wrist()
        if rgbd is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.controller.gripper_cmd]), ActionSpec.TWIST_TCP

        rgb, depth, _ = rgbd_to_numpy(rgbd)
        pose = self.env._robot_tool_pose_b
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        # KortexEnv returns [x, y, z, qx, qy, qz, qw], which is what IBVSController expects.
        tcp_pose = np.asarray(pose, dtype=np.float64).reshape(-1)[:7].copy()
        twist = np.asarray(self.controller(rgb, depth, tcp_pose=tcp_pose, dt=self.env.step_dt), dtype=np.float64)

        if self.controller.phase is GraspPhase.DONE:
            self._return_to_teleop("Grasp complete")

        return twist, ActionSpec.TWIST_TCP
