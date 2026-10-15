from collections.abc import Callable
from typing import override

import cv2
import mujoco
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import (
    PolicyProtocol,
    VerbosityLevel,
)
from mjlab.viewer.model_sync import (
    disable_model_sameframe_shortcuts,
)
from mjlab.viewer.native.keys import KEY_A, KEY_R
from mjlab.viewer.native.viewer import PlotCfg

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.teleop.devices.device_base import DeviceBase


class MjViewer(NativeMujocoViewer):
    def __init__(
        self,
        env: GymEnv,
        policy: PolicyProtocol | DeviceBase | None = None,
        frame_rate: float = 60.0,
        key_callback: Callable[[int], None] | None = None,
        plot_cfg: PlotCfg | None = None,
        enable_perturbations: bool = True,
        verbosity: VerbosityLevel = VerbosityLevel.SILENT,
        view_wrist: bool = True,
        view_tabletop: bool = True,
    ):
        super().__init__(env, policy, frame_rate, key_callback, plot_cfg, enable_perturbations, verbosity)
        if isinstance(self.policy, DeviceBase):
            self.policy.start()
        self._view_wrist = view_wrist
        self._view_tabletop = view_tabletop
        self._rgbd_wrist = None
        self._rgbd_table = None

    @override
    def setup(self) -> None:
        """Set up MuJoCo viewer resources."""
        sim = self.env.unwrapped.sim
        self.mjm = sim.mj_model
        self.mjd = sim.mj_data
        assert self.mjm is not None
        disable_model_sameframe_shortcuts(self.mjm)
        if self.cfg.fovy is not None:
            self.mjm.vis.global_.fovy = self.cfg.fovy

        if self.env.unwrapped.num_envs > 1:
            self.vd = mujoco.MjData(self.mjm)

        self.pert = mujoco.MjvPerturb() if self.enable_perturbations else None
        self.vopt = mujoco.MjvOption()

        assert self.mjm is not None
        assert self.mjd is not None
        self.viewer = mujoco.viewer.launch_passive(
            self.mjm,
            self.mjd,
            key_callback=self._safe_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        if self.viewer is None:
            raise RuntimeError("Failed to launch MuJoCo viewer")

        if not self.cfg.enable_shadows:
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 1

        self._setup_camera()

        if self.enable_perturbations:
            self.log("[INFO] Interactive perturbations enabled", VerbosityLevel.INFO)

    @override
    def _safe_key_callback(self, key: int) -> None:
        # A and R move the arm. The base viewer also binds them to debug toggles.
        if isinstance(self.policy, DeviceBase) and key in (KEY_A, KEY_R):
            return
        super()._safe_key_callback(key)

    @override
    def _execute_step(self) -> bool:
        """Run one obs/policy/step cycle.

        Returns True on success, False if step failed.
        """
        with torch.no_grad():
            obs = self.env.get_observations()
            self._rgbd_wrist = self.env._get_latest_rgbd_wrist()
            if self._view_wrist and self._rgbd_wrist is not None:
                cv2.imshow("RGB Wrist", self._rgbd_wrist["rgb"].squeeze().cpu().numpy().transpose(1, 2, 0))
                cv2.waitKey(1)
            self._rgbd_table = self.env._get_latest_rgbd()
            if self._view_tabletop and self._rgbd_table is not None:
                cv2.imshow("RGB Tabletop", self._rgbd_table["rgb"].squeeze().cpu().numpy().transpose(1, 2, 0))
                cv2.waitKey(1)
            if self.policy is not None:
                if isinstance(self.policy, DeviceBase):
                    actions = self.policy(self.env._robot_tool_pose_b)
                    action_spec = ActionSpec.TWIST_TCP
                else:
                    actions = self.policy(obs)
            else:
                actions = torch.zeros(self.env.action_space.shape, device=self.env.device)
                action_spec = ActionSpec.JOINTS_POS
            self.env.step(actions, action_spec=action_spec)
            self._step_count += 1
            self._stats_steps += 1
            return True
