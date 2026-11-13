from collections.abc import Callable
from pathlib import Path
from typing import Any, override

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
from mjlab.viewer.native.keys import KEY_A, KEY_P, KEY_R
from mjlab.viewer.native.viewer import PlotCfg

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.envs.base.viewers.recorder import Preview, as_rgbd_frame
from kinova_lfd.teleop.devices.device_base import DeviceBase

# mjlab binds these to debug toggles; teleop policies use them to move the arm / gripper.
_TELEOP_KEYS = (KEY_A, KEY_P, KEY_R)


class MjViewer(NativeMujocoViewer):
    """MuJoCo viewer with a combined OpenCV camera window, like ``KortexViewer``.

    The wrist and tabletop cameras are shown side by side in one window, scaled to
    ``preview_height``. Pressing ``record_key`` (default V, read globally through
    pynput) records the window, overlays included, to ``video_dir``.

    Policies may return an action or an ``(action, ActionSpec)`` tuple, and may expose
    the optional ``KortexPolicyProtocol`` hooks: ``should_quit``, ``status_text``,
    ``annotate_wrist(bgr)``, ``start()`` / ``stop()``.
    """

    def __init__(
        self,
        env: GymEnv,
        policy: PolicyProtocol | DeviceBase | None = None,
        frame_rate: float = 60.0,
        key_callback: Callable[[int], None] | None = None,
        plot_cfg: PlotCfg | None = None,
        enable_perturbations: bool = True,
        verbosity: VerbosityLevel = VerbosityLevel.SILENT,
        view_wrist: bool = False,
        view_tabletop: bool = True,
        ignore_mjlab_keys: bool = False,
        preview_fps: float = 20.0,
        preview_height: int = 360,
        record_key: str | None = "v",
        video_dir: str | Path = "exp/ibvs_videos",
    ):
        super().__init__(env, policy, frame_rate, key_callback, plot_cfg, enable_perturbations, verbosity)
        if isinstance(self.policy, DeviceBase):
            self.policy.start()
        self._policy_started = isinstance(self.policy, DeviceBase)
        self._ignore_mjlab_keys = ignore_mjlab_keys
        # Push mode: frames are read on the sim thread in _update_camera_previews.
        self._preview = Preview(
            "MuJoCo Cameras",
            view_wrist=view_wrist,
            view_tabletop=view_tabletop,
            fps=preview_fps,
            height=preview_height,
            annotate_wrist=getattr(self.policy, "annotate_wrist", None),
            record_key=record_key,
            video_dir=video_dir,
            video_prefix="mujoco",
            log_tag="MjViewer",
        )
        self._viewer_closed = False

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

        if not self._policy_started:
            start = getattr(self.policy, "start", None)
            if callable(start):
                start()
            self._policy_started = True
        self._preview.start()

    @override
    def _safe_key_callback(self, key: int) -> None:
        # During recording, skip mjlab Enter/P/Space/etc. so they do not reset
        # or toggle plots. Simulate built-in letter hotkeys still cannot be suppressed.
        if self._ignore_mjlab_keys:
            return
        # A/R move the arm and P toggles the gripper. The base viewer also binds them to debug toggles.
        teleop = isinstance(self.policy, DeviceBase) or hasattr(self.policy, "teleop")
        if teleop and key in _TELEOP_KEYS:
            return
        super()._safe_key_callback(key)

    def open_display(self) -> None:
        """Launch the passive MuJoCo window for an external control loop.

        Use this from LeRobot recording / custom loops that own ``env.step``
        themselves. Call :meth:`sync_display` after each step, then
        :meth:`close_display` when finished. Do **not** call :meth:`run`.
        """
        self.setup()

    def sync_display(self, wrist: Any | None = None, tabletop: Any | None = None) -> bool:
        """Submit camera previews and refresh the passive viewer without waiting for GUI work.

        Args:
            wrist: Already-captured wrist RGB image (e.g. from the robot observation). Shown
                as-is instead of re-rendering the camera; ``None`` renders it from the env.
            tabletop: Same as ``wrist`` for the tabletop camera.

        Returns:
            ``False`` once the MuJoCo window has been closed, else ``True``.

        """
        if self.viewer is None or not self.is_running():
            self._viewer_closed = True
            self._preview.stop()
            return False
        self._update_camera_previews(wrist=wrist, tabletop=tabletop)
        self.sync_env_to_viewer()
        return True

    def close_display(self) -> None:
        """Close the passive viewer and terminate the camera preview thread."""
        self.close()

    @override
    def close(self) -> None:
        """Stop recording / preview / policy, then close the MuJoCo window. Safe to call repeatedly."""
        try:
            self._preview.stop()
        finally:
            stop = getattr(self.policy, "stop", None)
            if self._policy_started and callable(stop):
                stop()
            self._policy_started = False
            super().close()

    @property
    def viewer_closed(self) -> bool:
        """Whether the MuJoCo window was closed by the user."""
        return self._viewer_closed

    def _update_camera_previews(self, wrist: Any | None = None, tabletop: Any | None = None) -> None:
        """Publish camera frames and policy status to the preview thread.

        Frames passed in (already-captured RGB images) are used directly; cameras without
        one are rendered from the env.
        """
        if not self._preview.enabled:
            return
        env = self.env.unwrapped
        wrist_frame = tabletop_frame = None
        if self._preview.view_wrist:
            wrist_frame = as_rgbd_frame(wrist) if wrist is not None else env._get_latest_rgbd_wrist()
        if self._preview.view_tabletop:
            tabletop_frame = as_rgbd_frame(tabletop) if tabletop is not None else env._get_latest_rgbd()
        self._preview.publish(
            wrist=wrist_frame,
            tabletop=tabletop_frame,
            status=str(getattr(self.policy, "status_text", "")),
        )

    @override
    def _execute_step(self) -> bool:
        """Run one obs/policy/step cycle.

        Returns True on success, False if step failed.
        """
        with torch.no_grad():
            obs = self.env.get_observations()
            self._update_camera_previews()
            if self.policy is not None:
                if isinstance(self.policy, DeviceBase):
                    actions = self.policy(self.env._robot_tool_pose_b)
                    action_spec = ActionSpec.TWIST_TCP
                else:
                    out = self.policy(obs)
                    actions, action_spec = out if isinstance(out, tuple) else (out, ActionSpec.TWIST_TCP)
                    if bool(getattr(self.policy, "should_quit", False)):
                        print("[INFO][MjViewer] Policy requested quit.")
                        self._interrupted = True
                        return True
                actions = torch.as_tensor(actions, device=self.env.device, dtype=torch.float32)
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0)
            else:
                actions = torch.zeros(self.env.action_space.shape, device=self.env.device)
                action_spec = ActionSpec.JOINTS_POS
            self.env.step(actions, action_spec=action_spec)
            self._step_count += 1
            self._stats_steps += 1
            return True
