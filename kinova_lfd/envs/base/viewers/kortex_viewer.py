"""kortex_viewer.py.

OpenCV viewer for Kortex (real-robot) environments. Mirrors the ``MjViewer`` API
(``run`` / ``open_display`` / ``sync_display`` / ``close_display`` / ``viewer_closed``)
but, instead of a MuJoCo window, shows the real wrist and tabletop cameras side by side
in a single OpenCV window, both scaled to ``preview_height``.

Pressing the record key (default V, read globally through pynput like the teleop
keyboard) starts recording the displayed window, overlays included; pressing it again
stops and saves an ``.mp4`` to ``video_dir``.

Written by Will Solow, 2026.
"""

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kinova_lfd.core.specs import ActionSpec
from kinova_lfd.envs.base.gym import GymEnv
from kinova_lfd.envs.base.viewers.recorder import Preview, RgbdFrame, as_rgbd_frame
from kinova_lfd.teleop.devices.device_base import DeviceBase


class KortexPolicyProtocol(Protocol):
    """Policy accepted by :class:`KortexViewer`.

    ``__call__`` returns either an action (sent as ``TWIST_TCP``) or an
    ``(action, ActionSpec)`` tuple. The optional attributes below are used when present:

    - ``should_quit: bool``: the viewer exits its loop when this becomes ``True``.
    - ``status_text: str``: drawn on the camera previews.
    - ``annotate_wrist(bgr: np.ndarray) -> np.ndarray``: extra drawing on the wrist preview.
    - ``start()`` / ``stop()``: called when the viewer opens / closes.
    """

    def __call__(self, observation: Any) -> np.ndarray | tuple[np.ndarray, ActionSpec]: ...


class KortexViewer:
    """Run a policy on a Kortex env and preview the robot cameras in OpenCV windows.

    The env paces itself to ``cfg.dt`` inside ``step``, so the control loop runs at
    the env rate. Frames are drawn on a separate daemon thread.

    With ``pull_frames=True`` (default, used by :meth:`run`) the preview thread fetches
    frames itself, so the blocking RealSense ``wait_for_frames`` never stalls the control
    loop, and it is the only caller of ``env._get_latest_rgbd()``. With
    ``pull_frames=False`` (external loops that already read the cameras, e.g. LeRobot
    recording) the viewer never touches the cameras; pass frames to :meth:`sync_display`.
    """

    def __init__(
        self,
        env: GymEnv,
        policy: KortexPolicyProtocol | DeviceBase | None = None,
        view_wrist: bool = True,
        view_tabletop: bool = True,
        preview_fps: float = 20.0,
        preview_height: int = 360,
        record_key: str | None = "v",
        video_dir: str | Path = "exp/ibvs_videos",
        pull_frames: bool = True,
    ) -> None:
        """Initialize the viewer.

        Args:
            env: Kortex environment. Needs ``cfg.use_wrist_camera`` /
                ``cfg.use_tabletop_camera`` for the corresponding preview.
            policy: Policy or teleop device. ``None`` holds the robot still.
            view_wrist: Show the wrist camera.
            view_tabletop: Show the tabletop RealSense camera.
            preview_fps: Preview refresh rate, also the frame rate of recorded videos.
            preview_height: Height (px) of each camera pane.
            record_key: Key that toggles video recording, or ``None`` to disable recording.
            video_dir: Directory where recorded videos are saved.
            pull_frames: Fetch frames from the env on the preview thread. Set ``False`` when
                the caller already reads the cameras and pushes frames via :meth:`sync_display`.

        """
        self.env = env
        self.policy = policy
        cfg = getattr(env, "cfg", None)
        view_wrist_ok = view_wrist and bool(getattr(cfg, "use_wrist_camera", False))
        view_tabletop_ok = view_tabletop and bool(getattr(cfg, "use_tabletop_camera", False))

        # Pull mode: the preview thread fetches frames itself, so the blocking RealSense
        # ``wait_for_frames`` never stalls the control loop. Push mode: sync_display supplies them.
        self._preview = Preview(
            "Kortex Cameras",
            view_wrist=view_wrist_ok,
            view_tabletop=view_tabletop_ok,
            fps=preview_fps,
            height=preview_height,
            frame_source=self._fetch_frames if pull_frames else None,
            annotate_wrist=getattr(policy, "annotate_wrist", None),
            record_key=record_key,
            video_dir=video_dir,
            video_prefix="kortex",
            log_tag="KortexViewer",
        )
        self._opened = False
        self._step_count = 0

    def run(self) -> None:
        """Run the control loop until the policy quits, a window closes, or Ctrl-C."""
        self.open_display()
        try:
            while self.sync_display():
                if bool(getattr(self.policy, "should_quit", False)):
                    print("[INFO][KortexViewer] Policy requested quit.")
                    break
                self._execute_step()
        except KeyboardInterrupt:
            print("[INFO][KortexViewer] Interrupted.")
        finally:
            self._hold_robot()
            self.close_display()
            self.env.close()

    def open_display(self) -> None:
        """Start the policy and the camera preview thread (for external control loops)."""
        if self._opened:
            return
        self._opened = True
        start = getattr(self.policy, "start", None)
        if callable(start):
            start()
        self._preview.start()

    def sync_display(self, wrist: Any | None = None, tabletop: Any | None = None) -> bool:
        """Publish the policy status, and any pushed camera frames, to the preview thread.

        Args:
            wrist: Already-captured wrist RGB image (push mode). Ignored in pull mode.
            tabletop: Already-captured tabletop RGB image (push mode). Ignored in pull mode.

        Returns:
            ``False`` once a preview window has been closed, else ``True``.

        """
        if self._preview.closed:
            self._preview.stop()
            return False
        self._preview.publish(
            wrist=as_rgbd_frame(wrist) if self._preview.view_wrist else None,
            tabletop=as_rgbd_frame(tabletop) if self._preview.view_tabletop else None,
            status=str(getattr(self.policy, "status_text", "")),
        )
        return True

    def close_display(self) -> None:
        """Stop the preview thread and the policy. Safe to call repeatedly."""
        try:
            self._preview.stop()
        finally:
            stop = getattr(self.policy, "stop", None)
            if self._opened and callable(stop):
                stop()
            self._opened = False

    @property
    def viewer_closed(self) -> bool:
        """Whether a preview window was closed by the user."""
        return self._preview.closed

    def _compute_action(self) -> tuple[np.ndarray, ActionSpec]:
        """Query the policy for the next action and its specification."""
        if self.policy is None:
            return np.zeros(7), ActionSpec.TWIST_TCP
        if isinstance(self.policy, DeviceBase):
            return np.asarray(self.policy(self.env._robot_tool_pose_b)), ActionSpec.TWIST_TCP
        out = self.policy(self.env._get_observations())
        if isinstance(out, tuple):
            action, action_spec = out
        else:
            action, action_spec = out, ActionSpec.TWIST_TCP
        return np.asarray(action, dtype=np.float64).reshape(-1), action_spec

    def _execute_step(self) -> bool:
        """Run one obs/policy/step cycle."""
        action, action_spec = self._compute_action()
        self.env.step(action, action_spec=action_spec)
        self._step_count += 1
        return True

    def _hold_robot(self) -> None:
        """Send a single zero twist (gripper unchanged) so the arm stops on exit."""
        try:
            prev = np.asarray(getattr(self.env, "_prev_actions", np.zeros(7))).reshape(-1)
            gripper = float(prev[-1]) if prev.size else 0.0
            self.env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]), action_spec=ActionSpec.TWIST_TCP)
        except Exception:
            pass

    def _fetch_frames(self) -> tuple[RgbdFrame, RgbdFrame]:
        """Latest ``(wrist, tabletop)`` RGB-D frames; called on the preview thread."""
        wrist = self.env._get_latest_rgbd_wrist() if self._preview.view_wrist else None
        tabletop = self.env._get_latest_rgbd() if self._preview.view_tabletop else None
        return wrist, tabletop

    @property
    def recording(self) -> bool:
        """Whether a recording is in progress."""
        return self._preview.recording
