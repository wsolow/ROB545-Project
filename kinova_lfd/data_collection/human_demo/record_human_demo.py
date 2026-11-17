"""Record human demonstration videos from a RealSense camera.

Each saved episode produces two outputs:

1. A LeRobot dataset in ``<root>/`` (no ``action`` feature) holding
   ``observation.images.tabletop`` (RGB video) and ``observation.intrinsic_k``
   (flattened 3x3 camera matrix).
2. A raw folder in the ``collect_rgbd_demo.py`` layout, used by the hand / object pose
   tools that run before retargeting::

       <root>/raw/episode_000000/rgb/00000.png     # 8-bit color PNG
       <root>/raw/episode_000000/depth/00000.png   # uint16 depth in millimeters, aligned to color
       <root>/raw/episode_000000/cam_K.txt         # 3x3 intrinsics

   Raw episode folders use the same index as the LeRobot ``episode_index``.

Written by Will Solow, 2026.
"""

import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import init_logging

from kinova_lfd.data_collection.configs import Gen3DatasetConfig
from kinova_lfd.data_collection.controls import init_recording_listener
from kinova_lfd.envs.base.cameras.camera import CameraEnv, RealsenseCameraCfg

CAMERA_KEY = "tabletop"
"""Camera name in the dataset: frames are stored as ``observation.images.tabletop``."""
INTRINSICS_KEY = "observation.intrinsic_k"


@dataclass
class HumanDemoRecordConfig:
    """Top-level settings for human demonstration recording."""

    camera: RealsenseCameraCfg = field(default_factory=RealsenseCameraCfg)
    dataset: Gen3DatasetConfig = field(default_factory=lambda: Gen3DatasetConfig(fps=30))
    save_raw: bool = True
    """Also write the raw ``rgb/`` + ``depth/`` PNG folder for each saved episode."""
    view: bool = True
    """Show the live camera feed in an OpenCV window."""


class RawEpisodeWriter:
    """Write one episode's RGB / depth PNGs on a background thread.

    Frames go to ``<raw_root>/.episode_in_progress`` and the folder is renamed to
    ``episode_XXXXXX`` on :meth:`save`, or deleted on :meth:`discard`, so a discarded or
    interrupted episode never leaves a half-written folder that looks complete.
    """

    def __init__(self, raw_root: Path, depth_scale: float) -> None:
        self.raw_root = Path(raw_root)
        self._mm_per_unit = depth_scale * 1000.0
        self._tmp_dir = self.raw_root / ".episode_in_progress"
        self._queue: queue.Queue[tuple[int, np.ndarray, np.ndarray | None] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._num_frames = 0
        self._intrinsic_k: np.ndarray | None = None
        self._error: BaseException | None = None

    def start_episode(self) -> None:
        """Clear any leftover temp folder and start the writer thread."""
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        (self._tmp_dir / "rgb").mkdir(parents=True)
        (self._tmp_dir / "depth").mkdir(parents=True)
        self._num_frames = 0
        self._intrinsic_k = None
        self._error = None
        self._thread = threading.Thread(target=self._run, name="RawEpisodeWriter", daemon=True)
        self._thread.start()

    def add_frame(self, rgb_hwc: np.ndarray, depth: np.ndarray | None, intrinsic_k: np.ndarray) -> None:
        """Queue one frame: ``rgb_hwc`` is (H, W, 3) uint8 RGB, ``depth`` is (H, W) raw depth units."""
        if self._error is not None:
            raise RuntimeError("Raw episode writer failed") from self._error
        if self._intrinsic_k is None:
            self._intrinsic_k = np.asarray(intrinsic_k, dtype=np.float64).reshape(3, 3)
        self._queue.put((self._num_frames, rgb_hwc, depth))
        self._num_frames += 1

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def save(self, episode_index: int) -> Path:
        """Finish writing, add ``cam_K.txt``, and rename the folder to ``episode_<index>``."""
        self._finish()
        if self._error is not None:
            raise RuntimeError("Raw episode writer failed") from self._error
        assert self._intrinsic_k is not None, "Cannot save a raw episode with no frames."
        np.savetxt(self._tmp_dir / "cam_K.txt", self._intrinsic_k)
        final_dir = self.raw_root / f"episode_{episode_index:06d}"
        if final_dir.exists():
            raise FileExistsError(f"Raw episode folder {final_dir} already exists.")
        self._tmp_dir.rename(final_dir)
        return final_dir

    def discard(self) -> None:
        """Finish writing and delete the episode's frames."""
        self._finish()
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)

    def _finish(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue  # drain the queue after a failure
            idx, rgb_hwc, depth = item
            try:
                name = f"{idx:05d}.png"
                cv2.imwrite(str(self._tmp_dir / "rgb" / name), cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))
                if depth is not None:
                    if self._mm_per_unit != 1.0:
                        depth = np.clip(np.rint(depth * self._mm_per_unit), 0, 65535)
                    cv2.imwrite(str(self._tmp_dir / "depth" / name), depth.astype(np.uint16, copy=False))
            except BaseException as exc:  # surfaced to the record loop on the next add_frame / save
                self._error = exc


def human_demo_features(height: int, width: int, use_videos: bool) -> dict[str, dict]:
    """LeRobot features for a human demo: one RGB camera plus intrinsics, no action."""
    features = hw_to_dataset_features({CAMERA_KEY: (height, width, 3)}, "observation", use_video=use_videos)
    features[INTRINSICS_KEY] = {"dtype": "float32", "shape": (9,), "names": None}
    return features


def record_human_demo(cfg: HumanDemoRecordConfig) -> LeRobotDataset:
    """Record human demonstration episodes from the RealSense camera.

    Keyboard controls (global, through pynput): ``]`` / Enter end & save the episode,
    ``[`` or R discard and re-record it, Esc stop the session. After each saved episode
    (except the last) a reset period runs with the preview on and nothing recorded;
    ``]`` / Enter ends it early.
    """
    init_logging()
    dataset_cfg = cfg.dataset
    fps = dataset_cfg.fps
    root = dataset_cfg.resolved_root()

    env = CameraEnv(cfg.camera)
    viewer = None
    listener = None
    dataset: LeRobotDataset | None = None
    raw_writer: RawEpisodeWriter | None = None
    timer = CycleTimer(fps)

    try:
        # Start the camera before creating the dataset so a camera failure leaves nothing on disk.
        first = env.reset()
        _, height, width = first["rgb"].shape

        if dataset_cfg.resume:
            dataset = LeRobotDataset.resume(
                dataset_cfg.repo_id,
                root=root,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
                streaming_encoding=dataset_cfg.streaming_encoding,
                image_writer_processes=dataset_cfg.num_image_writer_processes,
                image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera,
            )
            if dataset.fps != fps:
                raise ValueError(f"Dataset fps ({dataset.fps}) must equal requested fps ({fps}).")
        else:
            dataset = LeRobotDataset.create(
                dataset_cfg.repo_id,
                fps,
                root=root,
                robot_type="human_demo_realsense",
                features=human_demo_features(height, width, dataset_cfg.video),
                use_videos=dataset_cfg.video,
                image_writer_processes=dataset_cfg.num_image_writer_processes,
                image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
                streaming_encoding=dataset_cfg.streaming_encoding,
            )

        if cfg.save_raw:
            raw_writer = RawEpisodeWriter(root / "raw", depth_scale=env.camera.depth_scale)

        if cfg.view:
            from kinova_lfd.envs.base.viewers.camera_viewer import CameraViewer

            viewer = CameraViewer(env, preview_fps=fps, preview_height=height)
            viewer.open_display()

        print(
            "\nRecording controls (keyboard):\n"
            "  ] or Enter -> end episode & save (or end the reset period)\n"
            "  [ or R     -> discard & re-record\n"
            "  Esc        -> stop session\n"
        )
        listener, events = init_recording_listener()

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < dataset_cfg.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                print(f"[record] Recording episode {episode_index} ({recorded + 1}/{dataset_cfg.num_episodes})...")
                if raw_writer is not None:
                    raw_writer.start_episode()
                _record_loop(
                    env, events, fps, dataset_cfg.episode_time_s, timer, viewer,
                    dataset=dataset, raw_writer=raw_writer, task=dataset_cfg.single_task,
                    label=f"REC ep {episode_index}",
                )  # fmt: skip

                discard = events["stop_recording"] or events["rerecord_episode"] or events["reset_environment"]
                if not discard and not dataset.has_pending_frames():
                    print("[record] Empty episode; discarding.")
                    discard = True
                if discard:
                    dataset.clear_episode_buffer()
                    if raw_writer is not None:
                        raw_writer.discard()
                    timer.log_episode_summary("discarded episode")
                else:
                    dataset.save_episode()
                    msg = f"[record] Saved episode {episode_index}"
                    if raw_writer is not None:
                        msg += f" (raw: {raw_writer.save(episode_index)})"
                    print(msg)
                    recorded += 1
                    timer.log_episode_summary(f"episode {episode_index}")
                for key in ("exit_early", "rerecord_episode", "reset_environment"):
                    events[key] = False
                timer.restart()

                if events["stop_recording"] or recorded >= dataset_cfg.num_episodes:
                    break
                if not discard and dataset_cfg.reset_time_s > 0:
                    print(
                        f"[record] Reset period ({dataset_cfg.reset_time_s:.0f}s); press ] to start the next episode."
                    )
                    _record_loop(env, events, fps, dataset_cfg.reset_time_s, None, viewer, label="RESET")
                    # Any key during the reset period only ends it; it must not discard the next episode.
                    for key in ("exit_early", "rerecord_episode", "reset_environment"):
                        events[key] = False
    finally:
        timer.log_run_summary()
        if raw_writer is not None:
            raw_writer.discard()  # removes only an unsaved in-progress episode
        if viewer is not None:
            viewer.close_display()
        if dataset is not None:
            dataset.finalize()
        env.close()
        if listener is not None:
            listener.stop()
        if dataset_cfg.push_to_hub and dataset is not None and dataset.num_episodes > 0:
            dataset.push_to_hub()

    assert dataset is not None
    return dataset


def _record_loop(
    env: CameraEnv,
    events: dict[str, bool],
    fps: int,
    control_time_s: float,
    timer: CycleTimer | None,
    viewer: Any | None,
    dataset: LeRobotDataset | None = None,
    raw_writer: RawEpisodeWriter | None = None,
    task: str | None = None,
    label: str = "",
) -> None:
    """Show (and, with ``dataset``, record) camera frames at ``fps`` for ``control_time_s`` seconds.

    Each tick waits up to one period for a camera frame newer than the last one, so when
    ``fps`` matches the camera rate every recorded frame is a distinct camera frame.
    """
    timer = timer or CycleTimer(fps, records_data=False)
    period = 1.0 / fps
    last_ts: float | None = None
    start = time.perf_counter()
    while (elapsed := time.perf_counter() - start) < control_time_s:
        if events.get("exit_early"):
            break
        timer.tick()

        with timer.section("observe"):
            obs = _wait_for_new_frame(env, last_ts, timeout=period)
        if obs is None:
            timer.wait()
            continue
        last_ts = obs["timestamp"]
        rgb_hwc = np.ascontiguousarray(obs["rgb"].transpose(1, 2, 0))

        if viewer is not None:
            status = f"{label}  {elapsed:5.1f}/{control_time_s:.0f}s"
            if not viewer.sync_display(rgb_hwc, status=status):
                events["stop_recording"] = True
                events["exit_early"] = True
                break

        if dataset is not None:
            with timer.section("record"):
                dataset.add_frame(
                    {
                        f"observation.images.{CAMERA_KEY}": rgb_hwc,
                        INTRINSICS_KEY: np.asarray(obs["intrinsic_k"], dtype=np.float32).reshape(9),
                        "task": task,
                    }
                )
                if raw_writer is not None:
                    raw_writer.add_frame(rgb_hwc, obs["depth"], obs["intrinsic_k"])

        timer.wait()


def _wait_for_new_frame(env: CameraEnv, last_ts: float | None, timeout: float) -> dict[str, Any] | None:
    """Return the latest frame once it is newer than ``last_ts``, or the latest frame after ``timeout``."""
    deadline = time.perf_counter() + timeout
    obs = env.step()
    while obs is not None and obs["timestamp"] == last_ts and time.perf_counter() < deadline:
        time.sleep(0.001)
        obs = env.step()
    return obs
