"""Fixed-rate record loop for Gen3 LeRobot robots."""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.visualization_utils import log_visualization_data

logger = logging.getLogger(__name__)


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict[str, bool],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    timer: CycleTimer | None = None,
    mj_viewer: Any | None = None,
) -> None:
    """Drive the robot from teleop at ``fps``, optionally recording each frame.

    Matches the LeRobot ``record_loop`` contract: teleop actions are processed then
    sent; the processed teleop action (TWIST_TCP after our delta processor) is what
    gets written into the dataset.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"Dataset fps ({dataset.fps}) must equal requested fps ({fps}).")
    if control_time_s is None:
        raise ValueError("`control_time_s` is required.")

    if timer is None:
        timer = CycleTimer(fps, records_data=dataset is not None)

    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        if events.get("exit_early"):
            events["exit_early"] = False
            break

        timer.tick()

        with timer.section("observe"):
            obs = robot.get_observation()

        with timer.section("process_obs"):
            obs_processed = robot_observation_processor(obs)
            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        with timer.section("teleop"):
            if teleop is None:
                timer.wait()
                timestamp = time.perf_counter() - start_episode_t
                continue

            # Merge gamepad teleop episode events into the shared events dict when available.
            if hasattr(teleop, "get_teleop_events"):
                tevents = teleop.get_teleop_events()
                from lerobot.teleoperators.utils import TeleopEvents

                if tevents.get(TeleopEvents.TERMINATE_EPISODE) or tevents.get(TeleopEvents.SUCCESS):
                    events["exit_early"] = True
                if tevents.get(TeleopEvents.RERECORD_EPISODE):
                    events["rerecord_episode"] = True
                    events["exit_early"] = True

            act = teleop.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed, obs))

        with timer.section("send"):
            _sent = robot.send_action(robot_action_to_send)

        if mj_viewer is not None:
            with timer.section("viewer"):
                mj_viewer.sync_display()

        if dataset is not None:
            with timer.section("record"):
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

        if display_data:
            with timer.section("telemetry"):
                log_visualization_data(
                    display_mode,
                    observation=obs_processed,
                    action=act_processed,
                    compress_images=display_compressed_images,
                )

        timer.wait()
        timestamp = time.perf_counter() - start_episode_t
