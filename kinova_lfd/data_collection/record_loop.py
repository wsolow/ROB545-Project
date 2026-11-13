"""Fixed-rate record loop for Gen3 LeRobot robots."""

import time
from typing import Any

from lerobot.datasets import LeRobotDataset, safe_stop_image_writer
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame

from .controls import apply_recording_control


def _apply_teleop_events(teleop_events: dict[TeleopEvents, bool], events: dict[str, bool]) -> None:
    """Map standard LeRobot teleop episode events onto the recording controls.

    Success ends and saves the episode; re-record discards and re-records; a failure or
    a bare terminate discards the episode and resets the environment (never saves).
    """
    if teleop_events.get(TeleopEvents.RERECORD_EPISODE):
        apply_recording_control("rerecord", events)
    elif teleop_events.get(TeleopEvents.SUCCESS):
        apply_recording_control("end", events)
    elif teleop_events.get(TeleopEvents.FAILURE) or teleop_events.get(TeleopEvents.TERMINATE_EPISODE):
        apply_recording_control("reset", events)


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
    action_source: Any | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    timer: CycleTimer | None = None,
    viewer: Any | None = None,
) -> None:
    """Drive the robot from teleop at ``fps``, optionally recording each frame.

    Matches the LeRobot ``record_loop`` contract: teleop actions are processed then
    sent; the processed teleop action (TWIST_TCP) is what gets written into the dataset.

    Each tick the action source receives the raw observation (``update_observation``,
    used for base-frame teleop), and its recording controls / teleop events are merged
    into ``events``. ``viewer`` (``MjViewer`` / ``KortexViewer``) is fed the camera
    frames already in the observation instead of fetching its own.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"Dataset fps ({dataset.fps}) must equal requested fps ({fps}).")
    if control_time_s is None:
        raise ValueError("`control_time_s` is required.")

    if timer is None:
        timer = CycleTimer(fps, records_data=dataset is not None)

    timestamp = 0.0
    start_episode_t = time.perf_counter()
    source = action_source if action_source is not None else teleop
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
            if source is None:
                timer.wait()
                timestamp = time.perf_counter() - start_episode_t
                continue

            update_observation = getattr(source, "update_observation", None)
            if callable(update_observation):
                update_observation(obs)

            # Merge device episode controls (e.g. Xbox buttons) and standard teleop events.
            get_recording_controls = getattr(source, "get_recording_controls", None)
            if callable(get_recording_controls):
                for control in get_recording_controls():
                    apply_recording_control(control, events)
            get_teleop_events = getattr(source, "get_teleop_events", None)
            if callable(get_teleop_events):
                _apply_teleop_events(get_teleop_events(), events)

            if events.get("exit_early"):
                events["exit_early"] = False
                break

            act = source.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed, obs))

        with timer.section("send"):
            _sent = robot.send_action(robot_action_to_send)

        if viewer is not None:
            with timer.section("viewer"):
                if not viewer.sync_display(wrist=obs.get("wrist"), tabletop=obs.get("tabletop")):
                    events["stop_recording"] = True
                    events["exit_early"] = True
                    break

        if dataset is not None:
            with timer.section("record"):
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)

        timer.wait()
        timestamp = time.perf_counter() - start_episode_t
