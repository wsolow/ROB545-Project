"""Gen3 demonstration recording: episode orchestration and the fixed-rate record loop."""

import time
from functools import partial
from typing import Any

from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.utils import init_logging

from .configs import Gen3RecordConfig
from .controls import init_recording_listener, poll_source_controls
from .factories import make_robot, make_teleop, make_viewer


def record(cfg: Gen3RecordConfig) -> LeRobotDataset:
    """Record Gen3 demonstrations from a teleoperator or an RSL-RL policy."""
    init_logging()

    if cfg.teleop is not None and cfg.policy is not None:
        raise ValueError("Configure exactly one action source: teleop or policy.")
    if cfg.teleop is None and cfg.policy is None:
        raise ValueError("Configure an action source: teleop or policy.")

    robot = make_robot(cfg.robot)
    # Teleops and policies share connect / disconnect / reset / get_action.
    if cfg.policy is not None:
        from .policy import RslRlPolicySource

        source: Any = RslRlPolicySource(cfg.policy, robot)
    else:
        source = make_teleop(cfg.teleop)
    is_policy = cfg.policy is not None

    # Teleops and policies both emit the robot's native action keys: identity pipelines.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_cfg = cfg.dataset.to_lerobot_dataset_config()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_cfg.video,
        ),
    )

    root = cfg.dataset.resolved_root()
    num_cameras = sum(1 for v in robot.observation_features.values() if isinstance(v, tuple) and len(v) == 3)

    timer = CycleTimer(dataset_cfg.fps)
    listener = None
    dataset: LeRobotDataset | None = None
    viewer = None

    try:
        if cfg.dataset.resume:
            dataset = LeRobotDataset.resume(
                dataset_cfg.repo_id,
                root=root,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
                streaming_encoding=dataset_cfg.streaming_encoding,
                image_writer_processes=dataset_cfg.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=(
                    dataset_cfg.num_image_writer_threads_per_camera * num_cameras if num_cameras > 0 else 0
                ),
            )
        else:
            dataset = LeRobotDataset.create(
                dataset_cfg.repo_id,
                dataset_cfg.fps,
                root=root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=dataset_cfg.video,
                image_writer_processes=dataset_cfg.num_image_writer_processes,
                image_writer_threads=dataset_cfg.num_image_writer_threads_per_camera * num_cameras,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
                streaming_encoding=dataset_cfg.streaming_encoding,
            )

        robot.connect()
        source.connect()

        if cfg.view:
            viewer = make_viewer(robot, view_wrist=cfg.view_wrist, view_tabletop=cfg.view_tabletop)
            if viewer is not None:
                viewer.open_display()
                print(f"[record] {type(viewer).__name__} opened (syncs each control step).")

        if is_policy:
            print("\nRecording policy episodes; only cube-goal successes will be saved. Esc stops.")
        else:
            print(
                "\nRecording controls (keyboard, always active; Xbox buttons are in the device banner):\n"
                "  ] or Enter -> end episode & save\n"
                "  [          -> discard & re-record\n"
                "  R          -> reset environment, discard episode\n"
                "  Esc        -> reset environment & stop session\n"
            )

        listener, events = init_recording_listener()
        run_loop = partial(
            record_loop,
            robot=robot,
            events=events,
            fps=dataset_cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            source=source,
            single_task=dataset_cfg.single_task,
            viewer=viewer,
        )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < dataset_cfg.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                # Every episode starts from fresh source state: zero teleop motion, open
                # gripper, no stale button presses / cleared recurrent policy state.
                source.reset()
                run_loop(dataset=dataset, control_time_s=dataset_cfg.episode_time_s, timer=timer)

                if events["reset_environment"]:
                    dataset.clear_episode_buffer()
                    _reset_env(robot)
                    events["reset_environment"] = False
                    events["exit_early"] = False
                    timer.log_episode_summary("reset episode")
                    timer.restart()
                    continue  # the loop condition handles a pending stop

                if events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    break

                if is_policy:
                    from .policy import policy_episode_succeeded

                    if not policy_episode_succeeded(robot, cfg.policy.success_delta):
                        events["rerecord_episode"] = True
                elif recorded_episodes < dataset_cfg.num_episodes - 1 or events["rerecord_episode"]:
                    run_loop(control_time_s=dataset_cfg.reset_time_s)

                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if is_policy:
                        _reset_env(robot)
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                if not dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                    timer.log_episode_summary("discarded empty episode")
                    timer.restart()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        timer.log_run_summary()

        if viewer is not None:
            viewer.close_display()

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if source.is_connected:
            source.disconnect()

        if listener is not None:
            listener.stop()

        if dataset_cfg.push_to_hub and dataset is not None and dataset.num_episodes > 0:
            dataset.push_to_hub()

        # Perform all resource cleanup before optional speech. Blocking audio
        # must not delay shutdown after Ctrl+C or a closed MuJoCo window.

    assert dataset is not None
    return dataset


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict[str, bool],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    source: Any | None = None,
    dataset: LeRobotDataset | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    timer: CycleTimer | None = None,
    viewer: Any | None = None,
) -> None:
    """Drive the robot from ``source`` (teleop or policy) at ``fps``, optionally recording each frame.

    Matches the LeRobot ``record_loop`` contract: source actions are processed then
    sent; the processed action (the robot's native keys) is what gets written into the dataset.

    Each tick the source receives the raw observation (``update_observation``, used for
    base-frame teleop), and its device controls / teleop events are merged into ``events``.
    ``viewer`` (``MjViewer`` / ``KortexViewer``) is fed the camera frames already in the
    observation instead of fetching its own.
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
            if source is None:
                timer.wait()
                timestamp = time.perf_counter() - start_episode_t
                continue

            update_observation = getattr(source, "update_observation", None)
            if callable(update_observation):
                update_observation(obs)
            poll_source_controls(source, events)

            if events.get("exit_early"):
                events["exit_early"] = False
                break

            act = source.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action_to_send = robot_action_processor((act_processed, obs))

        with timer.section("send"):
            robot.send_action(robot_action_to_send)

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


def _reset_env(robot: Robot) -> None:
    """Reset the robot's environment (warns for backends without one)."""
    reset = getattr(getattr(robot, "env", None), "reset", None)
    if callable(reset):
        reset()
