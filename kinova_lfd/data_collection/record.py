"""Episode orchestration for Gen3 demonstration recording."""

from __future__ import annotations

import logging
from pprint import pformat

from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_robot_observation_processor
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

from .configs import Gen3RecordConfig, teleop_emits_twist_tcp
from .factories import make_robot, make_teleop, register_gen3_robots
from .keyboard_controls import init_gen3_recording_listener
from .processors import make_gen3_teleop_action_processor, make_identity_robot_action_processor
from .record_loop import record_loop
from .success import cube_goal_success


def record(cfg: Gen3RecordConfig) -> LeRobotDataset:
    """Record Gen3 demonstrations from a teleoperator or an RSL-RL policy."""
    init_logging()
    register_gen3_robots()
    logging.info(pformat(cfg))

    if cfg.display_data:
        init_visualization("rerun", session_name="gen3_recording")

    robot = make_robot(cfg.robot)
    if cfg.teleop is not None and cfg.policy is not None:
        raise ValueError("Configure exactly one action source: teleop or policy.")
    if cfg.teleop is None and cfg.policy is None:
        raise ValueError("Configure an action source: teleop or policy.")
    teleop = make_teleop(cfg.teleop) if cfg.teleop is not None else None
    if cfg.policy is not None:
        from .policy import RslRlPolicySource

        policy_source = RslRlPolicySource(cfg.policy, robot)
    else:
        policy_source = None
    action_source = policy_source or teleop

    # gen3_keyboard already emits TWIST_TCP; stock keyboard_ee / gamepad need delta->twist.
    if policy_source is not None or teleop_emits_twist_tcp(cfg.teleop):
        teleop_action_processor = make_identity_robot_action_processor()
    else:
        teleop_action_processor = make_gen3_teleop_action_processor(linear_scale=cfg.linear_scale)
    robot_action_processor = make_identity_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()

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

    timer = CycleTimer(dataset_cfg.fps)
    listener = None
    dataset: LeRobotDataset | None = None
    mj_viewer = None

    try:
        if cfg.resume or cfg.dataset.resume:
            num_cameras = sum(1 for k, v in robot.observation_features.items() if isinstance(v, tuple) and len(v) == 3)
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
            num_cameras = sum(1 for k, v in robot.observation_features.items() if isinstance(v, tuple) and len(v) == 3)
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
        if policy_source is not None:
            policy_source.connect()
        else:
            assert teleop is not None
            teleop.connect()

        if cfg.view_mujoco and getattr(robot, "name", None) == "mj_gen3" and robot.env is not None:
            from kinova_lfd.envs.base.mujoco.viewer import MjViewer

            mj_viewer = MjViewer(
                robot.env,
                policy=None,
                view_wrist=cfg.view_wrist,
                view_tabletop=cfg.view_tabletop,
                ignore_mjlab_keys=True,
            )
            mj_viewer.open_display()
            print("[record] MuJoCo viewer opened (syncs each control step).")

        if policy_source is None:
            print(
                "\nRecording controls:\n"
                "  ] / Enter / gamepad success  -> end episode & save\n"
                "  [         / gamepad rerecord -> discard & rerecord\n"
                "  R         -> reset environment and discard current episode\n"
                "  Esc       / gamepad failure  -> reset environment and stop session\n"
                "  Motion (gen3_keyboard):\n"
                "    Move X: W/S  Y: A/D  Z: Q/E  Rot: I/K J/L U/O  gripper: P\n"
                "  Motion (keyboard_ee/gamepad): stock LeRobot translation + gripper\n"
            )
        else:
            print("\nRecording policy episodes; only cube-goal successes will be saved. Esc stops.")

        listener, events = init_gen3_recording_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < dataset_cfg.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=dataset_cfg.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    action_source=action_source,
                    dataset=dataset,
                    control_time_s=dataset_cfg.episode_time_s,
                    single_task=dataset_cfg.single_task,
                    display_data=cfg.display_data,
                    timer=timer,
                    mj_viewer=mj_viewer,
                )

                if events["reset_environment"]:
                    dataset.clear_episode_buffer()
                    _reset_robot_and_action_source(robot, teleop, policy_source)
                    events["reset_environment"] = False
                    events["exit_early"] = False
                    timer.log_episode_summary("reset episode")
                    timer.restart()
                    if events["stop_recording"]:
                        break
                    continue

                if events["stop_recording"]:
                    dataset.clear_episode_buffer()
                    break

                policy_success = policy_source is not None and _policy_episode_succeeded(
                    robot, cfg.policy.success_delta if cfg.policy is not None else 0.05
                )
                if policy_source is not None and not events["stop_recording"] and not policy_success:
                    events["rerecord_episode"] = True

                if (
                    policy_source is None
                    and not events["stop_recording"]
                    and ((recorded_episodes < dataset_cfg.num_episodes - 1) or events["rerecord_episode"])
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=dataset_cfg.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=dataset_cfg.reset_time_s,
                        single_task=dataset_cfg.single_task,
                        display_data=cfg.display_data,
                        mj_viewer=mj_viewer,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if policy_source is not None:
                        assert robot.env is not None
                        robot.env.reset()
                        policy_source.on_episode_end()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                if not dataset.has_pending_frames():
                    logging.warning("Skipping episode %s because it contains no recorded frames.", episode_index)
                    dataset.clear_episode_buffer()
                    timer.log_episode_summary("discarded empty episode")
                    timer.restart()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                if policy_source is not None:
                    policy_source.on_episode_end()
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        timer.log_run_summary()

        if mj_viewer is not None:
            mj_viewer.close_display()

        if dataset is not None:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        if policy_source is not None and policy_source.is_connected:
            policy_source.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization("rerun")

        if dataset_cfg.push_to_hub and dataset is not None and dataset.num_episodes > 0:
            dataset.push_to_hub()

        # Perform all resource cleanup before optional speech. Blocking audio
        # must not delay shutdown after Ctrl+C or a closed MuJoCo window.
        log_say("Stop recording", cfg.play_sounds, blocking=False)
        log_say("Exiting", cfg.play_sounds, blocking=False)

    assert dataset is not None
    return dataset


def _reset_robot_and_action_source(robot: object, teleop: object | None, policy_source: object | None) -> None:
    """Reset the environment and clear any state held by the action source."""
    env = getattr(robot, "env", None)
    reset = getattr(env, "reset", None)
    if callable(reset):
        reset()
    else:
        logging.warning("Reset requested, but the active robot has no resettable environment.")

    if teleop is not None:
        teleop_reset = getattr(teleop, "reset", None)
        if callable(teleop_reset):
            teleop_reset()

    if policy_source is not None:
        policy_reset = getattr(policy_source, "on_episode_end", None)
        if callable(policy_reset):
            policy_reset()


def _policy_episode_succeeded(robot: object, success_delta: float) -> bool:
    """Return whether the final lift-cube position is within ``success_delta``."""
    env = getattr(robot, "env", None)
    cube_pose = getattr(env, "last_cube_pose_w", getattr(env, "cube_pose_w", None))
    goal_pose = getattr(env, "last_cube_goal_pose_w", getattr(env, "cube_goal_pose_w", None))
    if cube_pose is None or goal_pose is None:
        raise TypeError("Automatic policy success is only supported by MjGen3LiftCubeEnv.")
    return cube_goal_success(cube_pose, goal_pose, success_delta)
