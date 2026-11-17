r"""kortex_play_diffusion.py.

Teleoperation (keyboard, Xbox, ...) of the real Gen3 with an on-demand Diffusion Policy rollout.

The robot is built through the LeRobot ``Robot`` API (``make_robot``) with the dataset's
recording timing (``dt = 1 / fps``), so the policy sees the same observations it was trained on.
Press G (keyboard) / X (Xbox) to hand control to the policy, and again to take it back.

Run from the repository root::

    python -m scripts.imitation.kortex_play_diffusion \\
      --checkpoint outputs/train/<run>/checkpoints/<step>/pretrained_model

Written by Will Solow, 2026.
"""

import argparse

import numpy as np

from kinova_lfd.agents.lfd import DiffusionTeleopAdapter
from kinova_lfd.agents.lfd.util import (
    add_diffusion_args,
    build_realtime_controller,
    camera_size_overrides,
    load_diffusion_bundle,
    policy_camera_flags,
    require_twist_tcp,
    wait_for_camera,
)
from kinova_lfd.data_collection import make_robot, make_robot_config
from kinova_lfd.envs.base.viewers import KortexViewer
from kinova_lfd.teleop.factory import add_teleop_device_arg, make_teleop_device

parser = argparse.ArgumentParser(description="Teleop + Diffusion Policy on the real Kortex Gen3.")
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--pos-sensitivity", type=float, default=0.15, help="Teleop linear speed (m/s).")
parser.add_argument(
    "--rot-sensitivity", type=float, default=float(np.deg2rad(10.0)), help="Teleop angular speed (rad/s)."
)

add_diffusion_args(parser)
add_teleop_device_arg(parser, default="xbox")
args_cli = parser.parse_args()


def main() -> None:
    """Run teleop with an on-demand diffusion policy on the Kortex Gen3."""
    bundle = load_diffusion_bundle(args_cli)
    require_twist_tcp(bundle)
    use_wrist, use_tabletop = policy_camera_flags(bundle.image_keys)

    robot_cfg = make_robot_config(
        "kortex_tcp_gen3",
        robot_ip=args_cli.robot_ip,
        dt=bundle.dt,
        action_spec="TWIST_TCP",
        use_wrist_camera=use_wrist,
        use_tabletop_camera=use_tabletop,
        **camera_size_overrides(bundle.ds_features, "kortex"),
    )
    robot = make_robot(robot_cfg)
    robot.connect()
    wait_for_camera(robot, "wrist")

    controller = build_realtime_controller(bundle, args_cli, control_hz=1.0 / robot.env.step_dt)
    print(
        f"[INFO][LFD-TELEOP] checkpoint={bundle.pretrained_dir}\n"
        f"[INFO][LFD-TELEOP] dt={bundle.dt * 1000:g} ms "
        f"cameras={[k.removeprefix('observation.images.') for k in bundle.image_keys]} "
        f"chunk={controller.chunk_len} replan_after={controller.replan_after}"
    )

    teleop = make_teleop_device(
        args_cli.teleop_device,
        pos_sensitivity=args_cli.pos_sensitivity,
        rot_sensitivity=args_cli.rot_sensitivity,
    )
    policy = DiffusionTeleopAdapter(
        robot,
        teleop,
        controller=controller,
        preprocessor=bundle.preprocessor,
        ds_features=bundle.ds_features,
        max_policy_steps=args_cli.max_policy_steps,
        done_fn=None,
    )
    try:
        KortexViewer(
            robot.env,
            policy,
            view_tabletop=True,
            view_wrist=True,
        ).run()
    finally:
        controller.reset()


if __name__ == "__main__":
    main()
