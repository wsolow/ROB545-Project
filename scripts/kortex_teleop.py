"""ros2_teleop_vr.py.

Script to run a keyboard or joystick teleoperation with ROS2 manipulation environments.

Written by Will Solow, 2026.
"""

import argparse

from kinova_lfd.agents import DataLogger, LfDModerator
from kinova_lfd.envs.factory import create_env
from kinova_lfd.teleop.devices import (
    Se3Keyboard,
    Se3KeyboardCfg,
    VRHeadset,
    VRHeadsetCfg,
    VRJoystick,
    VRJoystickCfg,
    XboxController,
    XboxControllerCfg,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Kortex environments.")
parser.add_argument("--task", type=str, default="Kortex-UDP-Gen3-v0", help="Environment.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices={"keyboard", "vr_joystick", "vr_headset", "xbox"},
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--robot_ip", type=str, default="192.168.1.10", help="Robot IP.")
parser.add_argument("--log_dir", type=str, default="exp/tabletop_admittance/", help="Log location")

# parse the arguments
args_cli = parser.parse_args()


def main() -> None:
    """Run keyboard teleoperation with Kortex API.

    Defaults to Kortex-Gen3-v0 environment and passes the twist_tcp Action Specification.
    """
    env_cfg = {"robot_ip": args_cli.robot_ip, "use_tabletop_camera": False, "dt": 1 / 20, "dry_run": True}

    env = create_env(args_cli.task, env_cfg)

    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.15, rot_sensitivity=10))
    elif args_cli.teleop_device.lower() == "vr_joystick":
        teleop_interface = VRJoystick(VRJoystickCfg(pos_sensitivity=0.15, rot_sensitivity=10))
    elif args_cli.teleop_device.lower() == "vr_headset":
        teleop_interface = VRHeadset(VRHeadsetCfg(pos_sensitivity=0.15, rot_sensitivity=10))
    elif args_cli.teleop_device.lower() == "xbox":
        teleop_interface = XboxController(XboxControllerCfg(pos_sensitivity=0.12, rot_sensitivity=25))
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")
    logger = DataLogger(log_dir=args_cli.log_dir)

    env.reset()
    teleop_interface.reset()

    moderator = LfDModerator(env, teleop_interface, logger)

    moderator.run_teleop_loop()


if __name__ == "__main__":
    main()
