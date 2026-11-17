"""Factory for KinovaLfD teleoperation devices (``DeviceBase`` subclasses).

Written by Will Solow, 2026.
"""

import argparse
from typing import Any

from kinova_lfd.teleop.devices.device_base import DeviceBase

TELEOP_DEVICES = ("keyboard", "vr_joystick", "vr_headset", "xbox")


def make_teleop_device(name: str, **cfg_overrides: Any) -> DeviceBase:
    """Construct the teleop device named ``name``.

    ``cfg_overrides`` are ``DeviceCfg`` fields (``pos_sensitivity``, ``rot_sensitivity``,
    ``reference_frame``, ``gripper_term``, ...). ``None`` values are dropped so the device's
    own config defaults apply. Device modules are imported lazily so that, e.g., the keyboard
    does not require ``evdev`` or the VR dependencies.
    """
    overrides = {key: value for key, value in cfg_overrides.items() if value is not None}
    name = name.lower()
    if name == "keyboard":
        from kinova_lfd.teleop.devices.keyboard import Se3Keyboard, Se3KeyboardCfg

        return Se3Keyboard(Se3KeyboardCfg(**overrides))
    if name == "xbox":
        from kinova_lfd.teleop.devices.xbox import XboxController, XboxControllerCfg

        return XboxController(XboxControllerCfg(**overrides))
    if name == "vr_joystick":
        from kinova_lfd.teleop.devices.vr import VRJoystick, VRJoystickCfg

        return VRJoystick(VRJoystickCfg(**overrides))
    if name == "vr_headset":
        from kinova_lfd.teleop.devices.vr import VRHeadset, VRHeadsetCfg

        return VRHeadset(VRHeadsetCfg(**overrides))
    raise ValueError(f"Unsupported teleop device {name!r}. Use one of {TELEOP_DEVICES}.")


def add_teleop_device_arg(parser: argparse.ArgumentParser, default: str = "keyboard") -> None:
    """Add ``--teleop_device`` / ``--teleop-device`` (aliases) to the argparser."""
    parser.add_argument(
        "--teleop_device",
        "--teleop-device",
        dest="teleop_device",
        type=str.lower,
        default=default,
        choices=TELEOP_DEVICES,
        help="Device for interacting with the environment.",
    )


def teleop_device_name(device: DeviceBase) -> str:
    """Return the factory name (``"keyboard"``, ``"xbox"``, ...) of a device instance."""
    from kinova_lfd.teleop.devices.keyboard import Se3Keyboard

    if isinstance(device, Se3Keyboard):
        return "keyboard"
    # Compare by class name for devices whose modules pull in optional dependencies.
    names = {"XboxController": "xbox", "VRJoystick": "vr_joystick", "VRHeadset": "vr_headset"}
    for cls in type(device).__mro__:
        if cls.__name__ in names:
            return names[cls.__name__]
    raise ValueError(f"Unknown teleop device type {type(device).__name__}.")
