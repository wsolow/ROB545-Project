# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package providing interfaces to different teleoperation devices.

Currently, the following categories of devices are supported:

* **Keyboard**: Standard keyboard with WASD and arrow keys.
* **Spacemouse**: 3D mouse with 6 degrees of freedom.
* **Gamepad**: Gamepad with 2D two joysticks and buttons. Example: Xbox controller.

All device interfaces inherit from the :class:`DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`DeviceBase.advance` method is called. It also provides the function :meth:`DeviceBase.add_callback`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.
"""

from .device_base import DeviceBase as DeviceBase
from .device_base import DeviceCfg as DeviceCfg
from .keyboard import Se3Keyboard as Se3Keyboard
from .keyboard import Se3KeyboardCfg as Se3KeyboardCfg
from .vr import VRHeadset as VRHeadset
from .vr import VRHeadsetCfg as VRHeadsetCfg
from .vr import VRJoystick as VRJoystick
from .vr import VRJoystickCfg as VRJoystickCfg
from .xbox import XboxController as XboxController
from .xbox import XboxControllerCfg as XboxControllerCfg
