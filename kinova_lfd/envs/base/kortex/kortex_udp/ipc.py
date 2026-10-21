"""Wire format shared with the C++ Kortex UDP bridge.

Packets are little-endian and packed. Sizes are asserted against cpp/ipc.hpp.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = 0x4B555044
ARM_DOFS = 7
TWIST_DOFS = 6

_COMMAND = struct.Struct("<IIII7ff6f")
_CONTROL = struct.Struct("<IIIII")
_FEEDBACK = struct.Struct("<IIII18f21f3fIII")

assert _COMMAND.size == 72
assert _CONTROL.size == 20
assert _FEEDBACK.size == 196


class PacketType(IntEnum):
    """Datagram kind on the localhost socket."""

    COMMAND = 1
    FEEDBACK = 2
    CONTROL = 3


class CommandMode(IntEnum):
    """Setpoint the 1 kHz loop should hold."""

    HOLD = 0
    JOINT_POSITION = 1
    TWIST = 2


class ControlOp(IntEnum):
    """Out-of-band request to the bridge process."""

    STOP = 1
    SET_SERVOING_MODE = 2


@dataclass
class FeedbackSample:
    """One cyclic sample, still in Kortex units."""

    seq: int
    actuator_count: int
    tool_xyz: tuple[float, float, float]
    tool_theta_deg: tuple[float, float, float]
    tool_twist_linear: tuple[float, float, float]
    tool_twist_angular: tuple[float, float, float]
    wrench_force: tuple[float, float, float]
    wrench_torque: tuple[float, float, float]
    joint_position_deg: tuple[float, ...]
    joint_velocity_deg_s: tuple[float, ...]
    joint_torque_nm: tuple[float, ...]
    gripper_position: float
    gripper_velocity: float
    gripper_current: float
    command_seq: int
    command_mode: int
    flags: int


def pack_command(
    seq: int,
    mode: CommandMode,
    joints_rad: tuple[float, ...],
    gripper: float,
    twist: tuple[float, ...],
) -> bytes:
    """Pack a joint or twist setpoint.

    Args:
        seq: Monotonic host sequence number.
        mode: Joint-position or twist.
        joints_rad: Seven joint positions in radians. Unused entries may be zero.
        gripper: Opening in ``[0, 1]``.
        twist: Six tool-frame values, linear m/s then angular deg/s.

    """
    if len(joints_rad) != ARM_DOFS:
        raise ValueError(f"expected {ARM_DOFS} joint positions, got {len(joints_rad)}")
    if len(twist) != TWIST_DOFS:
        raise ValueError(f"expected {TWIST_DOFS} twist values, got {len(twist)}")
    return _COMMAND.pack(
        MAGIC,
        int(PacketType.COMMAND),
        seq,
        int(mode),
        *joints_rad,
        float(gripper),
        *twist,
    )


def pack_control(seq: int, op: ControlOp, value: int = 0) -> bytes:
    """Pack a stop or servoing-mode request."""
    return _CONTROL.pack(MAGIC, int(PacketType.CONTROL), seq, int(op), value)


def unpack_feedback(payload: bytes) -> FeedbackSample:
    """Decode one feedback datagram."""
    if len(payload) != _FEEDBACK.size:
        raise ValueError(f"feedback datagram is {len(payload)} bytes, expected {_FEEDBACK.size}")
    values = _FEEDBACK.unpack(payload)
    magic = int(values[0])
    packet_type = int(values[1])
    if magic != MAGIC or packet_type != int(PacketType.FEEDBACK):
        raise ValueError("datagram is not a Kortex UDP feedback packet")

    def three(start: int) -> tuple[float, float, float]:
        return (float(values[start]), float(values[start + 1]), float(values[start + 2]))

    def seven(start: int) -> tuple[float, ...]:
        return tuple(float(values[i]) for i in range(start, start + ARM_DOFS))

    return FeedbackSample(
        seq=int(values[2]),
        actuator_count=int(values[3]),
        tool_xyz=three(4),
        tool_theta_deg=three(7),
        tool_twist_linear=three(10),
        tool_twist_angular=three(13),
        wrench_force=three(16),
        wrench_torque=three(19),
        joint_position_deg=seven(22),
        joint_velocity_deg_s=seven(29),
        joint_torque_nm=seven(36),
        gripper_position=float(values[43]),
        gripper_velocity=float(values[44]),
        gripper_current=float(values[45]),
        command_seq=int(values[46]),
        command_mode=int(values[47]),
        flags=int(values[48]),
    )
