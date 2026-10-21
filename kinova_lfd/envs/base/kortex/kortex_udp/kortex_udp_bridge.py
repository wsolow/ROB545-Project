"""Launch the C++ Kortex bridge and speak to it as ``kortex`` / ``kortex_c``.

The proxies stand in for the Kortex Python ``BaseClient`` and ``BaseCyclicClient``.
They never open a Kortex session. The C++ process owns TCP session setup and the
1 kHz UDP cyclic socket.
"""

from __future__ import annotations

import os
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from kinova_lfd.envs.base.kortex.kortex_udp.ipc import (
    ARM_DOFS,
    TWIST_DOFS,
    CommandMode,
    ControlOp,
    FeedbackSample,
    pack_command,
    pack_control,
    unpack_feedback,
)

_CPP_DIR = Path(__file__).resolve().parent / "cpp"
_PKG_DIR = _CPP_DIR.parent
_DEFAULT_BINARY = _CPP_DIR / "build" / "kortex_udp_bridge"
_DEFAULT_URDF = _PKG_DIR / "assets" / "gen3_2f85.urdf"
_DEFAULT_SERVO = _PKG_DIR / "config" / "servo.yaml"


def default_bridge_binary() -> Path:
    """Return the bridge executable, honoring ``KORTEX_UDP_BRIDGE`` when set."""
    override = os.environ.get("KORTEX_UDP_BRIDGE")
    if override:
        return Path(override)
    return _DEFAULT_BINARY


class _Attr:
    """Attribute bag that mimics a Kortex protobuf message."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _cyclic_view(sample: FeedbackSample) -> _Attr:
    """Shape a sample the way ``KortexTcpEnv._update_robot_info`` reads feedback."""
    actuators = [
        _Attr(position=position, velocity=velocity, torque=torque)
        for position, velocity, torque in zip(
            sample.joint_position_deg,
            sample.joint_velocity_deg_s,
            sample.joint_torque_nm,
            strict=True,
        )
    ]
    motor = _Attr(
        position=sample.gripper_position,
        velocity=sample.gripper_velocity,
        current_motor=sample.gripper_current,
    )
    base = _Attr(
        tool_pose_x=sample.tool_xyz[0],
        tool_pose_y=sample.tool_xyz[1],
        tool_pose_z=sample.tool_xyz[2],
        tool_pose_theta_x=sample.tool_theta_deg[0],
        tool_pose_theta_y=sample.tool_theta_deg[1],
        tool_pose_theta_z=sample.tool_theta_deg[2],
        tool_twist_linear_x=sample.tool_twist_linear[0],
        tool_twist_linear_y=sample.tool_twist_linear[1],
        tool_twist_linear_z=sample.tool_twist_linear[2],
        tool_twist_angular_x=sample.tool_twist_angular[0],
        tool_twist_angular_y=sample.tool_twist_angular[1],
        tool_twist_angular_z=sample.tool_twist_angular[2],
        tool_external_wrench_force_x=sample.wrench_force[0],
        tool_external_wrench_force_y=sample.wrench_force[1],
        tool_external_wrench_force_z=sample.wrench_force[2],
        tool_external_wrench_torque_x=sample.wrench_torque[0],
        tool_external_wrench_torque_y=sample.wrench_torque[1],
        tool_external_wrench_torque_z=sample.wrench_torque[2],
    )
    interconnect = _Attr(gripper_feedback=_Attr(motor=[motor]))
    return _Attr(
        base=base,
        actuators=actuators,
        interconnect=interconnect,
        seq=sample.seq,
        command_seq=sample.command_seq,
        command_mode=sample.command_mode,
        flags=sample.flags,
    )


class KortexBaseProxy:
    """Stand-in for ``BaseClient``. Commands are forwarded to the C++ process."""

    def __init__(self, bridge: KortexUDPBridge) -> None:
        """Bind the proxy to a running bridge."""
        self._bridge = bridge

    def Stop(self) -> None:  # noqa: N802
        """Ask the bridge to leave low-level mode and exit."""
        self._bridge.close()

    def SetServoingMode(self, servoing_mode: int) -> None:  # noqa: N802
        """Forward a servoing-mode value.

        The bridge stays in low-level cyclic control for the life of the process.
        The packet is still delivered so the C++ side can record the request.

        """
        self._bridge.set_servoing_mode(servoing_mode)

    def send_joint_position(self, joints_rad: np.ndarray, gripper: float, duration_s: float = 1.0) -> None:
        """Send seven joint positions in radians, a gripper opening, and a trajectory duration."""
        joints = np.asarray(joints_rad, dtype=np.float64).reshape(-1)
        if joints.size != ARM_DOFS:
            raise ValueError(f"expected {ARM_DOFS} joint positions, got {joints.size}")
        self._bridge.send_setpoint(CommandMode.JOINT_POSITION, joints, gripper, np.zeros(TWIST_DOFS), duration_s)

    def send_twist(self, twist: np.ndarray, gripper: float) -> None:
        """Send a tool twist (m/s and deg/s) and a gripper opening in ``[0, 1]``."""
        twist_cmd = np.asarray(twist, dtype=np.float64).reshape(-1)
        if twist_cmd.size != TWIST_DOFS:
            raise ValueError(f"expected {TWIST_DOFS} twist values, got {twist_cmd.size}")
        self._bridge.send_setpoint(CommandMode.TWIST, np.zeros(ARM_DOFS), gripper, twist_cmd)


class KortexCyclicProxy:
    """Stand-in for ``BaseCyclicClient``.

    ``Refresh`` does not build a Kortex cyclic command. The C++ loop already sends
    that packet at 1 kHz. These methods return the newest feedback sample.
    """

    def __init__(self, bridge: KortexUDPBridge) -> None:
        """Bind the proxy to a running bridge."""
        self._bridge = bridge

    def RefreshFeedback(self) -> _Attr:  # noqa: N802
        """Return the latest 1 kHz feedback sample."""
        return self._bridge.latest_feedback()

    def Refresh(self, command: object | None = None) -> _Attr:  # noqa: N802
        """Return the latest feedback. ``command`` is ignored."""
        del command
        return self.RefreshFeedback()


class KortexUDPBridge:
    """Child process that owns the robot UDP socket."""

    def __init__(
        self,
        robot_ip: str,
        username: str = "admin",
        password: str = "admin",
        binary: Path | None = None,
        dry_run: bool = False,
        ready_timeout_s: float = 20.0,
    ) -> None:
        """Store connection settings. Call ``start`` before sending commands."""
        self.robot_ip = robot_ip
        self.username = username
        self.password = password
        self.binary = binary if binary is not None else default_bridge_binary()
        self.dry_run = dry_run
        self.ready_timeout_s = ready_timeout_s

        self._fb_sock: socket.socket | None = None
        self._cmd_sock: socket.socket | None = None
        self._cmd_addr: tuple[str, int] | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._seq = 0
        self._feedback: _Attr | None = None
        self._closed = False

        self.kortex = KortexBaseProxy(self)
        self.kortex_c = KortexCyclicProxy(self)

    def start(self) -> None:
        """Spawn the bridge and block until the first feedback sample arrives."""
        if self._proc is not None:
            return
        if not self.binary.is_file():
            raise FileNotFoundError(
                "Kortex UDP bridge is not built at "
                f"{self.binary}. From the repo root:\n"
                "  cmake -S kinova_lfd/envs/base/kortex/kortex_udp/cpp "
                "-B kinova_lfd/envs/base/kortex/kortex_udp/cpp/build "
                "-DKORTEX_DIR=/path/to/kortex_api\n"
                "  cmake --build kinova_lfd/envs/base/kortex/kortex_udp/cpp/build"
            )

        self._fb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._fb_sock.bind(("127.0.0.1", 0))
        self._fb_sock.setblocking(False)
        self._fb_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        feedback_port = int(self._fb_sock.getsockname()[1])

        command = [
            str(self.binary),
            "--ip",
            self.robot_ip,
            "--username",
            self.username,
            "--password",
            self.password,
            "--feedback-host",
            "127.0.0.1",
            "--feedback-port",
            str(feedback_port),
            "--watch-parent",
            "--urdf",
            str(_DEFAULT_URDF),
            "--servo-config",
            str(_DEFAULT_SERVO),
        ]
        if self.dry_run:
            command.append("--dry-run")

        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump_output, name="kortex-udp-bridge", daemon=True)
        self._reader.start()
        self._wait_until_ready()
        self._wait_for_feedback()
        print(f"[INFO][KORTEX_UDP] Bridge ready, commands -> 127.0.0.1:{self._cmd_addr[1]}")

    def close(self) -> None:
        """Stop the 1 kHz loop and wait for the process to exit."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            if proc is not None and proc.poll() is None and self._cmd_addr is not None:
                try:
                    self._seq += 1
                    packet = pack_control(self._seq, ControlOp.STOP, 0)
                    assert self._cmd_sock is not None
                    self._cmd_sock.sendto(packet, self._cmd_addr)
                except OSError:
                    pass
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if self._fb_sock is not None:
            self._fb_sock.close()
            self._fb_sock = None
        if self._cmd_sock is not None:
            self._cmd_sock.close()
            self._cmd_sock = None

    def set_servoing_mode(self, mode: int) -> None:
        """Deliver a servoing-mode request. The bridge does not leave low-level."""
        self._send_control(ControlOp.SET_SERVOING_MODE, mode)

    def send_setpoint(
        self,
        mode: CommandMode,
        joints_rad: np.ndarray,
        gripper: float,
        twist: np.ndarray,
        duration_s: float = 0.0,
    ) -> None:
        """Queue one setpoint. The C++ loop applies the latest packet at 1 kHz."""
        opening = float(np.clip(gripper, 0.0, 1.0))
        joints = tuple(float(value) for value in np.asarray(joints_rad, dtype=np.float64).reshape(-1))
        twist_cmd = tuple(float(value) for value in np.asarray(twist, dtype=np.float64).reshape(-1))
        with self._lock:
            self._ensure_open()
            self._seq += 1
            packet = pack_command(self._seq, mode, joints, opening, twist_cmd, duration_s)
            assert self._cmd_sock is not None
            assert self._cmd_addr is not None
            self._cmd_sock.sendto(packet, self._cmd_addr)

    def latest_feedback(self) -> _Attr:
        """Drain localhost feedback and return the newest sample."""
        self._drain_feedback()
        if self._feedback is None:
            raise RuntimeError("Kortex UDP bridge has not published feedback")
        return self._feedback

    def _send_control(self, op: ControlOp, value: int) -> None:
        with self._lock:
            self._ensure_open()
            self._seq += 1
            packet = pack_control(self._seq, op, value)
            assert self._cmd_sock is not None
            assert self._cmd_addr is not None
            self._cmd_sock.sendto(packet, self._cmd_addr)

    def _ensure_open(self) -> None:
        if self._closed or self._proc is None or self._cmd_addr is None:
            raise RuntimeError("Kortex UDP bridge is not running")
        if self._proc.poll() is not None:
            raise RuntimeError(f"Kortex UDP bridge exited with code {self._proc.returncode}")

    def _pump_output(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            print(line, end="")
            if line.startswith("KORTEX_UDP_READY"):
                port = int(line.split("cmd_port=", maxsplit=1)[1].strip())
                self._cmd_addr = ("127.0.0.1", port)
                self._ready.set()
        self._lines.put(None)
        self._ready.set()

    def _wait_until_ready(self) -> None:
        if not self._ready.wait(self.ready_timeout_s):
            self.close()
            raise TimeoutError(f"Kortex UDP bridge did not become ready within {self.ready_timeout_s:.0f}s")
        assert self._proc is not None
        if self._cmd_addr is None or self._proc.poll() is not None:
            code = self._proc.returncode
            self.close()
            raise RuntimeError(f"Kortex UDP bridge exited before the cyclic loop started (code {code})")
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _wait_for_feedback(self) -> None:
        deadline = time.monotonic() + 2.0
        while self._feedback is None:
            self._drain_feedback()
            if self._feedback is not None:
                return
            if time.monotonic() >= deadline:
                self.close()
                raise TimeoutError("Kortex UDP bridge did not publish feedback")
            time.sleep(0.002)

    def _drain_feedback(self) -> None:
        if self._fb_sock is None:
            return
        latest: FeedbackSample | None = None
        while True:
            try:
                payload, _addr = self._fb_sock.recvfrom(512)
            except BlockingIOError:
                break
            try:
                latest = unpack_feedback(payload)
            except ValueError:
                continue
        if latest is not None:
            self._feedback = _cyclic_view(latest)
