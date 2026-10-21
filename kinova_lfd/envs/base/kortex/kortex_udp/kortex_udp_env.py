"""kortex_udp_env.py.

Kortex environment that steps at 30 Hz and lets a C++ process servo the Gen3 at 1 kHz.

``self.kortex`` and ``self.kortex_c`` talk to that process. They do not import or
call the Kortex Python API. Joint-position and twist setpoints are forwarded as
the latest command. The C++ twist controller turns tool twists into joint
positions at 1 kHz.

Written by Will Solow, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinova_lfd.envs.base.kortex.kortex_env import KortexEnv, KortexEnvCfg
from kinova_lfd.envs.base.kortex.kortex_udp import KortexBaseProxy, KortexCyclicProxy, KortexUDPBridge


@dataclass
class KortexUDPEnvCfg(KortexEnvCfg):
    """Configuration for the UDP Kortex environment."""

    robot_ip: str = "192.168.1.10"
    """IP of the robot."""

    username: str = "admin"
    """Kortex session username, forwarded to the C++ bridge."""

    password: str = "admin"
    """Kortex session password, forwarded to the C++ bridge."""

    use_tabletop_camera: bool = False

    use_wrist_camera: bool = False

    dt: float = 1.0 / 30.0
    """Host step period. Setpoints are published at this rate."""

    decimation: float = 1.0

    episode_length_s: float = 1e9

    bridge_path: str | None = None
    """Optional path to the ``kortex_udp_bridge`` executable."""

    dry_run: bool = False
    """Launch the bridge without opening the robot UDP socket."""


class KortexUDPEnv(KortexEnv):
    """Kortex workflow whose realtime UDP session lives in a C++ child process."""

    def __init__(
        self,
        cfg: KortexUDPEnvCfg,
        render_mode: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
            kwargs: Additoinal arguments

        """
        binary = Path(cfg.bridge_path) if cfg.bridge_path else None
        self._bridge = KortexUDPBridge(
            robot_ip=cfg.robot_ip,
            username=cfg.username,
            password=cfg.password,
            binary=binary,
            dry_run=cfg.dry_run,
        )
        self._bridge.start()
        self.kortex: KortexBaseProxy = self._bridge.kortex
        self.kortex_c: KortexCyclicProxy = self._bridge.kortex_c
        self._closed = False

        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def __del__(self) -> None:
        """Cleanup for the environment."""
        self.close()

    def close(self) -> None:
        """Stop the arm loop and terminate the C++ bridge."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            bridge.close()

    def switch_controllers(self, active_controller: int) -> None:
        """Record a servoing-mode request.

        The bridge process enters low-level mode before it accepts setpoints and
        does not switch away from it while the cyclic loop is running.

        Args:
            active_controller: Kortex servoing-mode value.

        """
        print(f"[INFO][KORTEX_UDP] Servoing request {active_controller} forwarded to the bridge")
        self.kortex.SetServoingMode(active_controller)
        self.active_controller = active_controller
