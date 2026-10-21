"""kortex_env.py.

Main Kortex

Written by Will Solow, 2026

"""

from dataclasses import dataclass
from typing import Any

from kortex_api.autogen.messages import Base_pb2

from kinova_lfd.envs.base.kortex.kortex_env import KortexEnv, KortexEnvCfg
from kinova_lfd.envs.base.kortex.kortex_tcp.kortex_tcp_bridge import DeviceConnection, setup_kortex_tcp_udp


@dataclass
class KortexTcpEnvCfg(KortexEnvCfg):
    """The configuration class for Kortex Envs."""

    robot_ip: str = "www.xxx.yyy.zzz"
    """IP of the robot"""

    use_tabletop_camera: bool = False

    use_wrist_camera: bool = True

    dt: float = 1.0 / 30.0

    decimation: float = 1.0


class KortexTcpEnv(KortexEnv):
    """Kortex workflow for TCP control."""

    def __init__(
        self,
        cfg: KortexTcpEnvCfg,
        kortex_tcp_udp: tuple[DeviceConnection, DeviceConnection] | None = None,
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
        self.kortex_tcp_udp = setup_kortex_tcp_udp(ip=cfg.robot_ip)
        self.kortex = self.kortex_tcp_udp[0].open_base()
        self.kortex_c = self.kortex_tcp_udp[1].open_cyclic()
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def __del__(self) -> None:
        """Cleanup for the environment."""
        if self.kortex_connection is not None:
            self.kortex_connection.close()
        self.close()

    def close(self) -> None:
        """Cleanup for the environment."""
        self.kortex.Stop()

    """
    Helper functions.
    """

    """
    Helper functions to communicate with Kortex
    """

    def switch_controllers(self, active_controller: int) -> bool:
        """Switch Kortex controllers.

        Args:
            active_controller: Controller (Enum) to activate

        Returns:
            Bool if the controller switch was successful

        """
        print(f"[INFO][KORTEX] Switching to controller: {active_controller}")
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = active_controller
        self.kortex.SetServoingMode(base_servo_mode)
        self.active_controller = active_controller
