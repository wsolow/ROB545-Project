"""kortex_bridge.py.

Helper code for loading the Kortex API

Copied from Kinova Kortex, 2026. Modified by Will Solow
"""

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport


class DeviceConnection:
    """Class for handling connection to device."""

    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def create_tcp_connection(ip: str = "192.168.1.10", username: str = "admin", password: str = "admin"):
        """Return RouterClient required to create services and send requests to device or sub-devices."""
        return DeviceConnection(ip, port=DeviceConnection.TCP_PORT, credentials=(username, password))

    @staticmethod
    def create_udp_connection(ip: str = "192.168.1.10", username: str = "admin", password: str = "admin"):
        """Return RouterClient that allows to create services and send requests to a device @ 1khz."""
        return DeviceConnection(ip, port=DeviceConnection.UDP_PORT, credentials=(username, password))

    def __init__(self, ip_address: str, port: int = TCP_PORT, credentials: tuple = ("", "")) -> None:
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        # Setup API
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    # Called when entering 'with' statement
    def __enter__(self) -> RouterClient:
        """Return a base client for the robot."""
        self.transport.connect(self.ip_address, self.port)

        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ip_address)
            self.sessionManager.CreateSession(session_info)

        return self.router

    # Called when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.sessionManager != None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()

    def open_base(self) -> BaseClient:
        """Open connection and return (base_client)."""
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000
            self.sessionManager = SessionManager(self.router)
            print("[INFO][KORTEX] Logging into Base as", self.credentials[0], "on device", self.ip_address)
            self.sessionManager.CreateSession(session_info)
        self.base = BaseClient(self.router)
        return self.base

    def open_base_and_cyclic(self) -> tuple[BaseClient, BaseCyclicClient]:
        """Open connection and return (base_client, base_cyclic_client)."""
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000
            self.sessionManager = SessionManager(self.router)
            print(
                "[INFO][KORTEX] Logging into Base and BaseCyclic as", self.credentials[0], "on device", self.ip_address
            )
            self.sessionManager.CreateSession(session_info)
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        return self.base, self.base_cyclic

    def open_cyclic(self) -> BaseCyclicClient:
        """Open connection and return (base_cyclic_client)."""
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000
            session_info.connection_inactivity_timeout = 2000
            self.sessionManager = SessionManager(self.router)
            print("[INFO][KORTEX] Logging into BaseCyclic as", self.credentials[0], "on device", self.ip_address)
            self.sessionManager.CreateSession(session_info)
        self.base_cyclic = BaseCyclicClient(self.router)
        return self.base_cyclic

    def close(self):
        self.__exit__(None, None, None)


def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications.

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)

    """

    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()

    return check


def setup_kortex_tcp_udp(
    ip: str = "192.168.1.10", username: str = "admin", password: str = "admin"
) -> tuple[DeviceConnection, DeviceConnection]:
    """Set up the kortex base client."""
    print("[INFO] Connecting to Kortex TCP/UDP. Ensure that device is plugged in.")
    return DeviceConnection.create_tcp_connection(
        ip=ip, username=username, password=password
    ), DeviceConnection.create_udp_connection(ip=ip, username=username, password=password)
