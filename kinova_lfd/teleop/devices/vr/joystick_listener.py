import json
import threading
from typing import Any, Dict, Optional

import zmq


class VRJoystickListener:
    """
    Background ZMQ SUB listener that always keeps the latest decoded message.

    - Binds to tcp://host:port (publisher connects to us).
    - Expects each message to be a single JSON object.
    - If a topic prefix like "vr_data" is used, it is ignored.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.bind(f"tcp://{host}:{port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._sock.recv_string()
            except zmq.ZMQError:
                break
            if msg == "vr_data":
                continue

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            with self._lock:
                self._latest = data

    def read_latest(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent decoded message dict, or None if nothing seen yet.

        The message is returned as-is; you can access:
            state["headset"]
            state["left_controller"]["pose"]
            state["left_controller"]["inputs"]["joystick"]
            state["left_controller"]["inputs"]["trigger"]
            state["right_controller"]["inputs"]["grip"]
            ... etc.
        """
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Stop the background thread and close the ZMQ socket."""
        self._running = False
        try:
            self._sock.close(0)
        finally:
            self._ctx.term()
        self._thread.join(timeout=1.0)


def joystick_listener(host: str = "0.0.0.0", port: int = 5555) -> VRJoystickListener:
    """
    Convenience factory to match the requested API:

        j = joystick_listener(host, port)
        state = j.read_latest()
    """
    return VRJoystickListener(host=host, port=port)
