"""Shared helpers for the local perception model servers (SAM3, FoundationPose, HaMeR).

Each model runs as a long-lived FastAPI server that keeps its weights on the GPU. Clients and
servers share the filesystem, so requests carry paths (episode folder, output file) instead of
image data, and responses are small JSON summaries.

Written by Will Solow, 2026.
"""

import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import httpx


def create_model_app(title: str, info: Callable[[], dict[str, Any]]) -> tuple[Any, threading.Lock]:
    """Create a FastAPI app with ``GET /health`` and a lock that serializes GPU requests.

    Endpoints that touch the GPU should be plain ``def`` handlers (FastAPI runs them on a
    worker thread) and hold the returned lock for the duration of the request.
    """
    from fastapi import FastAPI

    app = FastAPI(title=title)
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "busy": lock.locked(), **info()}

    return app, lock


def run_locked(lock: threading.Lock, fn: Callable[[], dict[str, Any]], tag: str) -> dict[str, Any]:
    """Run ``fn`` under ``lock``, timing it and turning failures into an HTTP 500 with the traceback."""
    from fastapi import HTTPException

    with lock:
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"[{tag}] {type(exc).__name__}: {exc}") from exc
        result["seconds"] = round(time.perf_counter() - start, 2)
        return result


def serve(app: Any, host: str, port: int) -> None:
    """Run ``app`` with uvicorn (blocking)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


class ServerClient:
    """Minimal JSON client for a perception server."""

    def __init__(self, url: str, timeout_s: float = 3600.0, name: str = "server") -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = name

    def health(self) -> dict[str, Any]:
        """Return ``/health`` or raise a ``ConnectionError`` that says how to start the server."""
        try:
            response = httpx.get(f"{self.url}/health", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Could not reach the {self.name} server at {self.url} ({exc}). Start it first, see the module docstring."
            ) from exc

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{self.url}/{endpoint.lstrip('/')}", json=payload, timeout=self.timeout_s)
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{self.name} server error ({response.status_code}): {detail}")
        return response.json()
