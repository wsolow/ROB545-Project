"""Simple multi-dimensional PID controller for twist-based pose servoing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDGains:
    """Per-axis or scalar PID gains."""

    kp: float | np.ndarray = 1.0
    ki: float | np.ndarray = 0.0
    kd: float | np.ndarray = 0.0


class PIDController:
    """Discrete PID that maps an error vector to a command vector."""

    def __init__(
        self,
        dim: int = 3,
        gains: PIDGains | None = None,
        *,
        kp: float | np.ndarray = 1.0,
        ki: float | np.ndarray = 0.0,
        kd: float | np.ndarray = 0.0,
        i_limit: float | None = 1.0,
        out_limit: float | None = None,
    ) -> None:
        self.dim = int(dim)
        if gains is not None:
            kp, ki, kd = gains.kp, gains.ki, gains.kd
        self.kp = self._as_gain(kp)
        self.ki = self._as_gain(ki)
        self.kd = self._as_gain(kd)
        self.i_limit = None if i_limit is None else i_limit
        self.out_limit = None if out_limit is None else out_limit
        self._integral = np.zeros(self.dim, dtype=np.float64)
        self._prev_error: np.ndarray | None = None

    def _as_gain(self, g: float | np.ndarray) -> np.ndarray:
        arr = np.asarray(g)
        if arr.ndim == 0:
            return np.full(self.dim, float(arr), dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"Gain length {arr.shape[0]} != dim {self.dim}")
        return arr

    def reset(self) -> None:
        """Clear integrator and derivative state."""
        self._integral[:] = 0.0
        self._prev_error = None

    def __call__(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Compute the PID output for ``error`` over timestep ``dt``."""
        dt = max(dt, 1e-6)

        self._integral = self._integral + error * dt
        if self.i_limit is not None:
            self._integral = np.clip(self._integral, -self.i_limit, self.i_limit)

        derivative = (
            np.zeros(self.dim, dtype=np.float64) if self._prev_error is None else (error - self._prev_error) / dt
        )
        self._prev_error = error.copy()

        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.out_limit is not None:
            u = np.clip(u, -self.out_limit, self.out_limit)
        return u
