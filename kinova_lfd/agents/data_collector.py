"""data_collector.py.

Controls logging of data and messages within Skillet.
"""

import pickle
import threading
import time
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np

from kinova_lfd.envs.base.gym import GymEnv


class DataStatusCodes(IntEnum):
    """The codes for the status of data collection."""

    STOP = 0
    """When to not collect data."""
    COLLECT = 1
    """When to collect data."""
    SAVE = 2
    """When to save data."""
    SAVED = 3
    """Data is saved."""


class DataLogger:
    def __init__(self, log_dir: str, rate: int = 10):
        self._status = DataStatusCodes.STOP
        self._log_dir = log_dir
        self._rate = rate
        self._logging = False
        self._prev_logging = False
        self._save_log = False
        self._prev_save_log = False
        self._env = None
        self._kwarg_dict = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._num_points = 0
        self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        Path(self._log_dir).mkdir(exist_ok=True, parents=True)

    @property
    def log(self) -> bool:
        return self._logging

    @log.setter
    def log(self, value) -> None:
        self._prev_logging = self._logging
        self._logging = value

    @property
    def save_log(self) -> bool:
        return self._save_log

    @save_log.setter
    def save_log(self, value) -> None:
        self._prev_save_log = self._save_log
        self._save_log = value

    def run(
        self,
    ) -> None:
        """Run the data logging loop."""
        poll_period_s = 1.0 / self._rate
        next_poll_t = time.perf_counter()
        while not self._stop_event.is_set():
            self._update_state()
            if self._status == DataStatusCodes.COLLECT:
                if self._env is None or self._kwarg_dict is None:
                    return
                self.log_from_env(self._env, self._kwarg_dict)
            if self._status == DataStatusCodes.SAVE:
                self.save_log_from_env(self._kwarg_dict)
                self._status = DataStatusCodes.SAVED
            sleep_time = (time.perf_counter() - next_poll_t) - poll_period_s
            if sleep_time < 0:
                time.sleep(min(-sleep_time, poll_period_s))
            else:
                ...
                # print(f"[WARN][DATA] full loop overran by {sleep_time * 1000:.1f}ms")

        self.stop()

    def _update_state(self):
        if self._status == DataStatusCodes.COLLECT:
            if self._logging != self._prev_logging:
                self._status = DataStatusCodes.STOP
                print("[INFO][LOGGER] Stopping data collection. Press Y to continue and X to save.")
                self._prev_logging = self._logging
            if self._save_log != self._prev_save_log:
                print("[INFO][LOGGER] Saving data. Press Y to start new data collection.")
                self._status = DataStatusCodes.SAVE
                self._prev_save_log = self._save_log

        elif self._status == DataStatusCodes.STOP:
            if self._logging != self._prev_logging:
                print("[INFO][LOGGER] Starting data collection. Press Y to stop and X to save")
                self._status = DataStatusCodes.COLLECT
                self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._prev_logging = self._logging
            if self._save_log != self._prev_save_log:
                print("[INFO][LOGGER] Saving data. Press Y to start new data collection.")
                self._status = DataStatusCodes.SAVE
                self._prev_save_log = self._save_log

        elif self._status == DataStatusCodes.SAVE or self._status == DataStatusCodes.SAVED:
            if self._logging != self._prev_logging:
                print("[INFO][LOGGER] Starting data collection. Press Y to stop and X to save")
                self._status = DataStatusCodes.COLLECT
                self._start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._prev_logging = self._logging

    def start(self, env: GymEnv, kwarg_dict: dict[str, str]) -> None:
        self._env = env
        self._kwarg_dict = kwarg_dict

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="DataLoggingThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling loop to stop and wait for the worker thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def log_data(self, log_dir: str | None = None, save_log: bool = False, **kwargs) -> None:
        """Save data."""
        self._log_dir = log_dir if log_dir is not None else self._log_dir
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v = v.squeeze()

            if not hasattr(self, f"_{k}"):
                if isinstance(v, np.ndarray):
                    setattr(self, f"_{k}", v[None, ...])
                else:
                    raise ValueError
            else:
                if isinstance(v, np.ndarray):
                    new_v = np.concatenate((getattr(self, f"_{k}"), v[None, ...]))
                    setattr(self, f"_{k}", new_v)
                else:
                    raise ValueError
        if save_log:
            fpath = Path(f"{self._log_dir}/{self._start_time}")
            fpath.mkdir(exist_ok=True, parents=True)
            for k in kwargs:
                v = getattr(self, f"_{k}")
                if isinstance(v, np.ndarray):
                    np.save(f"{fpath}/_{k}.npy", v)
                if isinstance(v, list):
                    with Path(f"{fpath}/_{k}.pkl").open("wb") as f:
                        pickle.dump(v, f)

    def log_from_env(self, env: GymEnv, kwarg_dict: dict[str, str]) -> None:
        """Save data."""
        for k, v in kwarg_dict.items():
            if isinstance(v, np.ndarray):
                v = v.squeeze()

            if not hasattr(self, f"_{k}"):
                val = getattr(env, v)
                if isinstance(val, np.ndarray):
                    setattr(self, f"_{k}", val[None, ...])
                else:
                    raise ValueError
            else:
                val = getattr(env, v)
                if isinstance(val, np.ndarray):
                    new_v = np.concatenate((getattr(self, f"_{k}"), val[None, ...]))
                    setattr(self, f"_{k}", new_v)
                else:
                    raise ValueError

    def save_log_from_env(self, kwarg_dict: dict[str, str]):
        """Save the log from an environment."""
        fpath = Path(f"{self._log_dir}/{self._start_time}")
        fpath.mkdir(exist_ok=True, parents=True)
        for k in kwarg_dict:
            try:
                v = getattr(self, f"_{k}")
                if isinstance(v, np.ndarray):
                    np.save(f"{fpath}/_{k}.npy", v)
                    delattr(self, f"_{k}")
            except AttributeError as e:
                print(f"Missing attribute {e}.")
