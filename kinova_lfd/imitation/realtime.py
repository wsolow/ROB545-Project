"""Realtime overlapping action-chunk control for DiffusionPolicy rollouts."""

import math
import threading
import time
from collections import deque
import torch
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor


def compute_chunk_len(*, control_hz: float, replan_hz: float, n_action_steps: int) -> int:
    """Open-loop chunk length so half a chunk spans one replan period."""
    if control_hz <= 0 or replan_hz <= 0:
        raise ValueError(f"control_hz and replan_hz must be > 0, got {control_hz}, {replan_hz}")
    raw = round(2.0 * control_hz / replan_hz)
    return max(1, min(int(n_action_steps), int(raw)))


class RealtimeDiffusionController:
    """Keep executing actions while the next diffusion chunk runs asynchronously.

    Observation queues on ``policy`` are updated every control step (same as
    ``select_action``). Action execution uses an external deque so inference does
    not block the control loop once the first chunk is ready.
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        control_hz: float = 50.0,
        replan_hz: float = 20.0,
        overlap: float = 0.5,
        verbose: bool = True,
    ) -> None:
        if not 0.0 < overlap <= 1.0:
            raise ValueError(f"overlap must be in (0, 1], got {overlap}")

        self.policy = policy
        self.control_hz = float(control_hz)
        self.replan_hz = float(replan_hz)
        self.overlap = float(overlap)
        self.verbose = verbose

        self.chunk_len = compute_chunk_len(
            control_hz=self.control_hz,
            replan_hz=self.replan_hz,
            n_action_steps=policy.config.n_action_steps,
        )
        self.replan_after = max(1, math.ceil(self.chunk_len * self.overlap))

        self._actions: deque[Tensor] = deque()
        self._executed_since_plan = 0
        self._last_action: Tensor | None = None
        self._lock = threading.Lock()
        self._infer_thread: threading.Thread | None = None
        self._infer_generation = 0
        self._pending: tuple[int, list[Tensor]] | None = None
        self._replan_count = 0
        self._warmup_done = False

    def reset(self) -> None:
        """Clear queues and cancel in-flight inference (join worker)."""
        self._join_worker()
        with self._lock:
            self._actions.clear()
            self._executed_since_plan = 0
            self._last_action = None
            self._pending = None
            self._warmup_done = False
        self.policy.reset()

    def step(self, batch: dict[str, Tensor]) -> Tensor:
        """Update obs history, maybe kick off a replan, return one action ``(B, action_dim)``."""
        self._ingest_observation(batch)
        self._drain_pending()

        if not self._warmup_done:
            self._run_inference_blocking()
            self._warmup_done = True
        elif self._should_replan():
            self._maybe_start_inference()

        action = self._pop_action()
        self._executed_since_plan += 1
        return action

    def _ingest_observation(self, batch: dict[str, Tensor]) -> None:
        batch = dict(batch)
        if ACTION in batch:
            batch.pop(ACTION)
        if self.policy.config.image_features:
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.policy.config.image_features],
                dim=-4,
            )
        # Do not feed the policy's unused ACTION queue; we own execution.
        self.policy._queues = populate_queues(self.policy._queues, batch, exclude_keys=[ACTION])

    def _should_replan(self) -> bool:
        with self._lock:
            busy = self._infer_thread is not None and self._infer_thread.is_alive()
            executed = self._executed_since_plan
        return (not busy) and executed >= self.replan_after

    def _snapshot_obs_batch(self) -> dict[str, Tensor]:
        keys = [OBS_STATE]
        if OBS_IMAGES in self.policy._queues:
            keys.append(OBS_IMAGES)
        if OBS_ENV_STATE in self.policy._queues:
            keys.append(OBS_ENV_STATE)
        return {key: torch.stack(list(self.policy._queues[key]), dim=1).clone() for key in keys}

    def _run_inference_blocking(self) -> None:
        obs = self._snapshot_obs_batch()
        chunk = self._predict_chunk(obs)
        with self._lock:
            self._actions.clear()
            self._actions.extend(chunk)
            self._executed_since_plan = 0
            self._replan_count += 1
        if self.verbose:
            print(
                f"[realtime] warmup chunk ready: len={len(chunk)} "
                f"replan_after={self.replan_after} chunk_len={self.chunk_len}"
            )

    def _maybe_start_inference(self) -> None:
        with self._lock:
            if self._infer_thread is not None and self._infer_thread.is_alive():
                return
            generation = self._infer_generation
            queue_len = len(self._actions)
            executed = self._executed_since_plan
        obs = self._snapshot_obs_batch()
        if self.verbose:
            print(
                f"[realtime] replan start #{self._replan_count + 1}: "
                f"executed_since_plan={executed}/{self.replan_after} "
                f"queue_len={queue_len}"
            )

        def worker() -> None:
            t0 = time.perf_counter()
            try:
                chunk = self._predict_chunk(obs)
            except Exception as exc:  # noqa: BLE001 — surface on main via pending miss
                if self.verbose:
                    print(f"[realtime] inference failed: {exc}")
                return
            dt = time.perf_counter() - t0
            with self._lock:
                if generation != self._infer_generation:
                    return
                self._pending = (generation, chunk)
            if self.verbose:
                print(f"[realtime] replan done in {dt:.3f}s (chunk_len={len(chunk)})")

        thread = threading.Thread(target=worker, name="diffusion-replan", daemon=True)
        with self._lock:
            self._infer_thread = thread
        thread.start()

    def _predict_chunk(self, obs_batch: dict[str, Tensor]) -> list[Tensor]:
        """Run diffusion and return ``chunk_len`` single-step action tensors ``(B, dim)``."""
        with torch.inference_mode():
            actions = self.policy.diffusion.generate_actions(obs_batch)
        # (B, n_action_steps, dim) -> list of (B, dim)
        actions = actions[:, : self.chunk_len]
        return [actions[:, i].contiguous() for i in range(actions.shape[1])]

    def _drain_pending(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
            thread = self._infer_thread
        if thread is not None and not thread.is_alive():
            thread.join(timeout=0.0)
            with self._lock:
                if self._infer_thread is thread:
                    self._infer_thread = None
        if pending is None:
            return
        generation, chunk = pending
        with self._lock:
            if generation != self._infer_generation:
                return
            # Replace unused open-loop tail with the new plan.
            self._actions.clear()
            self._actions.extend(chunk)
            self._executed_since_plan = 0
            self._replan_count += 1

    def _pop_action(self) -> Tensor:
        with self._lock:
            if self._actions:
                action = self._actions.popleft()
                self._last_action = action
                return action
            last = self._last_action
            busy = self._infer_thread is not None and self._infer_thread.is_alive()

        if last is not None:
            if self.verbose and busy:
                print("[realtime] action queue empty; holding last action while inference runs")
            return last

        # Should only happen if warmup failed; block for a chunk.
        self._run_inference_blocking()
        with self._lock:
            if not self._actions:
                raise RuntimeError("Diffusion warmup produced an empty action chunk.")
            action = self._actions.popleft()
            self._last_action = action
            return action

    def _join_worker(self) -> None:
        with self._lock:
            thread = self._infer_thread
            self._infer_generation += 1
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)
        with self._lock:
            self._infer_thread = None
            self._pending = None


def pace_control_loop(step_start: float, control_hz: float) -> None:
    """Sleep so the control loop tracks wall-clock ``control_hz``."""
    if control_hz <= 0:
        return
    target = 1.0 / control_hz
    remaining = target - (time.perf_counter() - step_start)
    if remaining > 0:
        time.sleep(remaining)
