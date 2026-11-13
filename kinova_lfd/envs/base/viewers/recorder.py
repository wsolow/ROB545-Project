"""recorder.py.

Shared OpenCV camera preview for the MuJoCo and Kortex viewers: image conversion,
side-by-side composition, status / REC overlays, a key-toggled video recorder
(:class:`PreviewRecorder`), and the threaded camera window itself (:class:`Preview`).

Written by Will Solow, 2026.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pynput import keyboard as pynput_keyboard


def as_preview_bgr(rgb: Any) -> np.ndarray:
    """Convert an RGB (H, W, 3) or (3, H, W) array/tensor to a contiguous BGR uint8 image."""
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    image = np.asarray(rgb).squeeze()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if float(np.nanmax(image)) <= 1.0 + 1e-3:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize ``image`` to ``height`` pixels tall, keeping its aspect ratio."""
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, round(w * height / h))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def compose_side_by_side(panes: list[np.ndarray], gap: int = 4) -> np.ndarray:
    """Place ``panes`` left to right, top-aligned, on a black canvas."""
    height = max(p.shape[0] for p in panes)
    width = sum(p.shape[1] for p in panes) + gap * (len(panes) - 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for pane in panes:
        canvas[: pane.shape[0], x : x + pane.shape[1]] = pane
        x += pane.shape[1] + gap
    return canvas


def draw_status(image: np.ndarray, text: str) -> None:
    """Draw ``text`` in the top-left corner with a dark backing box."""
    if not text:
        return
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (5, 5), (15 + w, 15 + h + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 10 + h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_rec_indicator(image: np.ndarray, elapsed_s: float) -> None:
    """Draw a red ``REC mm:ss`` marker in the top-right corner."""
    text = f"REC {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d}"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = image.shape[1] - w - 40
    cv2.rectangle(image, (x0 - 10, 5), (image.shape[1] - 5, 15 + h + baseline), (0, 0, 0), -1)
    cv2.circle(image, (x0 + 2, 10 + h // 2 + 2), 8, (0, 0, 255), -1)
    cv2.putText(image, text, (x0 + 18, 10 + h), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)


class PreviewRecorder:
    """Record the composed preview window to ``.mp4``, toggled by a global key (pynput).

    The key listener only latches toggle requests. :meth:`handle_toggle`,
    :meth:`process` and :meth:`stop` must all be called from the preview thread,
    which owns the writer.
    """

    def __init__(
        self,
        record_key: str | None = "v",
        video_dir: str | Path = "exp/ibvs_videos",
        fps: float = 20.0,
        prefix: str = "video",
        log_tag: str = "Viewer",
    ) -> None:
        self._record_key = record_key.lower() if record_key else None
        self._video_dir = Path(video_dir)
        self._fps = fps if fps > 0 else 20.0
        self._prefix = prefix
        self._tag = log_tag
        self._toggle = threading.Event()
        self._key_down = False
        self._listener: pynput_keyboard.Listener | None = None
        self._recording = False
        self._writer: cv2.VideoWriter | None = None
        self._writer_size: tuple[int, int] | None = None
        self._video_path: Path | None = None
        self._rec_start = 0.0
        self._rec_next_frame = 0.0
        self._rec_frames = 0

    @property
    def recording(self) -> bool:
        """Whether a recording is in progress."""
        return self._recording

    def start_listener(self) -> None:
        """Listen (globally, like the teleop keyboard) for the record key."""
        if self._record_key is None or self._listener is not None:
            return
        self._listener = pynput_keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self._listener.start()
        print(f"[INFO][{self._tag}] Press {self._record_key.upper()} to start/stop recording to {self._video_dir}/")

    def stop_listener(self) -> None:
        """Stop the key listener; safe to call repeatedly."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _key_char(key: Any) -> str | None:
        try:
            return key.char.lower()
        except AttributeError:
            return None

    def _on_key_press(self, key: Any) -> None:
        if self._key_char(key) != self._record_key or self._key_down:
            return  # ignore other keys and auto-repeat while held
        self._key_down = True
        self._toggle.set()

    def _on_key_release(self, key: Any) -> None:
        if self._key_char(key) == self._record_key:
            self._key_down = False

    def handle_toggle(self) -> None:
        """Apply a pending start/stop request."""
        if not self._toggle.is_set():
            return
        self._toggle.clear()
        if self._recording:
            self.stop()
        else:
            self._recording = True
            self._writer = None  # opened on the next frame, which fixes its size
            print(f"[INFO][{self._tag}] Recording started.")

    def process(self, canvas: np.ndarray) -> np.ndarray:
        """Record ``canvas`` if recording; return the frame to display (with REC marker)."""
        if not self._recording:
            return canvas
        self._record_frame(canvas)
        if self._writer is None:
            return canvas
        shown = canvas.copy()
        draw_rec_indicator(shown, time.perf_counter() - self._rec_start)
        return shown

    def _record_frame(self, frame: np.ndarray) -> None:
        """Write ``frame``, repeating it as needed so the video plays back in real time."""
        now = time.perf_counter()
        if self._writer is None:
            self._video_dir.mkdir(parents=True, exist_ok=True)
            self._video_path = self._video_dir / f"{self._prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            h, w = frame.shape[:2]
            self._writer_size = (w, h)
            self._writer = cv2.VideoWriter(str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
            if not self._writer.isOpened():
                self._writer = None
                self._recording = False
                return
            self._rec_start = now
            self._rec_next_frame = now
            self._rec_frames = 0

        if (frame.shape[1], frame.shape[0]) != self._writer_size:
            frame = cv2.resize(frame, self._writer_size, interpolation=cv2.INTER_AREA)
        period = 1.0 / self._fps
        while self._rec_next_frame <= now:
            self._writer.write(frame)
            self._rec_frames += 1
            self._rec_next_frame += period

    def stop(self) -> None:
        """Finish the current video, if any."""
        if not self._recording:
            return
        self._recording = False
        writer, self._writer = self._writer, None
        if writer is None:
            print(f"[INFO][{self._tag}] Recording stopped before any frame was captured; nothing saved.")
            return
        writer.release()
        duration = self._rec_frames / self._fps
        print(f"[INFO][{self._tag}] Saved video -> {self._video_path} ({duration:.1f}s, {self._rec_frames} frames)")


RgbdFrame = dict[str, Any] | None
FrameSource = Callable[[], tuple[RgbdFrame, RgbdFrame]]


def as_rgbd_frame(rgb: Any) -> RgbdFrame:
    """Wrap an already-captured RGB image (e.g. from a robot observation) as an ``RgbdFrame``."""
    return None if rgb is None else {"rgb": rgb}


class Preview:
    """One OpenCV window showing the wrist and tabletop cameras side by side, on its own thread.

    Frames reach the preview thread in one of two ways:

    - **pull**: pass ``frame_source`` (returns ``(wrist_rgbd, tabletop_rgbd)``); the preview
      thread calls it every frame. Use this when fetching frames can block (RealSense), so
      the control loop never waits on the camera.
    - **push**: leave ``frame_source`` as ``None`` and call :meth:`publish` from the control
      loop. Use this when frames must be read on the simulation thread (MuJoCo).

    Each pane is scaled to ``height``, gets the latest status text, and the wrist pane is
    passed through ``annotate_wrist(bgr) -> bgr`` when given. If ``record_key`` is set, a
    :class:`PreviewRecorder` records the composed window (overlays included).

    Closing the window stops the preview and sets :attr:`closed`; the owning viewer decides
    whether that ends its run.
    """

    def __init__(
        self,
        window_name: str,
        view_wrist: bool = True,
        view_tabletop: bool = True,
        fps: float = 20.0,
        height: int = 360,
        frame_source: FrameSource | None = None,
        annotate_wrist: Callable[[np.ndarray], np.ndarray] | None = None,
        record_key: str | None = "v",
        video_dir: str | Path = "exp/ibvs_videos",
        video_prefix: str = "video",
        log_tag: str = "Viewer",
    ) -> None:
        self._window = window_name
        self._view_wrist = view_wrist
        self._view_tabletop = view_tabletop
        self._fps = fps
        self._height = height
        self._frame_source = frame_source
        self._annotate_wrist = annotate_wrist
        self._tag = log_tag

        self._recorder: PreviewRecorder | None = None
        if record_key and self.enabled:
            self._recorder = PreviewRecorder(record_key, video_dir, fps=fps, prefix=video_prefix, log_tag=log_tag)

        self._lock = threading.Lock()
        self._wrist: RgbdFrame = None
        self._tabletop: RgbdFrame = None
        self._status = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        """Whether any camera pane is shown."""
        return self._view_wrist or self._view_tabletop

    @property
    def view_wrist(self) -> bool:
        return self._view_wrist

    @property
    def view_tabletop(self) -> bool:
        return self._view_tabletop

    @property
    def closed(self) -> bool:
        """Whether the user closed the preview window."""
        return self._closed

    @property
    def recording(self) -> bool:
        """Whether a recording is in progress."""
        return self._recorder is not None and self._recorder.recording

    def start(self) -> None:
        """Start the record-key listener and the preview thread (no-op if disabled or running)."""
        if not self.enabled or self._thread is not None:
            return
        if self._recorder is not None:
            self._recorder.start_listener()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"{self._tag}-camera-preview", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the listener and the preview thread (finishing any recording). Safe to call repeatedly."""
        if self._recorder is not None:
            self._recorder.stop_listener()
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                return
        self._thread = None
        with self._lock:
            self._wrist = None
            self._tabletop = None

    def publish(self, wrist: RgbdFrame = None, tabletop: RgbdFrame = None, status: str | None = None) -> None:
        """Hand the latest frames (push mode) and/or status text to the preview thread."""
        with self._lock:
            if wrist is not None:
                self._wrist = wrist
            if tabletop is not None:
                self._tabletop = tabletop
            if status is not None:
                self._status = status

    def set_status(self, text: str) -> None:
        """Set the status text drawn on every pane."""
        self.publish(status=text)

    def _latest_frames(self) -> tuple[RgbdFrame, RgbdFrame, str]:
        with self._lock:
            wrist, tabletop, status = self._wrist, self._tabletop, self._status
        if self._frame_source is not None:
            wrist, tabletop = self._frame_source()
        return wrist, tabletop, status

    def _pane(self, rgbd: RgbdFrame, annotate: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray | None:
        if rgbd is None or rgbd.get("rgb") is None:
            return None
        image = as_preview_bgr(rgbd["rgb"])
        if annotate is not None:
            image = annotate(image)
        return resize_to_height(image, self._height)

    def _window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _loop(self) -> None:
        shown = False
        try:
            cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            period = 1.0 / self._fps if self._fps > 0 else 0.0
            while not self._stop.is_set():
                try:
                    if self._recorder is not None:
                        self._recorder.handle_toggle()
                    wrist, tabletop, status = self._latest_frames()
                    panes = []
                    if self._view_wrist and (pane := self._pane(wrist, self._annotate_wrist)) is not None:
                        panes.append(pane)
                    if self._view_tabletop and (pane := self._pane(tabletop)) is not None:
                        panes.append(pane)
                    if panes:
                        for pane in panes:
                            draw_status(pane, status)
                        canvas = compose_side_by_side(panes)
                        if self._recorder is not None:
                            canvas = self._recorder.process(canvas)
                        cv2.imshow(self._window, canvas)
                        shown = True
                    cv2.waitKey(1)
                    if shown and self._window_closed():
                        print(f"[INFO][{self._tag}] Preview window closed.")
                        self._closed = True
                        break
                except cv2.error as exc:
                    break
                if period > 0:
                    self._stop.wait(period)
        except Exception:
            pass
        finally:
            if self._recorder is not None:
                self._recorder.stop()
            try:
                cv2.destroyWindow(self._window)
            except cv2.error:
                pass
