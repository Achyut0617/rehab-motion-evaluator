"""Phase 2: Threaded video capture supporting Android IP Camera, USB webcam, and Demo mode."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class VideoStream:
    """Thread-safe video stream reader with auto-reconnect for IP camera streams."""

    def __init__(
        self,
        source: Union[int, str] = "demo",
        width: int = 640,
        height: int = 480,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.source: Union[int, str] = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._last_error: str = ""
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_ts: float = time.time()

    @property
    def is_demo(self) -> bool:
        return isinstance(self.source, str) and self.source.lower() == "demo"

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def last_error(self) -> str:
        return self._last_error

    def start(self) -> "VideoStream":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def switch_source(self, new_source: Union[int, str]) -> None:
        """Switch camera source at runtime (e.g., from demo to Android IP Camera URL)."""
        self.stop()
        self.source = new_source
        self._last_error = ""
        self._connected = False
        self.start()

    def _open_capture(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, str) and src.startswith(("http://", "https://", "rtsp://")):
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            self._last_error = f"Unable to open camera source: {self.source}"
            self._connected = False
            cap.release()
            return False

        # Minimize internal OpenCV buffer latency for live IP streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._cap = cap
        self._connected = True
        self._last_error = ""
        return True

    def _generate_demo_frame(self) -> np.ndarray:
        """Create a clean dark clinical backdrop for synthetic demo mode."""
        frame = np.full((self.height, self.width, 3), (24, 28, 36), dtype=np.uint8)
        # Subtle grid lines
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (34, 40, 52), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (34, 40, 52), 1)
        cv2.putText(
            frame,
            "DEMO MODE - Connect Android IP Camera URL in Dashboard",
            (16, self.height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _record_frame(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        with self._lock:
            self._frame = frame
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_ts
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_ts = now

    def _update_loop(self) -> None:
        while self._running:
            if self.is_demo:
                self._connected = True
                self._last_error = ""
                self._record_frame(self._generate_demo_frame())
                time.sleep(1.0 / 30.0)
                continue

            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._connected = False
                self._last_error = f"Lost stream from {self.source}; reconnecting..."
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            self._connected = True
            self._last_error = ""
            self._record_frame(frame)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
