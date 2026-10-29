"""Video I/O module — handles video reading, writing, camera capture, and preview.

Supports OpenCV and Picamera2 backends for Raspberry Pi camera module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import VideoIOConfig


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VideoIOError(Exception):
    """Raised on video I/O failures."""


# ---------------------------------------------------------------------------
# Frame wrapper
# ---------------------------------------------------------------------------

class Frame:
    """Container for a single video frame with metadata."""

    def __init__(
        self,
        data: np.ndarray,
        frame_number: int,
        source: str,
        timestamp_ms: float,
    ) -> None:
        self.data = data
        self.frame_number = frame_number
        self.source = source
        self.timestamp_ms = timestamp_ms

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape


# ---------------------------------------------------------------------------
# Reader interface
# ---------------------------------------------------------------------------

class BaseReader(ABC):
    """Abstract video frame reader."""

    def __init__(self, config: VideoIOConfig) -> None:
        self.config = config
        self._frame_count = 0

    @abstractmethod
    def open(self) -> None:
        """Open the video source."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read the next frame, or None at end of stream."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Native frame rate of source."""

    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Total frame count (0 for live cameras)."""

    def __iter__(self) -> Generator[Frame, None, None]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
            self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count


class OpenCVReader(BaseReader):
    """OpenCV-based video reader. Works on all platforms."""

    def __init__(self, config: VideoIOConfig) -> None:
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_str = str(config.input_source)
        self._is_camera = False

    def open(self) -> None:
        src = self.config.input_source
        # Detect camera index or file path
        if src.isdigit() or src in ("0", "1", "2"):
            cam_idx = int(src)
            self._cap = cv2.VideoCapture(cam_idx)
            self._is_camera = True
            logger.info("Opening camera {}", cam_idx)
        else:
            path = Path(src)
            if not path.exists():
                raise VideoIOError(f"Video file not found: {path}")
            self._cap = cv2.VideoCapture(str(path))
            logger.info("Opening video file {}", path)

        if self._cap is None or not self._cap.isOpened():
            raise VideoIOError(f"Failed to open source: {src}")

        # Apply resize if configured
        if self.config.resize_width and self.config.resize_height:
            # We'll resize frames in read()
            pass

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise VideoIOError("Reader not opened")

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Apply resize if configured
        if self.config.resize_width and self.config.resize_height:
            frame = cv2.resize(
                frame,
                (self.config.resize_width, self.config.resize_height),
                interpolation=cv2.INTER_LINEAR,
            )

        return Frame(
            data=frame,
            frame_number=self._frame_count,
            source=self._source_str,
            timestamp_ms=self._cap.get(cv2.CAP_PROP_POS_MSEC),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        if self._cap is None or self._is_camera:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))


class Picamera2Reader(BaseReader):
    """Picamera2-based reader for Raspberry Pi camera module."""

    def __init__(self, config: VideoIOConfig) -> None:
        super().__init__(config)
        self._camera: Optional[object] = None

    def open(self) -> None:
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError:
            raise VideoIOError(
                "picamera2 not installed. Install with: pip install pi-ad-blocker[webcam]"
            )

        self._camera = Picamera2()
        # Configure for 480p for performance
        config = self._camera.create_video_configuration(
            main={"size": (640, 480)},
        )
        self._camera.configure(config)
        self._camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        self._camera.start()
        logger.info("Picamera2 started (640x480)")

    def read(self) -> Optional[Frame]:
        if self._camera is None:
            raise VideoIOError("Camera not opened")

        frame = self._camera.capture_array()
        if frame is None:
            return None

        # Picamera2 returns RGB, convert to BGR for OpenCV pipeline
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return Frame(
            data=frame,
            frame_number=self._frame_count,
            source="picamera2",
            timestamp_ms=0.0,
        )

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    @property
    def fps(self) -> float:
        return 30.0  # Picamera2 default

    @property
    def total_frames(self) -> int:
        return 0  # Live camera


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class VideoWriter:
    """Write processed frames to a video file."""

    def __init__(self, config: VideoIOConfig) -> None:
        self.config = config
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self, frame_size: tuple[int, int], fps: float) -> None:
        output_path = self.config.output_path
        if output_path is None:
            logger.info("No output path configured — skipping video write")
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.config.output_fourcc)
        out_fps = self.config.output_fps if self.config.output_fps else fps

        self._writer = cv2.VideoWriter(str(path), fourcc, out_fps, frame_size)
        if not self._writer.isOpened():
            raise VideoIOError(f"Failed to open video writer: {path}")

        logger.info("Writing output to {} at {} FPS", path, out_fps)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def show_preview(frame: np.ndarray, window_name: str = "Pi Ad Blocker") -> bool:
    """Display a preview window. Returns False if user pressed 'q' or ESC."""
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        cv2.destroyWindow(window_name)
        return False
    return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_reader(config: VideoIOConfig) -> BaseReader:
    """Create video reader based on configuration."""
    if config.camera_backend == "picamera2":
        return Picamera2Reader(config)
    return OpenCVReader(config)


__all__ = [
    "BaseReader",
    "OpenCVReader",
    "Picamera2Reader",
    "VideoWriter",
    "Frame",
    "VideoIOError",
    "create_reader",
    "show_preview",
]