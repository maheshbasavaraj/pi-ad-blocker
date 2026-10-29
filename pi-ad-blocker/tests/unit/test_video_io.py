"""Unit tests for video I/O module."""

from __future__ import annotations

import numpy as np
import pytest

from pi_ad_blocker.config import VideoIOConfig
from pi_ad_blocker.modules.video_io import (
    Frame,
    VideoWriter,
    OpenCVReader,
    create_reader,
)


class TestFrame:
    def test_frame_creation(self):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data, frame_number=0, source="test.mp4", timestamp_ms=0.0)
        assert frame.height == 480
        assert frame.width == 640
        assert frame.shape == (480, 640, 3)
        assert frame.frame_number == 0

    def test_frame_increment(self):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data, frame_number=42, source="cam", timestamp_ms=123.4)
        assert frame.frame_number == 42
        assert frame.timestamp_ms == 123.4


class TestVideoWriter:
    def test_write_no_output(self):
        """Writer should be a no-op if no output path is set."""
        config = VideoIOConfig(output_path=None)
        writer = VideoWriter(config)
        writer.open((640, 480), 30.0)
        # Should not crash
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.close()

    def test_open_none_output(self):
        config = VideoIOConfig(output_path=None)
        writer = VideoWriter(config)
        # Should log and skip
        writer.open((640, 480), 30.0)
        assert writer._writer is None


class TestCreateReader:
    def test_default_opencv_reader(self):
        config = VideoIOConfig(input_source="test.mp4")
        reader = create_reader(config)
        assert isinstance(reader, OpenCVReader)

    def test_missing_file(self):
        config = VideoIOConfig(input_source="/nonexistent/video.mp4")
        reader = create_reader(config)
        with pytest.raises(Exception):
            reader.open()

    def test_picamera2_not_available(self):
        config = VideoIOConfig(input_source="0", camera_backend="picamera2")
        reader = create_reader(config)
        with pytest.raises(Exception):
            reader.open()