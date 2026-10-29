"""Main advertisement blocking pipeline.

Wires together detector → tracker → masker → video I/O into
a coherent processing pipeline for both offline video and live camera.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from loguru import logger

from pi_ad_blocker.config import AppConfig, VideoIOConfig
from pi_ad_blocker.modules.detector import BaseDetector, Detection, create_detector
from pi_ad_blocker.modules.tracker import BaseTracker, Track, create_tracker
from pi_ad_blocker.modules.masking import BaseMasker, create_masker
from pi_ad_blocker.modules.video_io import (
    BaseReader,
    VideoWriter,
    Frame,
    create_reader,
    show_preview,
)
from pi_ad_blocker.modules.evaluation import Evaluator, EvalMetrics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AdBlockPipeline:
    """Main processing pipeline: read → detect → track → mask → write.

    Usage:
        config = AppConfig(...)
        pipeline = AdBlockPipeline(config)
        pipeline.run()
    """

    def __init__(
        self,
        config: AppConfig,
        custom_mask_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.detector: BaseDetector = create_detector(config.detector)
        self.tracker: BaseTracker = create_tracker(config.tracker)
        self.masker: BaseMasker = create_masker(config.masking, custom_mask_fn)
        self.reader: BaseReader = create_reader(config.video_io)
        self.writer: VideoWriter = VideoWriter(config.video_io)

        self._frame_count = 0
        self._total_frames = 0
        self._processing_times: list[float] = []

        # Callbacks
        self.on_frame_callback: Optional[Callable[[Frame, list[Detection], list[Track]], None]] = None

    def run(self) -> dict:
        """Run the pipeline end-to-end.

        Returns:
            Dict with processing statistics.
        """
        logger.info("Starting AdBlock Pipeline")
        logger.info("Detector: {} | Tracker: {} | Mask: {}",
                     type(self.detector).__name__,
                     type(self.tracker).__name__,
                     self.config.masking.mode)

        # Open video source
        self.reader.open()

        # Fix: use first frame to get dimensions for writer
        first_frame = self.reader.read()
        if first_frame is None:
            raise RuntimeError("No frames available from source")

        # Open writer
        frame_h, frame_w = first_frame.data.shape[:2]
        self.writer.open((frame_w, frame_h), self.reader.fps)

        # Process first frame
        self._process_frame(first_frame)

        # Process remaining frames
        for frame in self.reader:
            self._process_frame(frame)

        self.reader.close()
        self.writer.close()

        # Stats
        stats = self._get_stats()
        logger.info("Pipeline complete: {} frames in {:.1f}s ({:.1f} FPS)",
                     stats["frames_processed"],
                     stats["total_time_s"],
                     stats["avg_fps"])
        return stats

    def _process_frame(self, frame: Frame) -> None:
        """Process a single frame through the pipeline."""
        t0 = time.perf_counter()

        # 1. Detect advertisements
        detections = self.detector.detect(frame.data)

        # 2. Track objects across frames (optional)
        if self.config.tracker.type != "none":
            tracks = self.tracker.update(detections)
            # Use track bboxes for masking (more stable)
            mask_source: list[Detection] = []
            for track in tracks:
                mask_source.append(
                    Detection(
                        class_id=track.class_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        bbox=track.bbox,
                        track_id=track.track_id,
                    )
                )
        else:
            mask_source = detections

        # 3. Apply masks
        masked_frame = self.masker.apply(frame.data, mask_source)

        # 4. Write / preview
        self.writer.write(masked_frame)
        if self.config.video_io.enable_preview:
            continue_preview = show_preview(masked_frame, self.config.video_io.preview_window_name)
            if not continue_preview:
                logger.info("Preview closed by user")
                # Can't easily stop iteration from inside callback
                # User would need to press 'q' in the preview window

        # Callback
        if self.on_frame_callback:
            self.on_frame_callback(frame, detections,
                                    self.tracker.active_tracks if hasattr(self, 'tracker') else [])

        elapsed = time.perf_counter() - t0
        self._processing_times.append(elapsed)
        self._frame_count += 1

        if self._frame_count % 100 == 0:
            avg_fps = 1.0 / (sum(self._processing_times[-100:]) / 100)
            logger.info("Frame {} — {:.1f} FPS", self._frame_count, avg_fps)

        self._total_frames = self._frame_count

    def _get_stats(self) -> dict:
        total_time = sum(self._processing_times)
        avg_fps = self._total_frames / total_time if total_time > 0 else 0
        avg_lat = (sum(self._processing_times) / max(len(self._processing_times), 1)) * 1000
        return {
            "frames_processed": self._total_frames,
            "total_time_s": round(total_time, 3),
            "avg_fps": round(avg_fps, 2),
            "avg_latency_ms": round(avg_lat, 2),
        }

    def reset(self) -> None:
        """Reset pipeline state for a new run."""
        self.detector.reset_timing()
        self.tracker.reset()
        self._frame_count = 0
        self._processing_times.clear()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def process_video(
    config: AppConfig,
    input_path: str,
    output_path: str,
    show_preview: bool = True,
) -> dict:
    """Process a single video file with the ad blocking pipeline."""
    config.video_io.input_source = input_path
    config.video_io.output_path = output_path
    config.video_io.enable_preview = show_preview

    pipeline = AdBlockPipeline(config)
    return pipeline.run()


def process_camera(
    config: AppConfig,
    camera_id: int = 0,
    output_path: Optional[str] = None,
) -> dict:
    """Process live camera feed."""
    config.video_io.input_source = str(camera_id)
    config.video_io.output_path = output_path
    config.video_io.enable_preview = True

    pipeline = AdBlockPipeline(config)
    return pipeline.run()


__all__ = [
    "AdBlockPipeline",
    "process_video",
    "process_camera",
]