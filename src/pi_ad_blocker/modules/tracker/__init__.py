"""Object tracking module for advertisement masking.

Provides lightweight tracking to maintain mask consistency across frames.
Reduces flicker by persisting ad track IDs between detections.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from pi_ad_blocker.config import TrackerConfig
from pi_ad_blocker.modules.detector import Detection


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A tracked object with temporal persistence."""

    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 0
    hit_count: int = 1
    missed_count: int = 0
    is_confirmed: bool = False
    polygon: Optional[np.ndarray] = None  # latest segmentation outline, if any

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def predict(self) -> None:
        """Simple Kalman-like position prediction (placeholder for full KF)."""
        self.age += 1

    def update(self, detection: Detection) -> None:
        """Update track with a new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.polygon = detection.polygon
        self.hit_count += 1
        self.missed_count = 0
        self.age += 1


# ---------------------------------------------------------------------------
# Abstract tracker
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """Abstract tracker interface."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker with new detections, return active tracks."""

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.is_confirmed]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# ByteTrack implementation
# ---------------------------------------------------------------------------

class ByteTrack(BaseTracker):
    """ByteTrack: Simple, effective multi-object tracking.

    Reference: https://github.com/ifzhang/ByteTrack
    Selected for Raspberry Pi due to:
    - No appearance features needed (vs DeepSORT)
    - Low compute overhead
    - Strong ID consistency with occlusion handling
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._last_frame_tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        # Split into high/low confidence detections
        high_conf = [
            d for d in detections
            if d.confidence >= self.config.track_high_thresh
        ]
        low_conf = [
            d for d in detections
            if self.config.track_low_thresh <= d.confidence < self.config.track_high_thresh
        ]

        # Predict existing tracks forward
        for track in self._tracks.values():
            track.predict()

        # First association: high confidence detections → existing tracks
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            high_conf, list(self._tracks.values())
        )

        # Second association: low confidence detections → unmatched tracks
        (
            second_matched,
            _,
            unmatched_tracks_final,
        ) = self._associate(low_conf, unmatched_tracks)

        # Update matched tracks
        for track_id, detection in matched_tracks:
            self._tracks[track_id].update(detection)

        for track_id, detection in second_matched:
            self._tracks[track_id].update(detection)

        # Create new tracks from unmatched high-confidence detections
        for det in unmatched_dets:
            track = Track(
                track_id=self._next_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                polygon=det.polygon,
            )
            track.is_confirmed = True
            self._tracks[self._next_id] = track
            self._next_id += 1

        # Remove stale tracks
        to_remove = []
        for tid, track in self._tracks.items():
            if track.missed_count > self.config.track_buffer:
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

        # Mark unmatched existing tracks as missed
        for track in unmatched_tracks_final:
            if track.track_id in self._tracks:
                self._tracks[track.track_id].missed_count += 1

        return self.active_tracks

    def _associate(
        self,
        detections: list[Detection],
        tracks: list[Track],
    ) -> tuple[list[tuple[int, Detection]], list[Detection], list[Track]]:
        """Match detections to tracks using IoU."""
        if not detections or not tracks:
            return [], detections, tracks

        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for ti, track in enumerate(tracks):
            for di, det in enumerate(detections):
                iou = self._iou(track.bbox, det.bbox)
                cost_matrix[ti, di] = 1.0 - iou

        # Hungarian algorithm
        matched_indices = self._linear_assignment(cost_matrix)

        matched: list[tuple[int, Detection]] = []
        unmatched_dets: set[int] = set(range(len(detections)))
        unmatched_tracks: set[int] = set(range(len(tracks)))

        for ti, di in matched_indices:
            iou = 1.0 - cost_matrix[ti, di]
            if iou < self.config.match_thresh:
                continue  # stays in unmatched set
            matched.append((tracks[ti].track_id, detections[di]))
            unmatched_dets.discard(di)
            unmatched_tracks.discard(ti)

        return (
            matched,
            [detections[i] for i in unmatched_dets],
            [tracks[i] for i in unmatched_tracks],
        )

    def _iou(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _linear_assignment(self, cost_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Linear assignment via Jonker-Volgenant (lapjv) or greedy fallback."""
        try:
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost_matrix)
            return list(zip(ri, ci))
        except ImportError:
            pass

        # Greedy fallback
        matched: list[tuple[int, int]] = []
        assigned_rows: set[int] = set()
        assigned_cols: set[int] = set()

        rows, cols = cost_matrix.shape
        for _ in range(min(rows, cols)):
            min_val = np.inf
            min_r, min_c = -1, -1
            for r in range(rows):
                if r in assigned_rows:
                    continue
                for c in range(cols):
                    if c in assigned_cols:
                        continue
                    if cost_matrix[r, c] < min_val:
                        min_val = cost_matrix[r, c]
                        min_r, min_c = r, c
            if min_r >= 0:
                matched.append((min_r, min_c))
                assigned_rows.add(min_r)
                assigned_cols.add(min_c)
        return matched


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class NoTracker(BaseTracker):
    """Pass-through tracker for ``tracker.type: "none"``.

    The docs promised a ~5% speedup for skipping tracking, but the config
    Literal rejected the value and the factory had no branch for it. This turns
    each detection into a confirmed track so the pipeline's single code path
    still works; ids are per-frame indices, which also switches off the
    masker's per-id EMA (that is what "no tracking" means).
    """

    def update(self, detections: list[Detection]) -> list[Track]:
        tracks = [
            Track(
                track_id=i,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                is_confirmed=True,
                polygon=det.polygon,
            )
            for i, det in enumerate(detections, start=1)
        ]
        self._tracks = {t.track_id: t for t in tracks}
        return tracks


def create_tracker(config: TrackerConfig) -> BaseTracker:
    """Create tracker by type."""
    if config.type == "bytetrack":
        return ByteTrack(config)
    elif config.type == "ocsort":
        from .ocsort import OCSort
        return OCSort(config)
    elif config.type == "botsort":
        from .botsort import BOTSort
        return BOTSort(config)
    elif config.type == "none":
        return NoTracker(config)
    else:
        raise ValueError(f"Unknown tracker: {config.type}")


__all__ = [
    "BaseTracker",
    "ByteTrack",
    "NoTracker",
    "Track",
    "create_tracker",
]