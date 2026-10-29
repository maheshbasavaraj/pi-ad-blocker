"""OC-SORT (Observation-Centric SORT) tracker implementation.

Reference: https://github.com/noahcao/OC_SORT
More robust than ByteTrack under occlusion and non-linear motion
but slightly higher compute cost.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

from pi_ad_blocker.config import TrackerConfig
from pi_ad_blocker.modules.detector import Detection
from . import BaseTracker, Track


class OCSort(BaseTracker):
    """OC-SORT tracker — observation-centric for occluded scenarios."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._kalman_filters: dict[int, Any] = {}
        logger.debug("OC-SORT tracker initialized")

    def update(self, detections: list[Detection]) -> list[Track]:
        from scipy.optimize import linear_sum_assignment

        # Predict all tracks
        for track in self._tracks.values():
            self._predict_kalman(track.track_id)

        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # Update matched
        for track_id, det in matched:
            self._tracks[track_id].update(det)
            self._update_kalman(track_id, det.bbox)

        # New tracks
        for det in unmatched_dets:
            if det.confidence >= self.config.track_high_thresh:
                track = Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    is_confirmed=True,
                )
                self._tracks[self._next_id] = track
                self._kalman_filters[self._next_id] = self._init_kalman(det.bbox)
                self._next_id += 1

        # Remove stale
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t.missed_count > self.config.track_buffer
        ]
        for tid in to_remove:
            del self._tracks[tid]
            self._kalman_filters.pop(tid, None)

        # Mark unmatched as missed
        for track in unmatched_tracks:
            if track.track_id in self._tracks:
                self._tracks[track.track_id].missed_count += 1

        return self.active_tracks

    def _init_kalman(self, bbox: tuple[int, int, int, int]) -> Any:
        import cv2
        kf = cv2.KalmanFilter(7, 4)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0]], dtype=np.float32)
        kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        return kf

    def _predict_kalman(self, track_id: int) -> None:
        kf = self._kalman_filters.get(track_id)
        if kf is not None:
            kf.predict()

    def _update_kalman(self, track_id: int, bbox: tuple[int, int, int, int]) -> None:
        kf = self._kalman_filters.get(track_id)
        if kf is None:
            return
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        kf.correct(meas)

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, Detection]], list[Detection], list[Track]]:
        from scipy.optimize import linear_sum_assignment

        active = self.active_tracks
        if not detections or not active:
            return [], detections, active

        cost = np.zeros((len(active), len(detections)), dtype=np.float32)
        for ti, track in enumerate(active):
            for di, det in enumerate(detections):
                iou = self._iou(track.bbox, det.bbox)
                cost[ti, di] = 1.0 - iou

        ti_idx, di_idx = linear_sum_assignment(cost)

        matched: list[tuple[int, Detection]] = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(active)))

        for t, d in zip(ti_idx, di_idx):
            iou = 1.0 - cost[t, d]
            if iou >= self.config.match_thresh:
                matched.append((active[t].track_id, detections[d]))
                unmatched_dets.discard(d)
                unmatched_tracks.discard(t)

        return (
            matched,
            [detections[i] for i in unmatched_dets],
            [active[i] for i in unmatched_tracks],
        )

    def _iou(self, b1: tuple[int, ...], b2: tuple[int, ...]) -> float:
        x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0