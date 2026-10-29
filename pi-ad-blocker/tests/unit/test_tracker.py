"""Unit tests for tracker module."""

from __future__ import annotations

import pytest

from pi_ad_blocker.config import TrackerConfig
from pi_ad_blocker.modules.detector import Detection
from pi_ad_blocker.modules.tracker import (
    ByteTrack,
    Track,
    create_tracker,
)


class TestTrack:
    def test_track_creation(self):
        t = Track(track_id=1, class_id=79, class_name="billboard",
                  bbox=(0, 0, 100, 100), confidence=0.9)
        assert t.track_id == 1
        assert t.is_confirmed is False
        assert t.hit_count == 1
        assert t.missed_count == 0

    def test_center(self):
        t = Track(track_id=1, class_id=79, class_name="billboard",
                  bbox=(0, 0, 100, 100), confidence=0.9)
        cx, cy = t.center
        assert cx == 50.0
        assert cy == 50.0

    def test_update(self):
        t = Track(track_id=1, class_id=79, class_name="billboard",
                  bbox=(0, 0, 100, 100), confidence=0.9)
        det = Detection(79, "billboard", 0.95, (10, 10, 90, 90))
        t.update(det)
        assert t.bbox == (10, 10, 90, 90)
        assert t.confidence == 0.95
        assert t.hit_count == 2
        assert t.missed_count == 0

    def test_predict(self):
        t = Track(track_id=1, class_id=79, class_name="billboard",
                  bbox=(0, 0, 100, 100), confidence=0.9)
        t.age = 5
        t.predict()
        assert t.age == 6


class TestByteTrack:
    def test_empty_update(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        tracks = tracker.update([])
        assert tracks == []

    def test_new_detection_creates_track(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        dets = [Detection(79, "billboard", 0.9, (0, 0, 100, 100))]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].class_id == 79

    def test_low_confidence_discarded(self):
        config = TrackerConfig(track_high_thresh=0.5)
        tracker = ByteTrack(config)
        dets = [Detection(79, "billboard", 0.3, (0, 0, 100, 100))]
        tracks = tracker.update(dets)
        assert len(tracks) == 0

    def test_multiple_detections(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        dets = [
            Detection(79, "billboard", 0.9, (0, 0, 100, 100)),
            Detection(83, "sign", 0.85, (200, 200, 300, 300)),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_track_persistence(self):
        config = TrackerConfig(track_buffer=5)
        tracker = ByteTrack(config)

        # First frame
        dets = [Detection(79, "billboard", 0.9, (0, 0, 100, 100))]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        track_id = tracks[0].track_id

        # Second frame — same detection, slightly moved
        dets = [Detection(79, "billboard", 0.9, (5, 5, 105, 105))]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == track_id  # same ID

    def test_track_fade_out(self):
        config = TrackerConfig(track_buffer=2, track_high_thresh=0.5)
        tracker = ByteTrack(config)

        dets = [Detection(79, "billboard", 0.9, (0, 0, 100, 100))]
        tracker.update(dets)

        # No detections for track_buffer+2 frames (removal check happens before increment)
        for _ in range(4):
            tracker.update([])

        assert len(tracker.active_tracks) == 0

    def test_iou_matching(self):
        config = TrackerConfig(match_thresh=0.8)
        tracker = ByteTrack(config)
        dets = [Detection(79, "billboard", 0.9, (0, 0, 100, 100))]
        tracker.update(dets)

        # Different object, far away — should create new track
        dets2 = [Detection(79, "billboard", 0.9, (300, 300, 400, 400))]
        tracks = tracker.update(dets2)
        assert len(tracks) == 2  # old track still active + new track


class TestCreateTracker:
    def test_bytetrack_default(self):
        config = TrackerConfig()
        tracker = create_tracker(config)
        assert isinstance(tracker, ByteTrack)

    def test_unknown_tracker(self):
        config = TrackerConfig(type="ocsort")  # valid type
        tracker = create_tracker(config)
        from pi_ad_blocker.modules.tracker.ocsort import OCSort
        assert isinstance(tracker, OCSort)