"""Tests for the real-time blocker: flow propagation, polygon helpers, miss tolerance."""
from __future__ import annotations

import numpy as np
import pytest

from pi_ad_blocker.config import (
    AppConfig,
    DetectorConfig,
    MaskingConfig,
    RealtimeConfig,
)
from pi_ad_blocker.modules.detector import Detection
from pi_ad_blocker.realtime import FlowPropagator, RealtimeAdBlocker, _points_in_polygon


def textured_frame(h=240, w=320) -> np.ndarray:
    """Deterministic noise texture so LK has features to track."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class StubDetector:
    """Returns a detection for the first call, then nothing."""

    def __init__(self, dets_first: int = 1):
        self.dets_first = dets_first
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls > self.dets_first:
            return []
        poly = np.array([[40, 40], [120, 40], [120, 120], [40, 120]], np.float32)
        return [Detection(0, "ad", 0.9, (40, 40, 120, 120), polygon=poly)]


def test_points_in_polygon_keeps_interior_only():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    pts = np.array([[5, 5], [50, 50], [9, 9], [-1, 5]], np.float32)
    inside = _points_in_polygon(pts, poly)
    assert len(inside) == 2
    assert all(0 <= p[0] <= 10 and 0 <= p[1] <= 10 for p in inside)


def test_points_in_polygon_handles_bad_input():
    poly = np.array([[0, 0], [1, 1]], np.float32)  # degenerate
    assert len(_points_in_polygon(np.array([[0, 0]], np.float32), poly)) == 0
    good = np.array([[0, 0], [10, 0], [10, 10]], np.float32)
    assert len(_points_in_polygon(np.empty((0, 2), np.float32), good)) == 0


def test_flow_propagator_follows_translation():
    """A pure image shift should move the propagated polygon by the same shift."""
    frame0 = textured_frame()
    dx, dy = 9, 6
    frame1 = np.roll(np.roll(frame0, dy, axis=0), dx, axis=1)

    poly = np.array([[60, 60], [160, 60], [160, 160], [60, 160]], np.float32)
    det = Detection(0, "ad", 0.9, (60, 60, 160, 160), polygon=poly)

    prop = FlowPropagator()
    prop.set_keyframe(frame0, [det])
    out = prop.propagate(frame1)
    assert len(out) == 1
    moved = np.asarray(out[0].polygon, np.float32)
    c0 = poly.mean(axis=0)
    c1 = moved.mean(axis=0)
    assert c1[0] - c0[0] == pytest.approx(dx, abs=3.0)
    assert c1[1] - c0[1] == pytest.approx(dy, abs=3.0)


def test_flow_propagator_returns_detection_without_keyframe():
    prop = FlowPropagator()
    assert prop.propagate(textured_frame()) == []


def _blocker(**rt_kwargs) -> RealtimeAdBlocker:
    cfg = AppConfig(
        detector=DetectorConfig(model_path="unused.onnx", model_type="yolo11"),
        masking=MaskingConfig(mode="black", use_polygon=True),
        realtime=RealtimeConfig(async_detection=False, detect_interval=1, **rt_kwargs),
    )
    blocker = RealtimeAdBlocker(cfg)
    blocker.detector = StubDetector(dets_first=1)
    return blocker


def _frames_until_clear(miss_tolerance: int, limit: int = 60):
    """Feed an empty detector; return the frame index where the outline drops."""
    blocker = _blocker(miss_tolerance=miss_tolerance, max_propagation_frames=60)
    frame = textured_frame()
    for i in range(limit):
        blocker.process_frame(frame, i)
        if not blocker._key_dets:
            return i
    return None


def test_miss_tolerance_delays_outline_drop():
    # The tracker already holds tracks for a few frames, so the outline does not
    # drop instantly; miss_tolerance must push it later by exactly that margin.
    strict = _frames_until_clear(0)
    tolerant = _frames_until_clear(2)
    assert strict is not None and tolerant is not None
    assert tolerant - strict == 2


# --- ONNX seg mask decode regression -----------------------------------------
# The original decode cropped prototype masks with original-frame coordinates,
# which produced empty crops (no masks at all) and later numpy shape crashes.
def test_onnx_seg_masks_land_inside_their_boxes():
    from pathlib import Path

    import cv2

    from pi_ad_blocker.config import DetectorConfig
    from pi_ad_blocker.modules.detector import create_detector

    model = Path("runs/seg/ads_nano_seg/weights/best.onnx")
    video = Path("data/outputs/test_ad_video.mp4")
    if not model.exists() or not video.exists():
        pytest.skip("trained ONNX seg model or sample video not available")

    det = create_detector(
        DetectorConfig(
            model_path=str(model),
            model_type="onnx_seg",
            confidence_threshold=0.3,
            input_width=640,
            input_height=640,
            filter_to_ad_classes=False,
        )
    )

    cap = cv2.VideoCapture(str(video))
    frame = None
    for _ in range(120):
        ok, frame = cap.read()
        if not ok:
            pytest.skip("video shorter than expected")
    cap.release()

    dets = det.detect(frame)
    assert dets, "ONNX seg detector returned nothing on a frame with billboards"
    for d in dets:
        assert d.polygon is not None and len(d.polygon) >= 3
        x1, y1, x2, y2 = d.bbox
        assert x2 > x1 and y2 > y1
        # every mask vertex must sit inside the detection box (within 2 px)
        assert d.polygon[:, 0].min() >= x1 - 2
        assert d.polygon[:, 0].max() <= x2 + 2
        assert d.polygon[:, 1].min() >= y1 - 2
        assert d.polygon[:, 1].max() <= y2 + 2
