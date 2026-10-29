"""Unit tests for detector module."""

from __future__ import annotations

import numpy as np
import pytest

from pi_ad_blocker.config import DetectorConfig
from pi_ad_blocker.modules.detector import (
    Detection,
    AD_CLASS_IDS,
)

# Local IoU helper (not exported from module)
def compute_iou_detections(
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


class TestDetection:
    def test_basic_creation(self):
        det = Detection(
            class_id=79,
            class_name="billboard",
            confidence=0.85,
            bbox=(100, 200, 400, 500),
        )
        assert det.class_id == 79
        assert det.class_name == "billboard"
        assert det.confidence == 0.85
        assert det.x1 == 100
        assert det.y1 == 200
        assert det.x2 == 400
        assert det.y2 == 500
        assert det.width == 300
        assert det.height == 300
        assert det.area == 90000

    def test_clip_to_frame(self):
        det = Detection(
            class_id=79, class_name="ad", confidence=0.9,
            bbox=(-50, -50, 700, 500),
        )
        det.clip_to_frame(640, 480)
        assert det.x1 == 0
        assert det.y1 == 0
        assert det.x2 == 640
        assert det.y2 == 480

    def test_track_id_default(self):
        det = Detection(class_id=83, class_name="sign", confidence=0.7,
                        bbox=(10, 10, 100, 100))
        assert det.track_id is None


class TestDetectionUtils:
    def test_iou_computation(self):
        d1 = Detection(79, "ad", 0.9, (0, 0, 100, 100))
        d2 = Detection(79, "ad", 0.9, (50, 50, 150, 150))
        d3 = Detection(79, "ad", 0.9, (200, 200, 300, 300))

        # ~14.3% overlap (50x50 overlap / 17500 union)
        iou = compute_iou_detections(d1.bbox, d2.bbox)
        assert abs(iou - 0.142857) < 0.01

        # No overlap
        iou = compute_iou_detections(d1.bbox, d3.bbox)
        assert iou == 0.0

        # Perfect overlap
        iou = compute_iou_detections(d1.bbox, d1.bbox)
        assert abs(iou - 1.0) < 0.01

    def test_ad_class_ids(self):
        assert 79 in AD_CLASS_IDS
        assert AD_CLASS_IDS[79] == "billboard"
        assert 83 in AD_CLASS_IDS
        assert 84 in AD_CLASS_IDS


class TestDetectorFactory:
    def test_create_yolo_detector(self):
        config = DetectorConfig(model_path="/tmp/fake.onnx", model_type="yolo11")
        from pi_ad_blocker.modules.detector import create_detector
        detector = create_detector(config)
        with pytest.raises(FileNotFoundError):
            detector.load_model()