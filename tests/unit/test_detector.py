"""Unit tests for detector module."""

from __future__ import annotations

import numpy as np
import pytest

from pi_ad_blocker.config import DetectorConfig
from pi_ad_blocker.modules.detector import (
    Detection,
    AD_CLASS_IDS,
    BaseDetector,
    dedupe_detections,
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
        # COCO's only ad-like class is 62 (tv/monitor). Real ad coverage comes
        # from the open-vocabulary / fine-tuned seg models, not COCO ids.
        assert 62 in AD_CLASS_IDS
        assert AD_CLASS_IDS[62] == "tv_monitor"

    def test_dedupe_drops_cross_class_duplicate(self):
        dets = [
            Detection(0, "billboard", 0.30, (10, 10, 110, 110)),
            Detection(1, "advertisement poster", 0.28, (12, 12, 112, 112)),
            Detection(2, "store sign", 0.20, (400, 300, 500, 380)),
        ]
        kept = dedupe_detections(dets, iou_threshold=0.6)
        assert len(kept) == 2
        assert kept[0].class_name == "billboard"  # highest confidence wins
        assert [d.class_name for d in kept][1] == "store sign"

    def test_dedupe_keeps_distinct_boxes(self):
        dets = [
            Detection(0, "billboard", 0.30, (0, 0, 50, 50)),
            Detection(0, "billboard", 0.25, (200, 200, 260, 260)),
        ]
        assert len(dedupe_detections(dets)) == 2


class TestDetectorFactory:
    def test_create_yolo_detector(self):
        config = DetectorConfig(model_path="/tmp/fake.onnx", model_type="yolo11")
        from pi_ad_blocker.modules.detector import create_detector
        detector = create_detector(config)
        with pytest.raises(FileNotFoundError):
            detector.load_model()


class _StubDetector(BaseDetector):
    """Detector with a canned class list and canned detections."""

    def __init__(self, config, class_names, detections):
        super().__init__(config)
        self._class_names = class_names
        self._canned = detections

    def load_model(self):
        self._warm = True

    def preprocess(self, image):
        return image

    def infer(self, tensor):
        return tensor

    def postprocess(self, raw_output, image_shape):
        return list(self._canned)


class TestAdClassFilter:
    """The COCO ad-class filter must not eat domain-model detections."""

    def test_domain_model_keeps_class_zero(self):
        det = Detection(0, "item", 0.9, (10, 10, 60, 60))
        cfg = DetectorConfig(model_type="onnx_seg", filter_to_ad_classes=True,
                             confidence_threshold=0.25)
        dets = _StubDetector(cfg, ["item"], [det]).detect(np.zeros((64, 64, 3), np.uint8))
        assert len(dets) == 1, "single-class ad model must survive the COCO filter"

    def test_coco_model_still_filters(self):
        from pi_ad_blocker.modules.detector import COCO_CLASSES

        ad = Detection(62, "tv", 0.9, (10, 10, 60, 60))
        person = Detection(0, "person", 0.9, (100, 10, 160, 60))
        cfg = DetectorConfig(model_type="yolo11", filter_to_ad_classes=True,
                             confidence_threshold=0.25)
        dets = _StubDetector(cfg, list(COCO_CLASSES), [ad, person]).detect(
            np.zeros((64, 200, 3), np.uint8)
        )
        assert [d.class_id for d in dets] == [62]

    def test_filter_can_be_disabled(self):
        det = Detection(62, "tv", 0.9, (10, 10, 60, 60))
        cfg = DetectorConfig(model_type="yolo11", filter_to_ad_classes=False,
                             confidence_threshold=0.25)
        dets = _StubDetector(cfg, ["person", "tv"], [det]).detect(
            np.zeros((64, 64, 3), np.uint8)
        )
        assert len(dets) == 1
