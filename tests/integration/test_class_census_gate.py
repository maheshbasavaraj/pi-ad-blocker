"""Over-masking gate: what is the detector actually painting over?

The glasses plan makes "a run that masks pedestrians, cars or traffic lights a
failure regardless of FPS" a kill criterion, but nothing enforced it in CI.
This test scans real footage with the shipped student model and asserts:

1. the census is not empty (a silent zero-detection model fails the gate),
2. every class reported is an ad class (single-class domain model),
3. no frame has an implausible share of its pixels masked (over-masking).

It skips when the weights are absent, since ``*.onnx`` / ``*.pt`` are not in the
repo — the gate is for the machine that trained the model, and for CI that
fetches weights as an artifact.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from pi_ad_blocker.config import DetectorConfig, MaskingConfig  # noqa: E402
from pi_ad_blocker.modules.detector.seg_detector import (  # noqa: E402
    UltralyticsSegDetector,
)
from pi_ad_blocker.modules.masking.polygon_masker import PolygonMasker  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "runs/seg/ads_nano_seg/weights/best.pt"
VIDEO = ROOT / "data/outputs/test_ad_video.mp4"
AD_NAMES = {"ad", "item", "billboard", "advertisement poster", "store sign",
            "digital screen", "sign", "poster"}

pytestmark = pytest.mark.skipif(
    not MODEL.exists() or not VIDEO.exists(),
    reason="student weights / footage not present (gitignored artifacts)",
)


def test_census_is_ad_only_and_mask_area_is_plausible():
    det = UltralyticsSegDetector(
        DetectorConfig(
            model_path=str(MODEL),
            model_type="yolo_seg",
            confidence_threshold=0.25,
            input_width=640,
            input_height=640,
            filter_to_ad_classes=False,
        )
    )
    masker = PolygonMasker(
        MaskingConfig(mode="black", use_polygon=True, edge_feather_px=0)
    )

    cap = cv2.VideoCapture(str(VIDEO))
    census: Counter[str] = Counter()
    areas: list[float] = []
    scanned = 0
    try:
        for i in range(150):
            ok, frame = cap.read()
            if not ok:
                break
            if i % 10:
                continue
            scanned += 1
            dets = det.detect(frame)
            for d in dets:
                census[d.class_name] += 1
            masked = masker.apply(frame, dets)
            changed = (
                cv2.absdiff(frame, masked).max(axis=2) > 20
            ).mean()
            areas.append(float(changed))
    finally:
        cap.release()

    assert scanned >= 10, "not enough frames scanned"
    assert census, "census EMPTY — detector found nothing (silent zero-detection model)"
    assert set(census) <= AD_NAMES, f"non-ad classes masked: {set(census) - AD_NAMES}"
    assert max(areas) < 0.45, f"over-masking: one frame had {max(areas):.0%} of pixels masked"
    assert sum(areas) / len(areas) > 0.01, "masker changed almost nothing — is anything masked?"
