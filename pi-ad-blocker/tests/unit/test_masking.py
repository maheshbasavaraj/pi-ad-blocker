"""Unit tests for masking module."""

from __future__ import annotations

import numpy as np
import pytest

from pi_ad_blocker.config import MaskingConfig
from pi_ad_blocker.modules.detector import Detection
from pi_ad_blocker.modules.masking import (
    BlackMasker,
    BlurMasker,
    PixelateMasker,
    create_masker,
)


@pytest.fixture
def test_frame() -> np.ndarray:
    """Create a 100x100 test image with color variation."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Gradient for non-uniform content
    for i in range(100):
        frame[i, :] = [i * 2, 255 - i, 128]
    return frame


@pytest.fixture
def ad_detections() -> list[Detection]:
    return [
        Detection(79, "billboard", 0.9, (20, 20, 60, 60)),
    ]


class TestBlackMasker:
    def test_black_mask_covers_region(self, test_frame, ad_detections):
        masker = BlackMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), ad_detections)

        # The masked region should be black
        assert (result[20:60, 20:60] == (0, 0, 0)).all()

        # Outside region should preserve the gradient content
        assert not (result[0:10, 0:10] == (0, 0, 0)).all()

    def test_custom_color(self, test_frame, ad_detections):
        masker = BlackMasker(MaskingConfig(mask_color=(255, 0, 0)))
        result = masker.apply(test_frame.copy(), ad_detections)
        assert (result[20:60, 20:60] == (255, 0, 0)).all()

    def test_no_detections(self, test_frame):
        masker = BlackMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), [])
        assert (result == test_frame).all()

    def test_opacity(self, test_frame, ad_detections):
        masker = BlackMasker(MaskingConfig(mask_opacity=0.5))
        result = masker.apply(test_frame.copy(), ad_detections)
        # With 50% opacity, region should be gray (mix of black and white)
        assert not (result[20:60, 20:60] == (0, 0, 0)).all()
        assert not (result[20:60, 20:60] == (255, 255, 255)).all()

    def test_padding(self, test_frame):
        config = MaskingConfig(padding_pixels=5)
        masker = BlackMasker(config)
        dets = [Detection(79, "ad", 0.9, (45, 45, 55, 55))]
        result = masker.apply(test_frame.copy(), dets)
        # With padding=5, should mask 40-60 range
        assert (result[40:60, 40:60] == (0, 0, 0)).all()


class TestBlurMasker:
    def test_blur_changes_region(self, test_frame, ad_detections):
        masker = BlurMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), ad_detections)

        # Blur should change the pixel values in the region
        original_roi = test_frame[20:60, 20:60]
        blurred_roi = result[20:60, 20:60]
        assert not np.allclose(original_roi, blurred_roi)

    def test_outside_unchanged(self, test_frame, ad_detections):
        masker = BlurMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), ad_detections)
        # Pixels far from any detection should match original
        assert (result[0:10, 0:10] == test_frame[0:10, 0:10]).all()


class TestPixelateMasker:
    def test_pixelation_changes_region(self, test_frame, ad_detections):
        masker = PixelateMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), ad_detections)

        original_roi = test_frame[20:60, 20:60]
        pixelated_roi = result[20:60, 20:60]
        assert not np.allclose(original_roi, pixelated_roi)

    def test_outside_unchanged(self, test_frame, ad_detections):
        masker = PixelateMasker(MaskingConfig())
        result = masker.apply(test_frame.copy(), ad_detections)
        assert (result[0:10, 0:10] == test_frame[0:10, 0:10]).all()


class TestCreateMasker:
    def test_black_masker(self):
        masker = create_masker(MaskingConfig(mode="black"))
        assert isinstance(masker, BlackMasker)

    def test_blur_masker(self):
        masker = create_masker(MaskingConfig(mode="blur"))
        assert isinstance(masker, BlurMasker)

    def test_pixelate_masker(self):
        masker = create_masker(MaskingConfig(mode="pixelate"))
        assert isinstance(masker, PixelateMasker)

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            create_masker(MaskingConfig(mode="unknown"))

    def test_custom_needs_fn(self):
        with pytest.raises(ValueError):
            create_masker(MaskingConfig(mode="custom"))