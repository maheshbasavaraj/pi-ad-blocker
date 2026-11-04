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
        masker = BlackMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), ad_detections)

        # The masked region should be black
        assert (result[20:60, 20:60] == (0, 0, 0)).all()

        # Outside region should preserve the gradient content
        assert not (result[0:10, 0:10] == (0, 0, 0)).all()

    def test_custom_color(self, test_frame, ad_detections):
        masker = BlackMasker(MaskingConfig(use_polygon=False, mask_color=(255, 0, 0)))
        result = masker.apply(test_frame.copy(), ad_detections)
        assert (result[20:60, 20:60] == (255, 0, 0)).all()

    def test_no_detections(self, test_frame):
        masker = BlackMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), [])
        assert (result == test_frame).all()

    def test_opacity(self, test_frame, ad_detections):
        masker = BlackMasker(MaskingConfig(use_polygon=False, mask_opacity=0.5))
        result = masker.apply(test_frame.copy(), ad_detections)
        # With 50% opacity, region should be gray (mix of black and white)
        assert not (result[20:60, 20:60] == (0, 0, 0)).all()
        assert not (result[20:60, 20:60] == (255, 255, 255)).all()

    def test_padding(self, test_frame):
        config = MaskingConfig(use_polygon=False, padding_pixels=5)
        masker = BlackMasker(config)
        dets = [Detection(79, "ad", 0.9, (45, 45, 55, 55))]
        result = masker.apply(test_frame.copy(), dets)
        # With padding=5, should mask 40-60 range
        assert (result[40:60, 40:60] == (0, 0, 0)).all()


class TestBlurMasker:
    def test_blur_changes_region(self, test_frame, ad_detections):
        masker = BlurMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), ad_detections)

        # Blur should change the pixel values in the region
        original_roi = test_frame[20:60, 20:60]
        blurred_roi = result[20:60, 20:60]
        assert not np.allclose(original_roi, blurred_roi)

    def test_outside_unchanged(self, test_frame, ad_detections):
        masker = BlurMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), ad_detections)
        # Pixels far from any detection should match original
        assert (result[0:10, 0:10] == test_frame[0:10, 0:10]).all()


class TestPixelateMasker:
    def test_pixelation_changes_region(self, test_frame, ad_detections):
        masker = PixelateMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), ad_detections)

        original_roi = test_frame[20:60, 20:60]
        pixelated_roi = result[20:60, 20:60]
        assert not np.allclose(original_roi, pixelated_roi)

    def test_outside_unchanged(self, test_frame, ad_detections):
        masker = PixelateMasker(MaskingConfig(use_polygon=False, ))
        result = masker.apply(test_frame.copy(), ad_detections)
        assert (result[0:10, 0:10] == test_frame[0:10, 0:10]).all()


class TestCreateMasker:
    def test_black_masker(self):
        masker = create_masker(MaskingConfig(use_polygon=False, mode="black"))
        assert isinstance(masker, BlackMasker)

    def test_blur_masker(self):
        masker = create_masker(MaskingConfig(use_polygon=False, mode="blur"))
        assert isinstance(masker, BlurMasker)

    def test_pixelate_masker(self):
        masker = create_masker(MaskingConfig(use_polygon=False, mode="pixelate"))
        assert isinstance(masker, PixelateMasker)

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            create_masker(MaskingConfig(use_polygon=False, mode="unknown"))

    def test_custom_needs_fn(self):
        with pytest.raises(ValueError):
            create_masker(MaskingConfig(use_polygon=False, mode="custom"))


class TestPolygonMasker:
    """Pixel-accurate segmentation outlining."""

    def _dets(self, track_id=None):
        poly = np.array([[25, 25], [70, 20], [75, 70], [30, 75]], np.float32)
        return [
            Detection(
                79, "billboard", 0.9, (20, 20, 75, 75),
                track_id=track_id, polygon=poly,
            )
        ]

    def test_centre_of_polygon_is_masked(self, test_frame):
        masker = create_masker(
            MaskingConfig(mode="black", use_polygon=True, edge_feather_px=0)
        )
        result = masker.apply(test_frame.copy(), self._dets())
        assert (result[40:55, 40:55] == 0).all()

    def test_bbox_corner_outside_polygon_survives(self, test_frame):
        # A bbox masker blacks out the whole 20..75 box; the polygon outline
        # leaves the bbox corner (21,21) untouched.
        masker = create_masker(
            MaskingConfig(mode="black", use_polygon=True, edge_feather_px=0)
        )
        result = masker.apply(test_frame.copy(), self._dets())
        assert not (result[21:24, 21:24] == 0).all()

    def test_overlay_mode_black_everywhere_else(self, test_frame):
        cfg = MaskingConfig(
            mode="black", use_polygon=True, output_mode="overlay", edge_feather_px=0
        )
        result = create_masker(cfg).apply(test_frame.copy(), self._dets())
        assert (result[0:10, 0:10] == 0).all()
        assert (result[40:55, 40:55] == 0).all()

    def test_temporal_state_tracks_ids(self, test_frame):
        masker = create_masker(
            MaskingConfig(
                mode="black",
                use_polygon=True,
                temporal_smoothing=True,
                hold_frames=0,
            )
        )
        dets = self._dets(track_id=7)
        for _ in range(3):
            masker.apply(test_frame.copy(), dets)
        assert 7 in masker._state
        masker.apply(test_frame.copy(), [])
        assert 7 not in masker._state

    def test_masks_from_bbox_when_polygon_missing(self, test_frame):
        masker = create_masker(
            MaskingConfig(mode="black", use_polygon=True, edge_feather_px=0)
        )
        det = Detection(79, "billboard", 0.9, (30, 30, 60, 60))
        result = masker.apply(test_frame.copy(), [det])
        assert (result[40:50, 40:50] == 0).all()

    def test_hysteresis_holds_outline_then_fades(self, test_frame):
        masker = create_masker(
            MaskingConfig(
                mode="black",
                use_polygon=True,
                edge_feather_px=0,
                hold_frames=4,
                fade_out=True,
            )
        )
        dets = self._dets(track_id=3)
        first = masker.apply(test_frame.copy(), dets)
        assert not np.array_equal(first, test_frame)

        # Detection gone: mask must persist (no blink) and fade towards nothing.
        held = [masker.apply(test_frame.copy(), []) for _ in range(4)]
        assert not np.array_equal(held[0], test_frame)
        diffs = [
            float(
                np.abs(row.astype(np.int16) - test_frame.astype(np.int16)).mean()
            )
            for row in held
        ]
        assert diffs[0] > 0.0
        assert diffs[0] > diffs[-1], "held mask should fade out"

        # After hold_frames the state is dropped and the frame stays untouched.
        stale = masker.apply(test_frame.copy(), [])
        assert np.array_equal(stale, test_frame)
        assert not masker._state

    def test_hysteresis_disabled_leaves_frame_clean(self, test_frame):
        masker = create_masker(
            MaskingConfig(mode="black", use_polygon=True, hold_frames=0)
        )
        masker.apply(test_frame.copy(), self._dets(track_id=1))
        assert np.array_equal(masker.apply(test_frame.copy(), []), test_frame)