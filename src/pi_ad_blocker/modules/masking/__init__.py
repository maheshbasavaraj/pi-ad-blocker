"""Masking module — applies visual masks over detected advertisement regions.

Supports solid black, blur, pixelation, replacement, and custom masking modes.
Configurable opacity, color, padding, edge feathering, and custom callbacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import MaskingConfig
from pi_ad_blocker.modules.detector import Detection


# ---------------------------------------------------------------------------
# Masker interface
# ---------------------------------------------------------------------------

class BaseMasker(ABC):
    """Abstract masker — applies visual modification to detection regions."""

    def __init__(self, config: MaskingConfig) -> None:
        self.config = config

    @abstractmethod
    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Apply masking to frame regions covered by detections."""

    def _pad_bbox(
        self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        pad = self.config.padding_pixels
        h, w = frame_shape
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad),
        )


# ---------------------------------------------------------------------------
# Concrete maskers
# ---------------------------------------------------------------------------

class BlackMasker(BaseMasker):
    """Solid black (or custom color) rectangle masking — default."""

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        result = frame.copy()
        color = self.config.mask_color

        for det in detections:
            x1, y1, x2, y2 = self._pad_bbox(det.bbox, frame.shape[:2])
            if self.config.mask_opacity >= 1.0:
                result[y1:y2, x1:x2] = color
            else:
                overlay = result.copy()
                overlay[y1:y2, x1:x2] = color
                alpha = self.config.mask_opacity
                result = cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0)

        return result


class BlurMasker(BaseMasker):
    """Gaussian blur masking — softer alternative to black."""

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        result = frame.copy()
        ksize = self.config.blur_kernel_size
        if ksize % 2 == 0:
            ksize += 1  # must be odd

        for det in detections:
            x1, y1, x2, y2 = self._pad_bbox(det.bbox, frame.shape[:2])
            roi = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

            if self.config.mask_opacity >= 1.0:
                result[y1:y2, x1:x2] = blurred
            else:
                alpha = self.config.mask_opacity
                result[y1:y2, x1:x2] = cv2.addWeighted(
                    blurred, alpha, roi, 1.0 - alpha, 0
                )

        return result


class PixelateMasker(BaseMasker):
    """Pixelation / mosaic masking."""

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        result = frame.copy()
        blocks = self.config.pixelation_blocks

        for det in detections:
            x1, y1, x2, y2 = self._pad_bbox(det.bbox, frame.shape[:2])
            roi = result[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]

            # Resize down then up to pixelate
            small = cv2.resize(
                roi,
                (max(1, roi_w // blocks), max(1, roi_h // blocks)),
                interpolation=cv2.INTER_LINEAR,
            )
            pixelated = cv2.resize(
                small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
            )

            if self.config.mask_opacity >= 1.0:
                result[y1:y2, x1:x2] = pixelated
            else:
                alpha = self.config.mask_opacity
                result[y1:y2, x1:x2] = cv2.addWeighted(
                    pixelated, alpha, roi, 1.0 - alpha, 0
                )

        return result


class CustomMasker(BaseMasker):
    """Custom user-provided masking function."""

    def __init__(self, config: MaskingConfig, custom_fn: Callable) -> None:
        super().__init__(config)
        self._fn = custom_fn

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        return self._fn(frame, detections, self.config)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_masker(
    config: MaskingConfig,
    custom_fn: Optional[Callable] = None,
) -> BaseMasker:
    """Factory: create masker by mode."""
    if config.use_polygon and config.mode in ("black", "blur", "pixelate"):
        from .polygon_masker import PolygonMasker

        return PolygonMasker(config)

    if config.mode == "black":
        return BlackMasker(config)
    elif config.mode == "blur":
        return BlurMasker(config)
    elif config.mode == "pixelate":
        return PixelateMasker(config)
    elif config.mode == "replacement":
        from .replacement_masker import ReplacementMasker, load_replacement_image
        repl_img = load_replacement_image(config.replacement_image_path)
        return ReplacementMasker(config, replacement_image=repl_img)
    elif config.mode == "custom":
        if custom_fn is None:
            raise ValueError("custom_fn required for mode='custom'")
        return CustomMasker(config, custom_fn)
    else:
        raise ValueError(f"Unknown masking mode: {config.mode}")


__all__ = [
    "BaseMasker",
    "BlackMasker",
    "BlurMasker",
    "PixelateMasker",
    "CustomMasker",
    "ReplacementMasker",
    "create_masker",
]