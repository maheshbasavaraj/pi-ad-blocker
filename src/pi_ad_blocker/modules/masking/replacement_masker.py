"""Replacement masker — replaces detected ad regions with custom content.

Instead of a black rectangle, this masker:
  1. Takes a replacement image (or video frame)
  2. Computes a perspective homography from replacement → detected region
  3. Warps the replacement to match the screen's angle & perspective
  4. Blends with edge feathering for a seamless composite

Supports both static image replacement and video frame cycling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.modules.masking import BaseMasker
from pi_ad_blocker.modules.detector import Detection


class ReplacementMasker(BaseMasker):
    """Replace detected ad regions with a perspective-warped replacement image.

    Config (in MaskingConfig):
        replacement_image_path: str  — path to replacement image (PNG/JPG)
        replacement_video_path: str  — optional path to replacement video
        replacement_scale: float     — scale replacement relative to bbox (0.8-1.2)
        feather_radius: int          — Gaussian blur radius for edge feathering (px)
        blend_alpha: float           — opacity of replacement over original (0-1)
        temporal_smoothing: bool     — smooth bbox corners across frames
    """

    def __init__(self, config, replacement_image: Optional[np.ndarray] = None):
        super().__init__(config)
        self._replacement: Optional[np.ndarray] = replacement_image
        self._prev_corners: dict[int, np.ndarray] = {}  # track_id → smoothed corners
        self._prev_alpha: float = 0.7  # EMA smoothing factor

    def set_replacement(self, image: np.ndarray) -> None:
        """Set or update the replacement image."""
        self._replacement = image

    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        if self._replacement is None:
            logger.warning("No replacement image set — falling back to black masking")
            result = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = self._pad_bbox(det.bbox, frame.shape[:2])
                result[y1:y2, x1:x2] = (0, 0, 0)
            return result

        return self._apply_replacement(frame, detections)

    def _apply_replacement(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        repl_h, repl_w = self._replacement.shape[:2]

        # Replacement source corners (4 corners of the replacement image)
        src_pts = np.float32([
            [0, 0],
            [repl_w, 0],
            [repl_w, repl_h],
            [0, repl_h],
        ])

        for det in detections:
            x1, y1, x2, y2 = self._pad_bbox(det.bbox, (h, w))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue  # too small to warp onto

            # Destination: 4 corners of the detected bbox
            dst_pts = np.float32([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ])

            # Apply temporal smoothing on corners (reduces jitter)
            if (hasattr(self.config, 'temporal_smoothing')
                    and self.config.temporal_smoothing
                    and det.track_id is not None):
                dst_pts = self._smooth_corners(det.track_id, dst_pts)

            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                # Fallback to affine if homography fails
                H = cv2.getAffineTransform(
                    src_pts[:3].astype(np.float32),
                    dst_pts[:3].astype(np.float32),
                )

            # Warp the replacement image to perspective
            warped = cv2.warpPerspective(
                self._replacement,
                H,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            # Create a mask for the warped region (white on black)
            warp_mask = cv2.warpPerspective(
                np.ones((repl_h, repl_w), dtype=np.uint8) * 255,
                H,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            # Edge feathering — blur the mask so edges blend smoothly
            feather_r = max(3, getattr(self.config, 'feather_radius', 15))
            if feather_r % 2 == 0:
                feather_r += 1
            warp_mask = cv2.GaussianBlur(warp_mask, (feather_r, feather_r), 0)

            # Normalize mask to [0, 1] float
            mask_f = warp_mask.astype(np.float32) / 255.0

            # Blend alpha
            alpha = getattr(self.config, 'blend_alpha', 0.95)

            # Multi-channel blend
            for c in range(3):
                result[:, :, c] = np.where(
                    mask_f > 0.01,
                    (result[:, :, c].astype(np.float32) * (1 - mask_f * alpha)
                     + warped[:, :, c].astype(np.float32) * mask_f * alpha),
                    result[:, :, c],
                ).astype(np.uint8)

        return result

    def _smooth_corners(
        self,
        track_id: int,
        corners: np.ndarray,
    ) -> np.ndarray:
        """Exponential moving average on corner positions to reduce jitter."""
        if track_id in self._prev_corners:
            prev = self._prev_corners[track_id]
            smoothed = (1 - self._prev_alpha) * corners + self._prev_alpha * prev
        else:
            smoothed = corners
        self._prev_corners[track_id] = smoothed
        return smoothed.astype(np.float32)

    def reset(self) -> None:
        """Reset temporal smoothing state for a new video."""
        self._prev_corners.clear()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def load_replacement_image(path: str) -> np.ndarray:
    """Load a replacement image from path. Supports PNG/JPG."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Replacement image not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to load replacement image: {p}")
    logger.info("Loaded replacement image: {} ({}x{})", p, img.shape[1], img.shape[0])
    return img