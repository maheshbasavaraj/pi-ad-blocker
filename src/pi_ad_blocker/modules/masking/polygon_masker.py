"""Pixel-accurate polygon masking with temporal smoothing.

Improvements over the rectangle maskers:

* Uses the instance outline produced by segmentation detectors, so a billboard
  cut off by a lamp post only has the billboard pixels removed.
* Temporally smooths per-track outlines (EMA of the normalised contour inside
  the track's bounding box) which kills the 1-2 px shimmer that makes
  rectangle masks look artificial on video glasses.
* Edge feathering hides the mask boundary.
* ``output_mode="overlay"`` returns a transparent-friendly overlay layer
  instead of a modified camera frame — the form an AR display compositor
  actually needs.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...config import MaskingConfig
from ..detector import Detection
from . import BaseMasker


def _resample_contour(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to exactly ``n`` equidistant points."""
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return poly
    # Closed polyline for arc-length parametrisation
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-6:
        return poly
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float32)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(j, len(poly) - 1)
        seg_len = max(cum[j + 1] - cum[j], 1e-6)
        frac = (t - cum[j]) / seg_len
        out[i] = closed[j] + frac * (closed[j + 1] - closed[j])
    return out


class PolygonMasker(BaseMasker):
    """Mask the exact segmented outline of each ad, smoothed over time."""

    def __init__(self, config: MaskingConfig):
        self.config = config
        self._state: dict[int, dict] = {}
        self._tick = 0

    def reset(self) -> None:
        self._state.clear()
        self._tick = 0

    # -- public API -----------------------------------------------------
    def apply(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        self._tick += 1
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pad = int(self.config.padding_pixels)
        got_any = False
        live_ids: set[int] = set()

        for det in detections:
            poly = self._smoothed_polygon(det, w, h)
            if poly is None or len(poly) < 3:
                continue
            pts = np.round(poly).astype(np.int32)
            if pad:
                # Grow the outline outwards from its centroid
                c = pts.mean(axis=0, keepdims=True)
                d = pts.astype(np.float32) - c
                norm = np.linalg.norm(d, axis=1, keepdims=True)
                norm[norm < 1e-3] = 1.0
                pts = np.round(pts + (d / norm) * pad).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            got_any = True
            if det.track_id is not None:
                live_ids.add(det.track_id)

        # Hysteresis: a track that vanished this keyframe keeps its last outline
        # for ``hold_frames``, fading out. Without this every empty detection
        # blinked the mask fully off (mask IoU p10 measured 0.048).
        hold = max(int(self.config.hold_frames), 0)
        if hold:
            held = np.zeros_like(mask)
            for tid, st in list(self._state.items()):
                if tid in live_ids or st.get("abs") is None:
                    continue
                age = self._tick - int(st.get("tick", self._tick))
                if 0 < age <= hold:
                    if self.config.fade_out:
                        val = int(255 * (1.0 - age / (hold + 1.0)))
                    else:
                        val = 255
                    if val > 4:
                        cv2.fillPoly(held, [st["abs"].astype(np.int32)], val)
            if held.any():
                mask = np.maximum(mask, held)

        for tid, st in list(self._state.items()):
            age = self._tick - int(st.get("tick", self._tick))
            if tid not in live_ids and age > hold:
                self._state.pop(tid, None)

        if not mask.any():
            return self._empty_output(frame)
        got_any = True

        f = float(self.config.edge_feather_px)
        if f >= 1.0:
            k = int(f * 2) | 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Composite only inside the union bounding box of the masks. Ads cover a
        # fraction of the frame; full-frame float blending cost 2-4x more.
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw <= 0 or bh <= 0:
            return self._empty_output(frame)
        pad = int(f) + 2
        x0, y0 = max(bx - pad, 0), max(by - pad, 0)
        x1, y1 = min(bx + bw + pad, w), min(by + bh + pad, h)

        roi = frame[y0:y1, x0:x1]
        alpha = (
            mask[y0:y1, x0:x1].astype(np.float32) / 255.0
            * float(self.config.mask_opacity)
        )[..., None]
        effect = self._effect(roi)

        if self.config.output_mode == "overlay":
            out = np.zeros_like(frame)
            base = None
        else:
            out = frame.copy()
            base = roi

        blended = effect.astype(np.float32) * alpha
        if base is not None:
            blended += base.astype(np.float32) * (1.0 - alpha)
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return out

    # -- internals ------------------------------------------------------
    def _empty_output(self, frame: np.ndarray) -> np.ndarray:
        if self.config.output_mode == "overlay":
            return np.zeros_like(frame)
        return frame

    def _smoothed_polygon(
        self, det: Detection, w: int, h: int
    ) -> np.ndarray | None:
        poly = det.polygon
        x1, y1, x2, y2 = det.bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        if poly is None or len(poly) < 3:
            # Degenerate case: rectangle outline so the masker still works
            poly = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
            return poly

        poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        n = int(self.config.polygon_points)

        if det.track_id is None or not self.config.temporal_smoothing:
            return poly

        # Normalise contour into bbox space, smooth both the box and the
        # contour, then re-project. Keeps correspondence across frames even
        # when the raw contour point count changes.
        norm = np.empty_like(poly)
        norm[:, 0] = (poly[:, 0] - x1) / bw
        norm[:, 1] = (poly[:, 1] - y1) / bh
        norm = _resample_contour(norm, n)

        prev = self._state.get(det.track_id)
        a = float(self.config.polygon_ema)
        if prev is None:
            sm_poly = norm
            sm_box = np.array([x1, y1, x2, y2], dtype=np.float32)
        else:
            sm_poly = a * norm + (1.0 - a) * prev["poly"]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            sm_box = a * box + (1.0 - a) * prev["box"]

        self._state[det.track_id] = {"poly": sm_poly, "box": sm_box}

        bw2 = max(float(sm_box[2] - sm_box[0]), 1.0)
        bh2 = max(float(sm_box[3] - sm_box[1]), 1.0)
        out = np.empty_like(sm_poly)
        out[:, 0] = np.clip(sm_box[0] + sm_poly[:, 0] * bw2, 0, w - 1)
        out[:, 1] = np.clip(sm_box[1] + sm_poly[:, 1] * bh2, 0, h - 1)

        # Remember the absolute outline + when it was last seen, so the masker
        # can hold it after the detection disappears (hysteresis).
        st = self._state.get(det.track_id)
        if st is not None:
            st["abs"] = out
            st["tick"] = self._tick
        return out

    def _effect(self, frame: np.ndarray) -> np.ndarray:
        mode = self.config.mode
        if mode == "blur":
            k = max(int(self.config.blur_kernel_size) | 1, 3)
            return cv2.GaussianBlur(frame, (k, k), 0)
        if mode == "pixelate":
            blocks = max(int(self.config.pixelation_blocks), 2)
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        # black / custom colour
        return np.full_like(frame, self.config.mask_color)
