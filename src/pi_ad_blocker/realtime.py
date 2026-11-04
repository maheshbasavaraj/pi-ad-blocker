"""Real-time, low-latency ad blocking for live camera / glasses feeds.

The original pipeline runs detect → track → mask on every frame, which costs
15-340 ms per frame depending on the detector. On a display device that is a
hard fail: the compositor must run at 30-60 FPS or the wearer sees stutter.

This module decouples *perception* from *rendering*:

* a detector thread (or a detect-every-N-frames schedule) runs the expensive
  model on the newest frame and produces a keyframe of tracked ad outlines;
* every rendered frame re-uses that keyframe and follows the scene motion with
  sparse Lucas-Kanade optical flow, so masks stay glued to the billboard while
  the head/camera moves;
* if flow tracking fails or the keyframe gets too old, the mask is dropped
  rather than left stuck in the wrong place.

Typical result on CPU: 3-9 FPS detector, 30-60 FPS output.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import AppConfig, RealtimeConfig
from pi_ad_blocker.modules.detector import (
    BaseDetector,
    Detection,
    create_detector,
    dedupe_detections,
)
from pi_ad_blocker.modules.masking import BaseMasker, create_masker
from pi_ad_blocker.modules.masking.polygon_masker import _resample_contour
from pi_ad_blocker.modules.tracker import BaseTracker, Track, create_tracker
from pi_ad_blocker.modules.video_io import (
    Frame,
    VideoWriter,
    create_reader,
    show_preview,
)


def _points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return the subset of ``points`` inside the closed polygon."""
    if poly is None or len(poly) < 3 or points is None or not len(points):
        return np.empty((0, 2), dtype=np.float32)
    contour = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    keep = [
        p
        for p in points
        if cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0
    ]
    return np.asarray(keep, np.float32).reshape(-1, 2)


class FlowPropagator:
    """Carry detection outlines between detector keyframes using LK flow."""

    def __init__(self, n_points: int = 12) -> None:
        self.n_points = n_points
        self._gray: Optional[np.ndarray] = None
        self._dets: list[Detection] = []
        self._pts: Optional[np.ndarray] = None
        self._groups: list[tuple[int, int]] = []

    def set_keyframe(self, gray: np.ndarray, detections: list[Detection]) -> None:
        self._gray = gray
        self._dets = detections
        pts: list[np.ndarray] = []
        groups: list[tuple[int, int]] = []
        start = 0
        for det in detections:
            poly = det.polygon
            if poly is None or len(poly) < 3:
                x1, y1, x2, y2 = det.bbox
                poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            sampled = _resample_contour(np.asarray(poly, np.float32), self.n_points)
            x1, y1, x2, y2 = det.bbox
            extra = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                ],
                dtype=np.float32,
            )
            # Interior grid points. Contour points sit on the ad's edge, where LK
            # latches onto the background instead of the ad; the interior is where
            # the texture that moves *with* the ad actually is.
            inner = []
            step = max((x2 - x1) / 5.0, (y2 - y1) / 5.0, 4.0)
            if step > 0:
                gx = np.arange(x1 + step * 0.5, x2, step, dtype=np.float32)
                gy = np.arange(y1 + step * 0.5, y2, step, dtype=np.float32)
                if len(gx) and len(gy):
                    mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
                    inside = _points_in_polygon(mesh, np.asarray(poly, np.float32))
                    if inside is not None and len(inside):
                        inner = inside
            pts.append(np.vstack([sampled, extra, inner]))
            groups.append((start, start + len(sampled) + len(extra) + len(inner)))
            start += len(sampled) + len(extra) + len(inner)

        self._pts = np.vstack(pts).reshape(-1, 1, 2) if pts else None
        self._groups = groups

    def propagate(self, gray: np.ndarray) -> list[Detection]:
        if self._pts is None or self._gray is None:
            return []
        h, w = gray.shape[:2]
        if self._gray.shape[:2] != gray.shape[:2]:
            return []

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._gray,
            gray,
            self._pts,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if nxt is None or status is None:
            return []

        status = status.reshape(-1).astype(bool)
        moved: list[Detection] = []
        for det, (a, b) in zip(self._dets, self._groups):
            ok = status[a:b]
            if ok.sum() < max(2, int(0.4 * (b - a))):
                continue
            src = self._pts[a:b][ok].reshape(-1, 2).astype(np.float32)
            dst = nxt[a:b][ok].reshape(-1, 2).astype(np.float32)

            # Partial-affine (translate + rotate + uniform scale) fitted with
            # RANSAC. Global translation alone cannot follow a sign that grows as
            # the wearer walks towards it, and its error snapped back on every
            # new keyframe — the dominant source of mask jitter.
            M = None
            if len(src) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    src,
                    dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=200,
                    confidence=0.99,
                )
                n_in = int(inliers.sum()) if inliers is not None else 0
                if M is None or n_in < 0.5 * len(src):
                    M = None
            if M is None:
                delta = dst - src
                dx, dy = np.median(delta, axis=0)
                M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)

            # Reject implausible motion between keyframes
            scale = float(np.hypot(M[0, 0], M[0, 1]))
            if not (0.9 <= scale <= 1.1):
                M[:2, :2] = np.eye(2, dtype=np.float32)

            poly = det.polygon
            if poly is not None and len(poly) >= 3:
                p = np.asarray(poly, np.float32)
                poly2 = p @ M[:, :2].T + M[:, 2]
                poly2[:, 0] = np.clip(poly2[:, 0], 0, w - 1)
                poly2[:, 1] = np.clip(poly2[:, 1], 0, h - 1)
            else:
                poly2 = None

            x1, y1, x2, y2 = det.bbox
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
            corners = corners @ M[:, :2].T + M[:, 2]
            bbox = (
                int(np.clip(corners[:, 0].min(), 0, w - 1)),
                int(np.clip(corners[:, 1].min(), 0, h - 1)),
                int(np.clip(corners[:, 0].max(), 0, w - 1)),
                int(np.clip(corners[:, 1].max(), 0, h - 1)),
            )
            moved.append(
                Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=bbox,
                    track_id=det.track_id,
                    polygon=poly2,
                )
            )
        return moved

    def reset(self) -> None:
        self._gray = None
        self._dets = []
        self._pts = None
        self._groups = []


class RealtimeAdBlocker:
    """Low-latency wrapper: detect on keyframes, propagate + mask every frame."""

    def __init__(
        self,
        config: AppConfig,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[BaseTracker] = None,
        masker: Optional[BaseMasker] = None,
    ) -> None:
        self.config = config
        self.rt: RealtimeConfig = config.realtime
        self.detector = detector or create_detector(config.detector)
        self.tracker = tracker or create_tracker(config.tracker)
        self.masker = masker or create_masker(config.masking)
        self._flow = FlowPropagator(self.rt.flow_points)

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pending: Optional[tuple[int, np.ndarray]] = None

        self._key_idx = -10**9
        self._key_dets: list[Detection] = []
        self._key_gray: Optional[np.ndarray] = None
        self._flow_key_idx = -10**9

        self._frame_idx = 0
        self._miss_streak = 0
        self._detect_ms: list[float] = []
        self._frame_ms: list[float] = []
        self._det_count = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if not self.rt.async_detection or self._thread is not None:
            return
        # Warm up the model on the calling thread: the first inference pays for
        # weight loading / graph capture and would otherwise stall the renderer
        # while it waits for the first keyframe.
        try:
            w = int(self.rt.detect_width or 480)
            dummy = np.zeros((max(int(w * 9 / 16), 64), w, 3), dtype=np.uint8)
            self.detector.detect(dummy)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Detector warm-up failed: {}", exc)
        self._stop.clear()
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None

    def reset(self) -> None:
        self.tracker.reset()
        if hasattr(self.masker, "reset"):
            self.masker.reset()
        self._flow.reset()
        self._key_idx = self._flow_key_idx = -10**9
        self._key_dets = []
        self._key_gray = None
        self._frame_idx = 0
        self._miss_streak = 0
        self._detect_ms.clear()
        self._frame_ms.clear()
        self._det_count = 0

    # -- detection ------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> list[Detection]:
        scale = 1.0
        work = frame
        dw = int(self.rt.detect_width or 0)
        if dw and dw < frame.shape[1]:
            scale = dw / float(frame.shape[1])
            work = cv2.resize(
                frame, (dw, max(int(frame.shape[0] * scale), 1)), interpolation=cv2.INTER_LINEAR
            )

        t0 = time.perf_counter()
        dets = self.detector.detect(work)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if scale != 1.0 and dets:
            inv = 1.0 / scale
            for d in dets:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (
                    int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)
                )
                if d.polygon is not None:
                    d.polygon = np.asarray(d.polygon, np.float32) * inv

        # Cross-class dedup before tracking: two classes on one sign produce
        # duplicate tracks, which is the main source of mask flicker.
        dets = dedupe_detections(dets)

        # Tracker assigns stable IDs across keyframes (EMA masking needs them)
        if self.config.tracker.type != "none":
            tracks = self.tracker.update(dets)
            out: list[Detection] = []
            for tr in tracks:
                out.append(self._from_track(tr))
        else:
            out = dets

        with self._lock:
            self._detect_ms.append(elapsed)
            self._det_count += len(out)
        return out

    @staticmethod
    def _from_track(tr: Track) -> Detection:
        return Detection(
            class_id=tr.class_id,
            class_name=tr.class_name,
            confidence=tr.confidence,
            bbox=tr.bbox,
            track_id=tr.track_id,
            polygon=tr.polygon,
        )

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(timeout=0.1)
                if self._stop.is_set():
                    return
                idx, frame = self._pending
                self._pending = None

            try:
                dets = self._detect(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    if dets:
                        self._miss_streak = 0
                        self._key_idx = idx
                        self._key_dets = dets
                        self._key_gray = gray
                    else:
                        # Detector missed: keep coasting on the last keyframe
                        # until miss_tolerance is exceeded. Instantly clearing
                        # here made the mask blink off for a frame.
                        self._miss_streak += 1
                        tol = max(int(self.rt.miss_tolerance), 0)
                        if self._miss_streak > tol or self._key_gray is None:
                            self._key_idx = idx
                            self._key_dets = []
                            self._key_gray = gray
            except Exception as exc:  # keep the render loop alive
                logger.warning("Detection failed on frame {}: {}", idx, exc)

    # -- rendering ------------------------------------------------------
    def process_frame(self, frame: np.ndarray, idx: Optional[int] = None) -> np.ndarray:
        t0 = time.perf_counter()
        idx = self._frame_idx if idx is None else idx
        self._frame_idx = idx

        if self.rt.async_detection:
            if self._thread is None:
                self.start()
            with self._cond:
                self._pending = (idx, frame)  # newest frame wins, stale dropped
                self._cond.notify()
        else:
            interval = max(int(self.rt.detect_interval), 1)
            if self._key_gray is None or idx - self._key_idx >= interval:
                dets = self._detect(frame)
                if dets or self._key_gray is None:
                    self._miss_streak = 0 if dets else self._miss_streak + 1
                    self._key_idx = idx
                    self._key_dets = dets
                    self._key_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    self._miss_streak += 1
                    if self._miss_streak > max(int(self.rt.miss_tolerance), 0):
                        self._key_idx = idx
                        self._key_dets = []
                        self._key_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with self._lock:
            key_idx = self._key_idx
            key_dets = list(self._key_dets)
            key_gray = self._key_gray

        dets: list[Detection] = []
        if key_gray is not None:
            if idx == key_idx:
                dets = key_dets
            elif idx - key_idx <= self.rt.max_propagation_frames:
                if self._flow_key_idx != key_idx:
                    self._flow.set_keyframe(key_gray, key_dets)
                    self._flow_key_idx = key_idx
                dets = self._flow.propagate(gray)

        out = self.masker.apply(frame, dets)

        with self._lock:
            self._frame_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    # -- stats ----------------------------------------------------------
    def stats(self, wall_s: float, frames: int) -> dict:
        det = sorted(self._detect_ms)
        fr = sorted(self._frame_ms)
        p95 = fr[int(0.95 * (len(fr) - 1))] if fr else 0.0
        return {
            "frames": frames,
            "wall_s": round(wall_s, 3),
            "render_fps": round(frames / wall_s, 2) if wall_s > 0 else 0.0,
            "detector_runs": len(det),
            "detector_fps": round(len(det) / wall_s, 2) if wall_s > 0 else 0.0,
            "detector_ms_avg": round(sum(det) / len(det), 1) if det else 0.0,
            "detector_ms_p95": round(det[int(0.95 * (len(det) - 1))], 1) if det else 0.0,
            "frame_ms_avg": round(sum(fr) / len(fr), 1) if fr else 0.0,
            "frame_ms_p95": round(p95, 1),
            "masked_detections_per_frame": round(self._det_count / max(frames, 1), 2),
        }


def process_video_realtime(
    config: AppConfig,
    input_path: str,
    output_path: Optional[str] = None,
    max_frames: int = 0,
    show_preview_window: bool = False,
    pace_to_source_fps: bool = False,
) -> dict:
    """Run the low-latency pipeline over a video (or camera) source."""
    config.video_io.input_source = input_path
    config.video_io.output_path = output_path
    config.video_io.enable_preview = show_preview_window
    config.realtime.enabled = True

    reader = create_reader(config.video_io)
    writer = VideoWriter(config.video_io)
    blocker = RealtimeAdBlocker(config)

    reader.open()
    first = reader.read()
    if first is None:
        reader.close()
        raise RuntimeError("No frames available from source")

    writer.open((first.width, first.height), reader.fps)
    src_fps = reader.fps if reader.fps and reader.fps > 1 else 30.0
    blocker.start()
    t_start = time.perf_counter()
    frames = 0
    try:
        frame: Optional[Frame] = first
        while frame is not None:
            out = blocker.process_frame(frame.data, frame.frame_number)
            writer.write(out)
            if config.video_io.enable_preview:
                show_preview(out, config.video_io.preview_window_name)
            frames += 1
            if max_frames and frames >= max_frames:
                break
            if pace_to_source_fps:
                target = frames / src_fps
                lag = target - (time.perf_counter() - t_start)
                if lag > 0:
                    time.sleep(lag)
            frame = reader.read()
    finally:
        blocker.stop()
        reader.close()
        writer.close()

    stats = blocker.stats(time.perf_counter() - t_start, frames)
    logger.info(
        "Realtime pipeline: {:.1f} FPS render / {:.1f} FPS detect, {:.1f} ms/frame p95",
        stats["render_fps"],
        stats["detector_fps"],
        stats["frame_ms_p95"],
    )
    return stats


__all__ = ["FlowPropagator", "RealtimeAdBlocker", "process_video_realtime"]
