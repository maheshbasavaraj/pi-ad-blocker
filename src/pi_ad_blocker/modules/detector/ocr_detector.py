"""OCR-based text/logo detection for ads on any surface.

Uses EasyOCR in a background thread so the main pipeline is never blocked.
Catches ads on digital billboards, signs, posters, and any surface that YOLO
misses — works regardless of screen type.

Architecture:
  Main thread (pipeline)              Background thread
  ┌─────────────────┐                 ┌──────────────────────┐
  │ detect(frame)   │ ──submit──▶     │ _ocr_loop()          │
  │ returns cached  │                 │  • readtext(frame)   │
  │ results INSTANT │ ◀──cache────    │  • postprocess       │
  └─────────────────┘                 │  • store results     │
                                      └──────────────────────┘
"""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.modules.detector import Detection


# OCR-ad class IDs (beyond YOLO COCO classes)
OCR_CLASS_ID = 1000
OCR_CLASS_NAME = "text_ad"


class OCRTextDetector:
    """Async text detector using EasyOCR in a background thread.

    Usage:
        ocr = OCRTextDetector(...)
        ocr.start()                     # launches background thread
        # ... pipeline loop ...
        detections = ocr.detect(frame)  # returns cached results instantly
        ocr.stop()                      # cleanup
    """

    def __init__(
        self,
        confidence: float = 0.3,
        languages: list[str] = None,
        min_text_area: int = 800,
        merge_distance: int = 40,
        run_every_n_frames: int = 5,
        skip_if_yolo_detected: bool = True,
    ) -> None:
        self._confidence = confidence
        self._languages = languages or ["en"]
        self._min_text_area = min_text_area
        self._merge_distance = merge_distance
        self._run_every_n = run_every_n_frames
        self._skip_if_yolo = skip_if_yolo_detected

        # Async state
        self._reader: Optional[object] = None
        self._frame_queue: Queue = Queue(maxsize=2)  # drop old frames if full
        self._cache_lock = threading.Lock()
        self._cached_detections: list[Detection] = []
        self._cached_frame_num: int = -1
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._init_event = threading.Event()  # signaled when reader is ready
        self._init_error: Optional[str] = None

        # Stats
        self._inference_times: list[float] = []
        self._stats_lock = threading.Lock()

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Lazy-init EasyOCR and start the background processing thread."""
        if self._running:
            return

        # Init EasyOCR on this thread (blocks until loaded)
        t0 = time.perf_counter()
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (langs={})...", self._languages)
            self._reader = easyocr.Reader(
                self._languages,
                gpu=False,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            logger.info("EasyOCR initialized in {:.1f}s", elapsed)
        except Exception as e:
            self._init_error = str(e)
            logger.error("Failed to init EasyOCR: {}", e)
            self._init_event.set()
            return

        self._init_event.set()
        self._running = True
        self._thread = threading.Thread(target=self._ocr_loop, daemon=True, name="ocr-bg")
        self._thread.start()
        logger.debug("OCR background thread started")

    def stop(self) -> None:
        """Stop the background thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
            self._thread = None
        # Drain queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

    # -------------------------------------------------------------------
    # Main API (called from pipeline, never blocks)
    # -------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        existing_detections: list[Detection] | None = None,
    ) -> list[Detection]:
        """Submit frame for OCR and return latest cached results.

        Returns instantly — never blocks. The background thread does the
        actual OCR on a copy of this frame (every Nth frame).
        """
        # Check initialization
        if not self._init_event.is_set():
            if self._init_error:
                return []  # OCR failed to init, silently skip
            return []  # still initializing, no results yet

        if not self._running:
            return []

        # Submit frame for background processing (every Nth frame)
        # We use an internal counter incremented here (main thread)
        if not hasattr(self, '_frame_counter'):
            self._frame_counter = 0
        self._frame_counter += 1

        if self._frame_counter % self._run_every_n == 0:
            # Skip if YOLO already covers enough area
            if self._skip_if_yolo and existing_detections:
                total_ad_area = sum(
                    (d.x2 - d.x1) * (d.y2 - d.y1)
                    for d in existing_detections
                )
                frame_area = frame.shape[0] * frame.shape[1]
                if total_ad_area > frame_area * 0.15:
                    return self._get_cached()

            # Submit a copy (don't hold ref to pipeline's frame)
            try:
                self._frame_queue.put_nowait({
                    "frame": frame.copy(),
                    "frame_num": self._frame_counter,
                    "ts": time.perf_counter(),
                })
            except Exception:
                pass  # queue full, skip this frame

        return self._get_cached()

    # -------------------------------------------------------------------
    # Background loop
    # -------------------------------------------------------------------

    def _ocr_loop(self) -> None:
        """Background thread: grab frames, run OCR, cache results."""
        while self._running:
            try:
                item = self._frame_queue.get(timeout=0.5)
            except Empty:
                continue

            frame = item["frame"]
            frame_num = item["frame_num"]
            t0 = item["ts"]

            try:
                results = self._reader.readtext(
                    frame,
                    paragraph=False,
                    width_ths=0.7,
                    height_ths=0.7,
                    slope_ths=0.1,
                    ycenter_ths=0.5,
                    adjust_contrast=0.5,
                    text_threshold=0.5,
                    low_text=0.3,
                )

                detections = self._process_results(results, frame.shape[:2])

                elapsed = time.perf_counter() - t0
                with self._stats_lock:
                    self._inference_times.append(elapsed)

                # Update cache
                with self._cache_lock:
                    self._cached_detections = detections
                    self._cached_frame_num = frame_num

                if detections:
                    logger.debug("OCR found {} text regions ({:.1f}s)",
                                 len(detections), elapsed)
            except Exception as e:
                logger.warning("OCR frame failed: {}", e)

    def _process_results(
        self, results: list, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        """Convert EasyOCR results → Detection objects."""
        detections: list[Detection] = []
        for result in results:
            if len(result) == 3:
                bbox, text, confidence = result
            else:
                bbox, text = result
                confidence = 0.5

            if confidence < self._confidence:
                continue

            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            if (x2 - x1) * (y2 - y1) < self._min_text_area:
                continue

            detections.append(Detection(
                class_id=OCR_CLASS_ID,
                class_name=OCR_CLASS_NAME,
                confidence=float(confidence),
                bbox=(x1, y1, x2, y2),
            ))

        # Merge nearby text boxes
        if len(detections) > 1:
            detections = self._merge_nearby(detections)

        return detections

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _get_cached(self) -> list[Detection]:
        """Thread-safe read of latest cached results."""
        with self._cache_lock:
            return self._cached_detections[:]

    def _merge_nearby(
        self, detections: list[Detection]
    ) -> list[Detection]:
        """Merge nearby OCR detections via centroid clustering."""
        if not detections:
            return []

        centroids = np.array([
            [(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2] for d in detections
        ])
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
        confs = np.array([d.confidence for d in detections])

        used = set()
        merged: list[Detection] = []

        for i in range(len(detections)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)

            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                if np.linalg.norm(centroids[i] - centroids[j]) < self._merge_distance:
                    cluster.append(j)
                    used.add(j)

            if len(cluster) == 1:
                merged.append(detections[i])
            else:
                c_boxes = boxes[cluster]
                x1 = int(c_boxes[:, 0].min())
                y1 = int(c_boxes[:, 1].min())
                x2 = int(c_boxes[:, 2].max())
                y2 = int(c_boxes[:, 3].max())
                merged.append(Detection(
                    class_id=OCR_CLASS_ID,
                    class_name=OCR_CLASS_NAME,
                    confidence=float(confs[cluster].mean()),
                    bbox=(x1, y1, x2, y2),
                ))

        return merged

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    @property
    def avg_inference_time_ms(self) -> float:
        with self._stats_lock:
            if not self._inference_times:
                return 0.0
            return (sum(self._inference_times) / len(self._inference_times)) * 1000

    @property
    def avg_fps(self) -> float:
        with self._stats_lock:
            avg_s = sum(self._inference_times) / max(len(self._inference_times), 1)
            return 1.0 / avg_s if avg_s > 0 else 0.0

    def reset_timing(self) -> None:
        with self._stats_lock:
            self._inference_times.clear()

    def visualize(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw OCR detections on frame for debugging."""
        viz = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"OCR {det.confidence:.2f}"
            cv2.putText(viz, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return viz