"""RT-DETR (Real-Time Detection Transformer) detector.

RT-DETR is a lightweight transformer-based detector from Baidu.
Good accuracy but higher compute cost than YOLO — recommended only 
with accelerator or on 480p video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import DetectorConfig
from . import BaseDetector, Detection, YOLOONNXDetector


class RTDETRDetector(YOLOONNXDetector):
    """RT-DETR detector using ONNX Runtime.

    RT-DETR uses an end-to-end transformer decoder — no NMS needed.
    We inherit YOLOONNXDetector's ONNX Runtime session management
    but override preprocess/postprocess for RT-DETR's specific format.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w = self.config.input_width
        target_h = self.config.input_height

        # RT-DETR uses direct resize (no letterbox) 
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def postprocess(
        self, raw_output: Any, image_shape: tuple[int, int]
    ) -> list[Detection]:
        orig_h, orig_w = image_shape
        target_w = self.config.input_width
        target_h = self.config.input_height

        # RT-DETR output: [boxes(1,300,4), scores(1,300,80)]
        boxes, scores = raw_output[0], raw_output[1]

        scale_x = orig_w / target_w
        scale_y = orig_h / target_h

        detections: list[Detection] = []
        for i in range(boxes.shape[1]):
            class_scores = scores[0, i, :]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.config.confidence_threshold:
                continue

            # RT-DETR outputs [cx, cy, w, h] in normalized coords
            cx, cy, bw, bh = boxes[0, i, :]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_names[class_id]
                    if class_id < len(self._class_names)
                    else f"class_{class_id}",
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections