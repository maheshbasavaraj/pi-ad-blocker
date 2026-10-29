"""MobileNet SSD detector (lightweight alternative)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import DetectorConfig
from . import BaseDetector, Detection


class MobileNetSSDDetector(BaseDetector):
    """MobileNet SSD via OpenCV DNN — lightweight, CPU-friendly.

    Pros: Very fast on CPU, tiny model (~6 MB FP16, ~3 MB INT8).
    Cons: Lower accuracy than YOLO variants for small/distant ads.
    Best for: 480p real-time CPU-only mode on Raspberry Pi 5.
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._net: Any = None
        self._model_cfg: str = ""

    def load_model(self) -> None:
        model_path = self.config.model_path
        # MobileNet SSD uses separate prototxt + caffemodel OR one .pbtxt
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-detect format: .caffemodel/.prototxt or frozen .pb/.pbtxt
        p = str(model_path)
        if p.endswith(".caffemodel"):
            prototxt = p.replace(".caffemodel", ".prototxt")
            self._net = cv2.dnn.readNetFromCaffe(prototxt, p)
        elif p.endswith(".pb"):
            pbtxt = p.replace(".pb", ".pbtxt")
            self._net = cv2.dnn.readNetFromTensorflow(p, pbtxt)
        elif p.endswith(".onnx"):
            self._net = cv2.dnn.readNetFromONNX(p)
        else:
            raise ValueError(f"Unrecognized model format: {p}")

        # Set backend and target for CPU
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("MobileNet SSD loaded: {}", Path(model_path).name)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # MobileNet SSD uses 300x300 input
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 127.5,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        return blob

    def infer(self, input_tensor: np.ndarray) -> Any:
        self._net.setInput(input_tensor)
        return self._net.forward()

    def postprocess(
        self, raw_output: Any, image_shape: tuple[int, int]
    ) -> list[Detection]:
        h, w = image_shape
        detections = []

        for i in range(raw_output.shape[2]):
            confidence = float(raw_output[0, 0, i, 2])
            if confidence < self.config.confidence_threshold:
                continue

            class_id = int(raw_output[0, 0, i, 1])
            x1 = int(raw_output[0, 0, i, 3] * w)
            y1 = int(raw_output[0, 0, i, 4] * h)
            x2 = int(raw_output[0, 0, i, 5] * w)
            y2 = int(raw_output[0, 0, i, 6] * h)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_names[class_id]
                    if class_id < len(self._class_names)
                    else f"class_{class_id}",
                    confidence=confidence,
                    bbox=(max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                )
            )

        return detections