"""Hailo-8 / Hailo-8L accelerator detector (optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from pi_ad_blocker.config import DetectorConfig
from . import BaseDetector, Detection


class HailoDetector(BaseDetector):
    """Hailo-8 / Hailo-8L NPU-accelerated detector.

    Requires hailo-rt >= 4.17 (pip install pi-ad-blocker[hailo]).
    Uses Hailo's SDK for model inference via HEF file.
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._hailo_session: Any = None
        self._input_vstream: Any = None
        self._output_vstream: Any = None

    def load_model(self) -> None:
        try:
            from hailo_platform import (
                HailoInferenceHardware,
                FormatType,
                HailoStreamInterface,
            )
        except ImportError:
            logger.error("hailo_platform not installed. Install with: pip install pi-ad-blocker[hailo]")
            raise

        model_path = self.config.model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Hailo model not found: {model_path}")

        # Hailo uses .hef format
        hef_path = model_path.replace(".onnx", ".hef")
        if not Path(hef_path).exists():
            logger.warning(
                "HEF file not found at {}, falling back to ONNX CPU inference",
                hef_path,
            )
            from . import YOLOONNXDetector
            self._fallback = YOLOONNXDetector(self.config)
            self._fallback.load_model()
            return

        # Configure Hailo device
        hardware = HailoInferenceHardware()
        device = hardware.configure_device(device_id="")
        self._hailo_session = device.create_network_group(hef_path)
        self._hailo_session.activate()

        # Get input/output stream info
        self._input_vstream = self._hailo_session.input.get_vstreams()[0]
        self._output_vstream = self._hailo_session.output.get_vstreams()[0]

        logger.info("Hailo model loaded: {}", Path(hef_path).name)
        self._fallback = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if hasattr(self, "_fallback") and self._fallback is not None:
            return self._fallback.preprocess(image)
        # Hailo expects quantized uint8 input
        return cv2_resize_letterbox(image, self.config.input_width, self.config.input_height)

    def infer(self, input_tensor: np.ndarray) -> Any:
        if hasattr(self, "_fallback") and self._fallback is not None:
            return self._fallback.infer(input_tensor)

        self._input_vstream.write(input_tensor.tobytes())
        raw_output = self._output_vstream.read()
        # Parse Hailo raw output — format is device-specific
        return self._parse_hailo_output(raw_output)

    def postprocess(self, raw_output: Any, image_shape: tuple[int, int]) -> list[Detection]:
        if hasattr(self, "_fallback") and self._fallback is not None:
            return self._fallback.postprocess(raw_output, image_shape)
        # Hailo postprocessing — returns pre-parsed detections
        return raw_output

    def _parse_hailo_output(self, raw: bytes) -> list[Detection]:
        """Parse Hailo raw tensor output into Detection list."""
        # This is Hailo/YOLO format specific — simplified placeholder
        # Real implementation depends on specific HEF model
        logger.debug("Parsing Hailo output ({} bytes)", len(raw))
        return []


def cv2_resize_letterbox(
    image: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Letterbox resize returning uint8 normalized image."""
    import cv2
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas