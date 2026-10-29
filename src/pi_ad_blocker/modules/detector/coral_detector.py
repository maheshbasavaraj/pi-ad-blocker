"""Coral TPU Edge TPU detector (optional accelerator)."""

from __future__ import annotations

from pi_ad_blocker.config import DetectorConfig
from . import YOLOONNXDetector


class CoralDetector(YOLOONNXDetector):
    """Coral TPU-accelerated detector via ONNX Runtime with Edge TPU provider.

    Requires pycoral and tflite-runtime installed (pip install pi-ad-blocker[tpu]).
    Falls back to CPU if Edge TPU is unavailable.
    """

    def __init__(self, config: DetectorConfig) -> None:
        config.providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        super().__init__(config)