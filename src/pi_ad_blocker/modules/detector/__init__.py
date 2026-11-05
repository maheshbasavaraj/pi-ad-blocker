"""Advertisement detection module.

Supports YOLOv8, YOLO11, MobileNet-SSD, and RT-DETR with ONNX Runtime
for efficient inference on Raspberry Pi. Provides unified interface for
all detectors with configurable backend (CPU, Coral TPU, Hailo).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import DetectorConfig

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection result.

    ``polygon`` carries the pixel-accurate instance outline when the detector
    is segmentation-capable (YOLO11-seg / YOLOE). Maskers prefer the polygon
    over the bounding box when it is present.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    track_id: Optional[int] = None
    polygon: Optional[np.ndarray] = None  # (N, 2) int32 outline, frame coords
    source: str = "yolo"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip_to_frame(self, frame_w: int, frame_h: int) -> None:
        """Clip bounding box to frame boundaries."""
        x1, y1, x2, y2 = self.bbox
        self.bbox = (
            max(0, x1),
            max(0, y1),
            min(frame_w, x2),
            min(frame_h, y2),
        )


# ---------------------------------------------------------------------------
# COCO class names relevant to advertisements
# ---------------------------------------------------------------------------

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Ad-relevant COCO class IDs and names
AD_CLASS_IDS: dict[int, str] = {
    62: "tv_monitor",       # TV / monitor — digital billboards, screens
}


# ---------------------------------------------------------------------------
# Abstract base detector
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract interface for all advertisement detectors."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._class_names: list[str] = COCO_CLASSES.copy()
        self._warm = False
        self._inference_times: list[float] = []

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @class_names.setter
    def class_names(self, names: list[str]) -> None:
        self._class_names = names

    # Model types that ship as COCO-80 detectors. Fine-tuned ad models
    # (``yolo_seg`` / ``onnx_seg`` / ``yoloe``) are single-class and must not
    # be filtered by COCO ids.
    _COCO_MODEL_TYPES = frozenset(
        {"yolov8", "yolo11", "yolov11", "mobilenet_ssd", "rtdetr"}
    )

    def _is_coco_detector(self) -> bool:
        """Whether the COCO ad-class filter is meaningful for this model.

        A fine-tuned ad model reports class 0 (``ad``/``item``), which never
        intersects ``AD_CLASS_IDS`` (COCO 62 = tv). Applying the filter to it
        silently dropped every detection: the ONNX edge path returned 0 boxes
        on a frame where the same weights detect 5 ads.
        """
        names = self._class_names or []
        if len(names) >= 20:
            return True
        if len(names) <= 1:
            return False
        return self.config.model_type in self._COCO_MODEL_TYPES

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model into memory."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess one frame → model input tensor."""

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> Any:
        """Run model inference."""

    @abstractmethod
    def postprocess(
        self, raw_output: Any, image_shape: tuple[int, int]
    ) -> list[Detection]:
        """Convert raw model output to list of Detections."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Full detect pipeline: preprocess → infer → postprocess."""
        if not self._warm:
            self.load_model()
            self._warm = True

        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        raw = self.infer(tensor)
        detections = self.postprocess(raw, image.shape[:2])
        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)

        # Filter to ad-relevant COCO classes only for COCO-style detectors.
        # A fine-tuned single-class ad model reports class 0 ("ad"/"item"),
        # which is not in AD_CLASS_IDS — filtering it dropped every detection
        # (the ONNX edge path returned 0 boxes on footage it detects 5 ads in).
        if self.config.filter_to_ad_classes and self._is_coco_detector():
            ad_ids = set(self.config.ad_class_ids)
            detections = [d for d in detections if d.class_id in ad_ids]

        # Apply confidence threshold
        detections = [d for d in detections if d.confidence >= self.config.confidence_threshold]

        # NMS
        detections = self._nms(detections, self.config.iou_threshold)

        for d in detections:
            d.clip_to_frame(image.shape[1], image.shape[0])

        return detections

    def _nms(self, detections: list[Detection], iou_thresh: float) -> list[Detection]:
        """Non-maximum suppression."""
        if not detections:
            return []

        boxes = np.array([(d.x1, d.y1, d.x2, d.y2) for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 0.0, iou_thresh
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [detections[i] for i in indices]

    @property
    def avg_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return (sum(self._inference_times) / len(self._inference_times)) * 1000

    @property
    def avg_fps(self) -> float:
        avg_s = sum(self._inference_times) / max(len(self._inference_times), 1)
        return 1.0 / avg_s if avg_s > 0 else 0.0

    def reset_timing(self) -> None:
        self._inference_times.clear()


# ---------------------------------------------------------------------------
# YOLO ONNX detector (YOLOv8 / YOLO11)
# ---------------------------------------------------------------------------

class YOLOONNXDetector(BaseDetector):
    """YOLOv8/YOLO11 ONNX detector. Primary recommended detector for RPi 5."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._output_names: list[str] = []
        self._input_shape: Optional[tuple[int, ...]] = None

    def load_model(self) -> None:
        import onnxruntime as ort

        model_path = self.config.model_path
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Configure ONNX Runtime session options for edge deployment
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_profiling = self.config.enable_trace
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        if self.config.use_quantized and "quant" in model_path.lower():
            logger.info("Loading quantized INT8 model")

        self._session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=self.config.providers,
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_shape = self._session.get_inputs()[0].shape
        logger.info(
            "Loaded model: {} | Input: {} | Outputs: {}",
            Path(model_path).name,
            self._input_shape,
            self._output_names,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_w = self.config.input_width
        target_h = self.config.input_height

        # Letterbox resize preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Normalize and transpose to NCHW
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob

    def infer(self, input_tensor: np.ndarray) -> Any:
        outputs = self._session.run(
            self._output_names,
            {self._input_name: input_tensor},
        )
        return outputs

    def postprocess(
        self, raw_output: Any, image_shape: tuple[int, int]
    ) -> list[Detection]:
        orig_h, orig_w = image_shape
        target_w = self.config.input_width
        target_h = self.config.input_height

        # Calculate scale and offset from letterbox
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_x = (target_w - orig_w * scale) / 2
        pad_y = (target_h - orig_h * scale) / 2

        predictions = raw_output[0]  # (1, num_classes+4, num_dets) or (1, num_dets, num_classes+4)
        if predictions.ndim == 3 and predictions.shape[1] < predictions.shape[2]:
            predictions = predictions.transpose(0, 2, 1)

        detections: list[Detection] = []
        for det in predictions[0]:
            scores = det[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.config.confidence_threshold:
                continue

            # Denormalize box coordinates
            cx, cy, bw, bh = det[:4]
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_names[class_id]
                    if class_id < len(self._class_names)
                    else f"class_{class_id}",
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def dedupe_detections(
    detections: list[Detection], iou_threshold: float = 0.6
) -> list[Detection]:
    """Drop duplicate boxes that different classes claim over the same region.

    Open-vocabulary prompts overlap by nature ("billboard" vs "advertisement
    poster"), and a per-class NMS keeps both. Two masks over one sign double
    the area and make the outline jitter between keyframes.
    """

    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
        area_b = max((bx2 - bx1) * (by2 - by1), 1)
        return inter / float(area_a + area_b - inter)

    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(iou(det.bbox, k.bbox) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def create_detector(config: DetectorConfig) -> BaseDetector:
    """Factory: create the appropriate detector based on configuration."""
    if config.accelerator == "coral":
        from .coral_detector import CoralDetector
        return CoralDetector(config)
    elif config.accelerator == "hailo":
        from .hailo_detector import HailoDetector
        return HailoDetector(config)

    if config.model_type in ("yolov8", "yolo11", "yolov11"):
        return YOLOONNXDetector(config)
    elif config.model_type == "mobilenet_ssd":
        from .mobilenet_detector import MobileNetSSDDetector
        return MobileNetSSDDetector(config)
    elif config.model_type == "rtdetr":
        from .rtdetr_detector import RTDETRDetector
        return RTDETRDetector(config)
    elif config.model_type in ("yoloe", "yolo_seg", "ultralytics"):
        from .seg_detector import UltralyticsSegDetector
        return UltralyticsSegDetector(config)
    elif config.model_type == "onnx_seg":
        from .seg_detector import ONNXSegDetector
        return ONNXSegDetector(config)
    else:
        raise ValueError(f"Unknown detector type: {config.model_type}")


__all__ = [
    "BaseDetector",
    "Detection",
    "YOLOONNXDetector",
    "create_detector",
    "dedupe_detections",
    "AD_CLASS_IDS",
    "COCO_CLASSES",
]