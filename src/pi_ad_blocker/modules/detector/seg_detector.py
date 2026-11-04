"""Segmentation-capable advertisement detectors.

Two backends:

* ``UltralyticsSegDetector`` — YOLO11-seg / YOLOE (open-vocabulary) via the
  Ultralytics runtime. Used for training, auto-labelling and evaluation.
  YOLOE accepts free-text prompts, so ads can be found without a custom
  dataset — ideal for bootstrapping a commercial ad-blocking model.
* ``ONNXSegDetector`` — YOLO11-seg ONNX with hand-written mask decoding.
  This is the on-device path (RPi 5 / NPU class hardware) — no PyTorch.

Both return ``Detection`` objects carrying a pixel-accurate ``polygon`` so the
masker can cut the exact ad silhouette instead of a rectangle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from . import COCO_CLASSES, BaseDetector, Detection
from pi_ad_blocker.config import DetectorConfig


def _polygon_from_mask(mask: np.ndarray, min_area: float = 16.0) -> Optional[np.ndarray]:
    """Largest external contour of a binary mask as an (N, 2) int32 polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    # Approximate to keep the polygon cheap to warp / transform each frame
    eps = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2).astype(np.int32)


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Ultralytics (PyTorch / auto-export) detector
# ---------------------------------------------------------------------------

class UltralyticsSegDetector(BaseDetector):
    """YOLO11-seg and YOLOE open-vocabulary detector."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._prompted = False

    # -- lifecycle ---------------------------------------------------------
    def load_model(self) -> None:
        from ultralytics import YOLO, YOLOE

        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        is_yoloe = self.config.model_type == "yoloe" or "yoloe" in path.name.lower()
        prompts = [p for p in self.config.prompt_classes if p.strip()]

        if is_yoloe:
            self._model = YOLOE(str(path))
            if prompts:
                self._model.set_classes(prompts, self._model.get_text_pe(prompts))
                self._prompted = True
                self._class_names = prompts
                logger.info("YOLOE text prompts: {}", prompts)
        else:
            self._model = YOLO(str(path))
            self._class_names = (
                list(self._model.names.values())
                if isinstance(self._model.names, dict)
                else list(self._model.names)
            )

        logger.info(
            "Loaded seg model: {} | prompted={} | classes={}",
            path.name, self._prompted, len(self._class_names),
        )

    # -- inference ---------------------------------------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:  # handled by ultralytics
        return image

    def infer(self, input_tensor: Any) -> Any:  # handled by ultralytics
        return input_tensor

    def postprocess(self, raw_output: Any, image_shape) -> list[Detection]:  # unused
        return []

    def detect(self, image: np.ndarray) -> list[Detection]:  # type: ignore[override]
        if not self._warm:
            self.load_model()
            self._warm = True

        t0 = time.perf_counter()
        results = self._model.predict(
            image,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_width,
            agnostic_nms=True,  # billboard vs "advertisement poster" on the same sign
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        self._inference_times.append(elapsed)

        r = results[0]
        detections: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        masks_xy = r.masks.xy if getattr(r, "masks", None) is not None else [None] * len(r.boxes)
        names = r.names if isinstance(r.names, dict) else {
            i: n for i, n in enumerate(r.names)
        }

        for box, poly in zip(r.boxes, masks_xy):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            polygon = None
            if poly is not None and len(poly) >= 3:
                polygon = np.asarray(poly, dtype=np.int32).reshape(-1, 2)

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    polygon=polygon,
                    source="yoloe" if self._prompted else "yolo_seg",
                )
            )

        return self._filter_small(detections, image.shape[:2])

    def _filter_small(
        self, detections: list[Detection], shape: tuple[int, int], min_ratio: float = 0.0015
    ) -> list[Detection]:
        frame_area = float(shape[0] * shape[1])
        return [
            d for d in detections
            if (d.width * d.height) / frame_area >= min_ratio
        ]


# ---------------------------------------------------------------------------
# ONNX Runtime segmentation detector (edge path)
# ---------------------------------------------------------------------------

class ONNXSegDetector(BaseDetector):
    """YOLO11-seg ONNX detector with native mask decoding.

    Expects a standard Ultralytics segmentation export:
        output0: (1, 4 + nc + 32, num_anchors)
        output1: (1, 32, mh, mw)  prototype masks
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._output_names: list[str] = []
        self._proto: Optional[np.ndarray] = None
        self._mask_dim = 32
        self._mask_thresh = 0.5
        self._in_w = int(config.input_width)
        self._in_h = int(config.input_height)

    def load_model(self) -> None:
        import onnxruntime as ort

        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 2

        providers = self.config.providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(path), opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Honour the model's static input size (exports are usually 480/640)
        shape = self._session.get_inputs()[0].shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                self._in_h = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0:
                self._in_w = shape[3]

        # Class names embedded by Ultralytics exports
        try:
            import ast
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            if "names" in meta:
                names = ast.literal_eval(meta["names"])
                if isinstance(names, dict):
                    self._class_names = [names[i] for i in sorted(names)]
                elif isinstance(names, (list, tuple)):
                    self._class_names = list(names)
        except Exception:  # pragma: no cover - metadata optional
            pass

        logger.info("Loaded ONNX seg model: {} | outputs={} | input={}x{}",
                    path.name, self._output_names, self._in_w, self._in_h)

        if self.config.custom_ad_classes:
            self._class_names = list(self.config.custom_ad_classes)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tw, th = self._in_w, self._in_h
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB
        blob = np.transpose(blob, (2, 0, 1))[None]
        return np.ascontiguousarray(blob)

    def infer(self, input_tensor: np.ndarray) -> Any:
        return self._session.run(self._output_names, {self._input_name: input_tensor})

    def postprocess(self, raw_output: Any, image_shape) -> list[Detection]:
        orig_h, orig_w = image_shape
        tw, th = self._in_w, self._in_h
        scale = min(tw / orig_w, th / orig_h)
        pad_x = (tw - orig_w * scale) / 2
        pad_y = (th - orig_h * scale) / 2

        preds = np.asarray(raw_output[0])
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:      # (channels, anchors) → (anchors, channels)
            preds = preds.T

        nc = preds.shape[1] - 4 - self._mask_dim
        if nc <= 0:
            logger.warning("ONNX seg model layout unexpected: {}", preds.shape)
            return []

        scores_all = preds[:, 4:4 + nc]
        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(preds)), cls_ids]

        keep = confs >= self.config.confidence_threshold
        if not np.any(keep):
            return []
        preds, cls_ids, confs = preds[keep], cls_ids[keep], confs[keep]

        boxes = preds[:, :4]
        coeffs = preds[:, 4 + nc:4 + nc + self._mask_dim]

        # NMS
        xywh = np.empty_like(boxes)
        xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xywh[:, 2] = boxes[:, 2]
        xywh[:, 3] = boxes[:, 3]
        idx = cv2.dnn.NMSBoxes(
            xywh.tolist(), confs.tolist(), 0.0, self.config.iou_threshold
        )
        if len(idx) == 0:
            return []
        idx = np.asarray(idx).flatten()
        boxes, cls_ids, confs, coeffs = boxes[idx], cls_ids[idx], confs[idx], coeffs[idx]

        proto_out = raw_output[-1]
        proto = np.asarray(proto_out)
        if proto.ndim == 4:
            proto = proto[0]
        mh, mw = proto.shape[1], proto.shape[2]
        flat_proto = proto.reshape(self._mask_dim, -1)
        masks = _sig(coeffs @ flat_proto).reshape(-1, mh, mw)

        detections: list[Detection] = []
        for i in range(len(boxes)):
            cx, cy, bw, bh = (float(v) for v in boxes[i])

            # Box in letterboxed input space (what the mask crop must use)
            ix1 = max(0.0, cx - bw / 2)
            iy1 = max(0.0, cy - bh / 2)
            ix2 = min(float(tw), cx + bw / 2)
            iy2 = min(float(th), cy + bh / 2)

            # Same box mapped back to the original frame
            x1 = (ix1 - pad_x) / scale
            y1 = (iy1 - pad_y) / scale
            x2 = (ix2 - pad_x) / scale
            y2 = (iy2 - pad_y) / scale
            bx1 = max(0, int(round(x1))); by1 = max(0, int(round(y1)))
            bx2 = min(orig_w, int(round(x2))); by2 = min(orig_h, int(round(y2)))
            if bx2 - bx1 < 2 or by2 - by1 < 2:
                continue

            m = cv2.resize(masks[i], (tw, th), interpolation=cv2.INTER_LINEAR)
            crop = m[int(iy1):int(np.ceil(iy2)), int(ix1):int(np.ceil(ix2))]
            if crop.size == 0:
                continue

            # Crop is in input space; scale it straight to the box's real size
            binm = (cv2.resize(crop, (bx2 - bx1, by2 - by1),
                               interpolation=cv2.INTER_LINEAR) > self._mask_thresh)
            full = np.zeros((orig_h, orig_w), dtype=np.uint8)
            full[by1:by2, bx1:bx2] = binm.astype(np.uint8) * 255

            detections.append(
                Detection(
                    class_id=int(cls_ids[i]),
                    class_name=self._class_names[int(cls_ids[i])]
                    if int(cls_ids[i]) < len(self._class_names) else f"class_{int(cls_ids[i])}",
                    confidence=float(confs[i]),
                    bbox=(bx1, by1, bx2, by2),
                    polygon=_polygon_from_mask(full),
                    source="onnx_seg",
                )
            )

        return detections


__all__ = ["UltralyticsSegDetector", "ONNXSegDetector"]
