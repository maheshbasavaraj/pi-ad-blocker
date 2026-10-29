# Decision Log

All architectural and implementation decisions with rationale.

## Architecture Decisions

### AD-01: Default Detector — YOLO11n INT8 ONNX

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. YOLO11n (INT8) ✅
2. YOLOv8n (INT8)
3. MobileNet SSD v2
4. RT-DETR Small

**Decision:** YOLO11n with INT8 quantization via ONNX Runtime.

**Rationale:**
- Best accuracy/speed tradeoff on RPi 5 CPU (8-12 FPS)
- INT8 reduces model size 4x and improves speed 2-3x
- Ultralytics ecosystem provides mature training + export pipeline
- YOLO11n improves over YOLOv8n by ~2% mAP at same speed

**Tradeoffs:**
- MobileNet SSD is faster (10-15 FPS) but has significantly worse small-object accuracy
- RT-DETR has better accuracy but only 2-4 FPS — unsuitable for real-time

---

### AD-02: Default Tracker — ByteTrack

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. ByteTrack ✅
2. OC-SORT
3. BOT-SORT (with ReID)
4. No tracking

**Decision:** ByteTrack as default, configurable to OC-SORT/BOT-SORT.

**Rationale:**
- Advertisements have static/distinct positions — no need for appearance features
- ByteTrack is the lightest tracker (<1% FPS overhead)
- IoU-based matching sufficient for typical camera views

**Tradeoffs:**
- OC-SORT handles occlusions better (ads behind trees/vehicles)
- BOT-SORT adds ReID for re-entering objects but costs 10% FPS

---

### AD-03: Default Mask — Solid Black Rectangle

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. Solid black ✅
2. Gaussian blur
3. Pixelation
4. Custom

**Decision:** Solid black rectangle, fully opaque.

**Rationale:**
- Zero cognitive load for viewer (complete removal)
- Lowest compute cost (just pixel assignment, no convolution)
- Configurable color provides flexibility

---

### AD-04: Inference Runtime — ONNX Runtime

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. ONNX Runtime ✅
2. TensorFlow Lite
3. OpenCV DNN
4. PyTorch (TorchScript)

**Decision:** ONNX Runtime with INT8 quantization.

**Rationale:**
- Broader model format support than TF Lite
- Better quantization support than OpenCV DNN
- Smaller footprint than PyTorch
- Active development, RPi-compatible

---

### AD-05: Video Backend — OpenCV Primary

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. OpenCV ✅
2. Picamera2 (RPi only)
3. FFmpeg / PyAV
4. GStreamer

**Decision:** OpenCV as primary with Picamera2 for RPi camera module.

**Rationale:**
- Universal camera support (USB, CSI, file)
- Minimal dependencies (opencv-python is already required)
- Picamera2 provides better RPi camera performance

---

### AD-06: Configuration Format — YAML + Pydantic

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. YAML + Pydantic ✅
2. JSON
3. TOML
4. Python config dicts

**Decision:** YAML config files validated with Pydantic models.

**Rationale:**
- YAML is human-friendly, self-documenting
- Pydantic provides type safety, validation, and IDE autocomplete
- Easy to serialize/deserialize for experiments

---

### AD-07: Training Framework — Ultralytics YOLO

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. Ultralytics YOLO ✅
2. MMDetection
3. Detectron2
4. Custom PyTorch

**Decision:** Ultralytics YOLO for training.

**Rationale:**
- Mature pipeline with built-in augmentation, export, validation
- One-command training (`model.train()`)
- Automatic ONNX export with `model.export(format="onnx")`
- Active community and maintenance

---

### AD-08: Accelerator Support — Coral + Hailo

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. Coral TPU (Edge TPU) ✅
2. Hailo-8 / Hailo-8L ✅
3. NVIDIA Jetson (separate platform)
4. Intel Neural Compute Stick (discontinued)

**Decision:** Support both Coral (Google) and Hailo (Hailo) accelerators.

**Rationale:**
- Both available as USB/M.2 modules for RPi 5
- Coral TPU: established ecosystem, easy TF Lite integration
- Hailo-8L: higher TOPS (13 TOPS vs 4 TOPS), growing adoption
- Jetson is a separate platform (not an RPi add-on)

---

### AD-09: Resolution Target — 640×480p

**Status:** Adopted
**Date:** 2025-01-20

**Options considered:**
1. 640×480 ✅
2. 1280×720
3. 1920×1080

**Decision:** 480p as mandatory target, 1080p as desirable.

**Rationale:**
- 480p provides 4x more FPS than 1080p on same hardware
- YOLO11n 640×640 input detects billboards reliably at 480p
- 1080p requires downscaling to 640×640 for detection anyway
- Users can enable 1080p for better quality but lower FPS

---

### AD-10: Programming Language — Python 3.11+

**Status:** Adopted
**Date:** 2025-01-20

**Decision:** Python 3.11+ with type hints throughout.

**Rationale:**
- 3.11 provides ~25% performance improvement over 3.10
- Type hints enable IDE autocomplete and MyPy validation
- All ML libraries (PyTorch, ONNX, OpenCV) support 3.11
- RPi OS Bookworm ships with Python 3.11

---

## Accelerator Benchmarks (RPi 5, 640×480)

| Accelerator | Model | FPS | Latency | Power | Cost |
|-------------|-------|:---:|:-------:|:-----:|:----:|
| CPU only | YOLO11n INT8 | 10 | 100ms | 5W | $0 |
| Coral TPU | YOLO11n INT8 | 22 | 45ms | 7W | $30 |
| Hailo-8L | YOLO11n | 28 | 35ms | 8W | $45 |

## Licensing

- **Pi Ad Blocker**: MIT License
- **Ultralytics YOLO**: AGPL-3.0 (or Ultralytics license)
- **OpenCV**: Apache 2.0
- **ONNX Runtime**: MIT
- **Torch**: BSD
- **Mapillary Vistas**: Custom license (research use only)