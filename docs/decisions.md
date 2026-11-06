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

### AD-10: Real-time Render — Async Detector Thread + Optical-Flow Propagation

**Status:** Adopted
**Date:** 2026-11-03

**Decision:** Run detection on a background thread that consumes only the newest
frame (never a queue), and propagate the last detected polygons on every render
frame with Lucas-Kanade flow re-anchored to the keyframe.

**Rationale:** A 340–450 ms detector cannot gate a 30 fps display loop. Measured
29.97 fps paced render / 60.7 fps uncapped with the detector at 2.2–2.9 fps.
Re-anchoring every frame (not chaining) bounds drift; keyframes older than
`max_propagation_frames` are dropped.

### AD-11: Cross-Class Dedupe Before Tracking

**Status:** Adopted
**Date:** 2026-11-03

**Decision:** Run `agnostic_nms=True` in the detector and `dedupe_detections()`
(IoU 0.6, highest confidence wins) before the tracker.

**Rationale:** Open-vocab prompts fire twice on one sign (`billboard` +
`advertisement poster`), producing two competing track IDs per sign. ID churn
was a direct cause of mask swimming and hysteresis resetting.

### AD-12: Mask Hysteresis + Miss Tolerance

**Status:** Adopted
**Date:** 2026-11-03

**Decision:** Hold the last polygon for `hold_frames` (12) with a linear alpha
fade instead of dropping it on the first empty detection, and let the async
detector keep the previous keyframe through `miss_tolerance` (2) consecutive
empty results.

**Rationale:** Detectors miss a sign for individual keyframes. Instant eviction
made the mask blink. Measured effect: mean consecutive IoU 0.662 → 0.804.

### AD-13: Teacher/Student Distillation for Runtime Detection

**Status:** Adopted
**Date:** 2026-11-03

**Decision:** Ship a fine-tuned nano segmentation model (YOLO11n-seg, ONNX INT8);
use YOLOE open-vocabulary prompts only to auto-label domain frames.

**Rationale:** Four masker-side stability mechanisms moved mean IoU to 0.804 but
left p10 at 0.08 — the gate is p10 ≥ 0.85. The binding constraint is detector
rate: at 2.3 fps each detection is ~13 render frames apart and flow drift snaps
back on every keyframe. YOLOE at 480 doubles throughput and detects nothing, so
it cannot simply be shrunk. A nano model trained on auto-labels makes every frame
a keyframe and removes propagation from the critical path.

---

### AD-14: uint8 Composite Instead of float32 Alpha Blending

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** Composite masks with `cv2.multiply` / `cv2.subtract` in uint8
saturating arithmetic rather than `effect.float * a + base.float * (1-a)` +
`np.clip`.

**Rationale:** The float path allocated three ROI-sized float32 temporaries and
a fourth for the clip: measured 10.87 ms for a 280×320 ROI, and 7.35 ms for the
whole `PolygonMasker.apply` with a single 8.4%-area detection. The uint8 path is
1.68 ms / 3.31 ms — 6.5× faster on the composite, 2.3× on the masker — with a
maximum per-pixel delta of 1 LSB on feathered edges and no change in mask IoU
(0.915 mean / 0.805 p10 over 150 frames). It fits the 3–6 ms composite budget in
the glasses plan for the first time.

**Also:** replaced the full-frame `mask.any()` guard (0.33 ms) with the existing
drawn/held booleans.

---

### AD-15: COCO Ad-Class Filter Applies Only to COCO Detectors

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** `BaseDetector.detect` applies `filter_to_ad_classes` only when the
loaded model looks like a COCO detector (`_is_coco_detector`: ≥20 class names, or
a COCO model type). Single-class domain models bypass it.

**Rationale:** A fine-tuned ad model reports class 0 (`ad`/`item`), which never
intersects `AD_CLASS_IDS` (COCO 62 = tv). With the config default
`filter_to_ad_classes: True`, the ONNX edge path returned **0 detections** on a
frame where the same weights detect 5 ads — the shipped RPi deployment masked
nothing. The `.pt` detector hid the bug because it overrides `detect()` and
skipped the filter entirely. Regression-covered by `TestAdClassFilter`.

---

### AD-16: Static INT8 Is Not the x86 Path

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** Keep FP32 ONNX (and `.pt`) as the default runtime; ship
`scripts/quantize_static.py` for NPU/ARM targets only.

**Rationale:** Static QDQ with 120 calibration frames: the whole-graph INT8
model is 1.57× faster (32.8 vs 50.1 ms) but returns **zero detections** — the
Ultralytics head mixes box, sigmoid-score and mask-coefficient ranges in one
tensor and a single activation scale collapses every score to 0.0. Keeping the
head in FP32 restores detections (box IoU 0.801) but lands at 0.93× FP32
throughput; QOperator is 0.89×. On this x86 CPU (`avx_vnni`, 32 threads) INT8
buys nothing, matching the earlier dynamic-quant finding (225 ms vs 33 ms). Re-test
only on the reference device.

---

### AD-17: `tracker.type: "none"` Actually Works

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** Add `"none"` to `TrackerConfig.type` and a pass-through `NoTracker`
in the tracker factory.

**Rationale:** The deployment guide has told users to set
`tracker.type: "none"` for a ~5% speedup since the first commit, but the
Pydantic `Literal` rejected the value and `create_tracker` had no branch for it —
the documented optimisation was impossible. `pipeline.py` already carried the
`if tracker.type != "none"` path, so only the config gate and factory were
missing. Covered by `test_tracker_none_disables_tracking`.

---

### AD-18: Confidence Threshold 0.25 Chosen by Sweep

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** Keep `confidence_threshold: 0.25` for the student model; document
the curve in `scripts/conf_sweep.py`.

**Rationale:** Measured on the 178-frame val split, 646 GT boxes, NMS IoU 0.5
(Ultralytics' 0.7 leaves 18% self-overlapping duplicates at this threshold):

| conf | precision | recall | F1 |
|---|---|---|---|
| 0.15 | 0.540 | 0.766 | 0.633 |
| 0.20 | 0.574 | 0.729 | 0.642 |
| **0.25** | **0.621** | **0.712** | **0.663** |
| 0.30 | 0.663 | 0.692 | **0.677** (peak) |
| 0.45 | 0.723 | 0.577 | 0.642 |

0.30 peaks by 2.1% F1 with 5% less recall; 0.25 sits inside the noise band and
keeps more ads masked, which is the correct trade for a blocker (a false
positive blacks a sign, a false negative shows the ad). Note the GT is
teacher-derived, so "FP" includes real ads the teacher missed — precision is a
lower bound.

---

### AD-19: Over-Masking Gate Runs in CI

**Status:** Adopted
**Date:** 2026-11-05

**Decision:** `tests/integration/test_class_census_gate.py` scans real footage
with the shipped student and fails the build when (a) the census is empty,
(b) any non-ad class appears, or (c) a single frame has >45% of its pixels
masked. Skips cleanly when weights are absent (they are gitignored artifacts).

**Rationale:** The glasses plan's kill criteria render "masks pedestrians, cars
or traffic lights" a failure regardless of FPS, but no gate enforced it.
A silent zero-detection model (the AD-15 bug) also just looked fast.

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
---

### AD-11: Teacher/Student Distillation for Ad Segmentation

**Status:** Adopted
**Date:** 2026-11-04

**Decision:** Never ship the open-vocabulary teacher. Use `yoloe-11s-seg` offline
to label domain footage, then fine-tune `yolo11n-seg` on those labels.

**Rationale:** The teacher runs at 2.2 fps on CPU and its masks flicker (IoU p10
0.073) because 13 frames coast between keyframes. A 5.96 MB nano student reaches
15.2 fps @640 / 23.5 fps @480 in-pipeline and lifts IoU p10 to 0.776 / 0.800 —
mask hysteresis and flow stop being load-bearing.

**Consequences:**
- Labelled data is free: `scripts/autolabel.py` kept 891 of 977 sampled frames.
- Each new domain (malls, transit, stadiums) is a 20-minute relabel + retrain,
  not a model-vocabulary problem.
- ONNX export path had a latent bug: prototype masks were cropped with
  original-frame coordinates inside letterboxed space, so every mask decoded
  empty. Fixed in `ONNXSegDetector.postprocess`, regression-tested by
  `test_onnx_seg_masks_land_inside_their_boxes`.
- `quantize_dynamic` INT8 is slower than FP32 for this net (225 ms vs 33 ms);
  static QDQ with calibration is required before any INT8 claim.
