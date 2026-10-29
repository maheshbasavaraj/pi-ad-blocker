# Pi Ad Blocker — Architecture Document

## Overview

The Pi Ad Blocker is a modular, pipeline-based system designed to detect
and mask advertisements in real-time video streams on resource-constrained
edge devices (Raspberry Pi 5).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │   CLI    │  │  Python  │  │  Config  │  │   Docker / CI     │  │
│  │  Entry   │  │   API    │  │  System  │  │   Deployment      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                       Pipeline Layer                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              AdBlockPipeline                                 │  │
│  │  Frame → Detector → Tracker → Masker → Writer/Preview       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                        Module Layer                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │Detector  │ │ Tracker  │ │ Masker   │ │Video I/O │ │ Evaluator││
│  │YOLO11n   │ │ByteTrack │ │Black/Blur│ │OpenCV/   │ │Precision ││
│  │YOLOv8n   │ │OC-SORT   │ │Pixelate  │ │Picamera2 │ │Recall/mAP││
│  │MobileNet │ │BOT-SORT  │ │Custom    │ │File/Cam  │ │FPS/Lat   ││
│  │RT-DETR   │ │          │ │          │ │Writer    │ │Reports   ││
│  │Coral TPU │ │          │ │          │ │          │ │          ││
│  │Hailo NPU │ │          │ │          │ │          │ │          ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ONNX      │ │ONNX RT   │ │Model     │ │Training  │ │Data      ││
│  │Export    │ │Inference  │ │Quant (INT8)│Pipeline  │ │Pipeline  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Detector Module (`modules/detector/`)

Responsible for detecting advertisement regions in each frame.

**Components:**
- `BaseDetector` — Abstract interface with `detect()`, `preprocess()`, `infer()`, `postprocess()`
- `YOLOONNXDetector` — YOLOv8/YOLO11 via ONNX Runtime (primary)
- `MobileNetSSDDetector` — OpenCV DNN fallback for very low-power scenarios
- `RTDETRDetector` — Transformer-based detector (high accuracy, higher compute)
- `CoralDetector` / `HailoDetector` — Hardware accelerator wrappers

**Key decisions:**
- YOLO11n selected over MobileNet SSD for superior small-object detection
- ONNX Runtime chosen over TensorFlow Lite for broader model format support
- INT8 quantization critical for hitting 5-10 FPS target on CPU

### 2. Tracker Module (`modules/tracker/`)

Maintains temporal consistency of detected advertisements across frames.

**Components:**
- `BaseTracker` — Abstract interface
- `ByteTrack` — Default, IoU-based matching, no appearance features
- `OCSort` — Observation-centric, better occlusion handling
- `BOTSort` — ByteTrack + ReID features

**Rationale:** ByteTrack selected as default because advertisements have
distinct positions — no appearance-based re-identification needed.

### 3. Masking Module (`modules/masking/`)

Applies visual modifications to detected regions.

**Components:**
- `BlackMasker` — Solid color fill (default black)
- `BlurMasker` — Gaussian blur (softer alternative)
- `PixelateMasker` — Mosaic/pixelation effect
- `CustomMasker` — User-supplied callback function

### 4. Video I/O Module (`modules/video_io/`)

Handles reading from and writing to video sources.

**Components:**
- `OpenCVReader` — Universal video/camera reader
- `Picamera2Reader` — Raspberry Pi camera module
- `VideoWriter` — Output to MP4/AVI files
- `show_preview()` — Real-time display with keyboard controls

### 5. Evaluation Module (`modules/evaluation/`)

Measures detection accuracy and system performance.

**Metrics:**
- Precision, Recall, F1-score
- mAP50 (mean Average Precision at IoU=0.5)
- FPS, latency (ms), CPU %, memory (MB)

### 6. Deployment Module (`modules/deployment/`)

Utilities for edge deployment.

**Features:**
- ONNX export from PyTorch
- INT8 quantization pipeline
- Raspberry Pi detection and configuration
- Accelerator detection (Coral, Hailo)
- Performance benchmarking

## Data Flow

1. **Frame acquisition**: Reader captures frame from file or camera
2. **Preprocessing**: Letterbox resize to 640×640, normalization
3. **Detection**: ONNX Runtime inference → bounding boxes + class IDs
4. **Post-processing**: Filter by ad class IDs, confidence threshold, NMS
5. **Tracking**: IoU-based association with existing tracks → stable IDs
6. **Masking**: Apply configured mask (black/blur/pixelate) to each bbox
7. **Output**: Write masked frame to file and/or display preview

## Configuration System

YAML-based configuration with Pydantic validation:

```yaml
detector:
  model_path: "data/models/yolo11n_ad.onnx"
  confidence_threshold: 0.35
  accelerator: "none"  # none | coral | hailo

tracker:
  type: "bytetrack"    # bytetrack | ocsort | botsort | none

masking:
  mode: "black"        # black | blur | pixelate | custom
```

All config fields are type-validated with sensible defaults.

## Edge Optimization Strategy

1. **INT8 Quantization**: 4x smaller model, 2-3x faster inference
2. **ONNX Runtime Session Config**: 
   - `ORT_ENABLE_ALL` graph optimization
   - 4 intra-op threads (quad-core RPi 5)
   - 2 inter-op threads
3. **Letterbox preprocessing**: Minimizes padding overhead
4. **ByteTrack**: No appearance features → minimal compute per frame
5. **640×480 resolution**: Best quality/speed tradeoff
6. **Skip frame tracking**: Configurable frame stride for further speedup

## Future Enhancements

- TensorRT backend for NVIDIA Jetson
- Optical flow-based temporal smoothing
- Active learning pipeline for rare ad types
- Web dashboard for remote monitoring
- RTSP/streaming support