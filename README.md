# Pi Ad Blocker 🚫📺

**Raspberry Pi Advertisement Blocking System**

Detect advertisements — billboards, posters, digital screens, shop signs, roadside hoardings — in video streams and mask them automatically. Runs in real-time on **Raspberry Pi 5** (5-10 FPS CPU, 20-30 FPS with accelerator).

## Features

- **Multi-detector support**: YOLO11n (default), YOLOv8n, MobileNet SSD, RT-DETR
- **Real-time on RPi 5**: 5-10 FPS CPU-only, 20-30 FPS with Coral/Hailo accelerator
- **Tracking**: ByteTrack (default), OC-SORT, BOT-SORT for flicker-free masking
- **Masking modes**: Solid black, Gaussian blur, pixelation, custom callbacks
- **ONNX Runtime**: Optimized with INT8 quantization for edge deployment
- **Video sources**: Offline files, USB cameras, Raspberry Pi camera module (Picamera2)
- **Resolutions**: 480p (mandatory), 1080p (desirable)
- **Modular architecture**: Pluggable detector, tracker, masker pipeline
- **Evaluation pipeline**: Precision, recall, mAP, FPS, latency, resource usage
- **CLI + Python API**: Use as command-line tool or library

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/nousresearch/pi-ad-blocker.git
cd pi-ad-blocker
pip install -e .

# Or via pip (once published)
pip install pi-ad-blocker
```

### Download a model

```bash
# Download YOLO11n and export to ONNX
python scripts/download_models.py --model yolo11n --export-onnx
```

### Run on a video

```bash
# Process a video file
pi-ad-blocker run --video my_video.mp4 --output masked_output.mp4

# Or via Python API
python -c "
from pi_ad_blocker.pipeline import process_video
from pi_ad_blocker.config import AppConfig

config = AppConfig()
stats = process_video(config, 'input.mp4', 'output.mp4')
print(f'Processed {stats[\"frames_processed\"]} frames at {stats[\"avg_fps\"]:.1f} FPS')
"
```

### Run on live camera

```bash
# Using first camera
pi-ad-blocker run --camera 0 --output camera_output.mp4
```

### Real-time / AR mode (glasses, 30–60 fps display loop)

Detection runs on a background thread; every render frame propagates the last
polygons with optical flow, so the display loop is decoupled from detection
latency. Measured on laptop CPU (150 frames, 854×480): **35–39 fps render** with
a fine-tuned `yolo11n-seg` at **15.2 fps (640)** / **23.5 fps (480)** detection,
mask IoU p10 **0.78–0.80** (was 0.07 with the zero-shot YOLOE teacher). Full
ledger: `docs/realtime_mask_quality.md`.

CLI:

```bash
# paced to source fps, no preview window, first 300 frames
pi-ad-blocker run --video clip.mp4 --realtime --no-preview --max-frames 300

# live camera, AR compositor output (mask only, transparent elsewhere)
pi-ad-blocker run --camera 0 --realtime --display-mode black
```

```python
from pi_ad_blocker.config import AppConfig, DetectorConfig, MaskingConfig, RealtimeConfig
from pi_ad_blocker.realtime import RealtimeAdBlocker
import cv2

cfg = AppConfig(
    detector=DetectorConfig(
        model_path="runs/seg/ads_nano_seg/weights/best.pt",
        model_type="yolo_seg",          # ultralytics .pt seg model
        confidence_threshold=0.25,
        input_width=640,
        input_height=640,
    ),
    masking=MaskingConfig(mode="black", use_polygon=True, edge_feather_px=2,
                          output_mode="overlay"),   # "overlay" = AR compositor frame
    realtime=RealtimeConfig(async_detection=True, max_propagation_frames=15,
                            miss_tolerance=2),
)
blocker = RealtimeAdBlocker(cfg)
blocker.start()                          # also warms the model
cap = cv2.VideoCapture(0)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    out = blocker.process_frame(frame, i)
    i += 1
blocker.stop()
```

Render a masked file and measure mask quality:

```bash
python scripts/render_masked.py --video in.mp4 --frames 150 \
    --model runs/seg/ads_nano_seg/weights/best.pt --model-type yolo_seg \
    --imgsz 640 --conf 0.25 --out data/outputs/masked.mp4
python scripts/mask_quality.py --source in.mp4 --masked data/outputs/masked.mp4
```

### Train a domain-specific ad segmenter (teacher → student)

Open-vocabulary YOLOE labels the footage, a nano model learns it and ships:

```bash
python scripts/autolabel.py --video data/outputs/test_ad_video.mp4 \
    --out data/dataset_ads --every 10 --imgsz 640 --conf 0.10
python scripts/train_ads.py --data data/dataset_ads/data.yaml \
    --epochs 12 --imgsz 512 --batch 8
```

## Architecture

```
                    ┌─────────────┐
                    │  Video I/O  │
                    │  (Reader)   │
                    └──────┬──────┘
                           │ Frame
                           ▼
                    ┌─────────────┐
                    │   Detector  │ ◄── YOLO11n ONNX (INT8)
                    │  (ONNX RT)  │
                    └──────┬──────┘
                           │ Detections
                           ▼
                    ┌─────────────┐
                    │   Tracker   │ ◄── ByteTrack (optional)
                    │  (ByteTrack)│
                    └──────┬──────┘
                           │ Tracks
                           ▼
                    ┌─────────────┐
                    │   Masker    │ ◄── Black / Blur / Pixelate
                    │  (Config)   │
                    └──────┬──────┘
                           │ Masked frame
                           ▼
                    ┌─────────────┐
                    │  Video I/O  │
                    │  (Writer)   │
                    └─────────────┘
```

## Directory Structure

```
pi-ad-blocker/
├── src/
│   └── pi_ad_blocker/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point
│       ├── config.py           # Pydantic configuration
│       ├── pipeline.py         # Main processing pipeline
│       ├── training.py         # YOLO training pipeline
│       └── modules/
│           ├── detector/       # YOLO, MobileNet, RT-DETR, Coral, Hailo
│           ├── tracker/        # ByteTrack, OC-SORT, BOT-SORT
│           ├── masking/        # Black, blur, pixelate, custom
│           ├── video_io/       # OpenCV, Picamera2, writer
│           ├── evaluation/     # Metrics, reports, benchmarks
│           └── deployment/     # ONNX export, quantization, platform detection
├── configs/
│   └── default.yaml           # Default configuration
├── data/
│   ├── models/                 # Model weights (.pt, .onnx)
│   ├── datasets/               # Training/validation data
│   └── outputs/                # Processed videos, evaluation reports
├── scripts/
│   ├── download_models.py      # Download pretrained models
│   └── benchmark.py            # Performance benchmarks
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── docs/                       # Documentation
├── Dockerfile                  # Multi-arch Docker build
├── pyproject.toml              # Project metadata
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Detector Selection

| Detector | RPi 5 CPU FPS | RPi 5+Accel FPS | Accuracy | Size | Recommendation |
|----------|:---:|:---:|:---:|:---:|:---:|
| **YOLO11n** (INT8) | **8-12** | **25-35** | ★★★★☆ | 3 MB | ✅ **Default — best balance** |
| YOLO11n (FP32) | 5-8 | 20-30 | ★★★★☆ | 6 MB | Good fallback |
| YOLOv8n (INT8) | 7-10 | 20-28 | ★★★☆☆ | 3 MB | Legacy support |
| MobileNet SSD v2 | 10-15 | — | ★★☆☆☆ | 3 MB | Lightweight baseline |
| RT-DETR-Small | 2-4 | 10-15 | ★★★★★ | 8 MB | High accuracy only |

**Default choice: YOLO11n INT8 ONNX** — Best accuracy/speed tradeoff on RPi 5.

## Tracking Selection

| Tracker | Pros | Cons | FPS Impact |
|---------|------|------|:---:|
| **ByteTrack** ✅ | Simple, fast, no ReID needed | Less robust to long occlusions | ~1% overhead |
| OC-SORT | Better under occlusion, non-linear motion | Higher compute | ~3% overhead |
| BOT-SORT | Best ID stability, re-identification | Requires ReID model | ~10% overhead |

**Default: ByteTrack** — Best for advertisement blocking (no appearance features needed).

## Masking Modes

```
--mask black     █████████  → Solid black (default, zero cognitive load)
--mask blur      ░░░░░░░░░  → Gaussian blur (softer)
--mask pixelate  █░█░█░█░█  → Mosaic effect 
--mask custom    (user fn)  → Any OpenCV operation
```

## Raspberry Pi 5 Deployment

### Step 1: Install Raspberry Pi OS

```bash
# Use Raspberry Pi OS Lite (64-bit) or Desktop
# https://www.raspberrypi.com/software/
```

### Step 2: Install dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libgl1-mesa-glx
pip install pi-ad-blocker

# Optional: Raspberry Pi camera
sudo apt install -y python3-picamera2
pip install pi-ad-blocker[webcam]
```

### Step 3: Download model and run

```bash
pi-ad-blocker install
python scripts/download_models.py --model yolo11n --export-onnx

# Run on camera
pi-ad-blocker run --camera 0
```

### Step 4 (Optional): Enable hardware accelerators

**Coral TPU:**
```bash
pip install pi-ad-blocker[tpu]
# Edit config: detector.accelerator = "coral"
```

**Hailo-8:**
```bash
pip install pi-ad-blocker[hailo]
# Edit config: detector.accelerator = "hailo"
```

## Training a Custom Model

### 1. Prepare dataset

See [Dataset Strategy](#dataset-strategy) below for available datasets.

### 2. Create dataset YAML

```bash
python -c "
from pi_ad_blocker.training import create_dataset_yaml
create_dataset_yaml(
    train_path='data/datasets/ad_train/images',
    val_path='data/datasets/ad_val/images',
    class_names=['billboard', 'poster', 'sign', 'digital_ad', 'shop_sign', 'banner'],
    output_path='data/datasets/ad_dataset.yaml'
)
"
```

### 3. Train

```bash
pi-ad-blocker train \
    --data data/datasets/ad_dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --batch 16 \
    --device cpu
```

### 4. Export and deploy

```bash
pi-ad-blocker export --model ad_blocker_training/yolo11n_ad/weights/best.pt --format int8
```

## Dataset Strategy

### Existing Datasets

| Dataset | Billboards | Advertisements | Notes |
|---------|:---:|:---:|:---:|
| **Open Images v7** | ✓ | ✓ | 600+ billboard/sign annotations |
| **Mapillary Vistas** | ✓ | ✓ | 147+ sign classes, 25K images |
| **BDD100K** | ✓ | ~ | Traffic signs, billboards visible |
| **MTSD (Mapillary Traffic Sign)** | ✓ | ✗ | 100K+ signs, 313 classes |
| **Google Open Buildings** | ✗ | ✗ | Not applicable |

**Recommendation**: Use **Open Images v7** + **Mapillary Vistas** as primary sources. Combined they provide 800+ annotated advertisement-related objects. Additional Roboflow public datasets provide billboard-specific annotations.

### If Custom Labeling Required

1. **Collection**: Capture 5,000+ diverse street-scene images from multiple cities
2. **Annotation schema**:
   ```yaml
   classes:
     - billboard: Large roadside advertisement structures
     - poster: Paper/posted advertisements
     - digital_ad: Electronic/LED display ads
     - shop_sign: Storefront promotional signage
     - banner: Temporary hanging banners
     - hoarding: Construction site advertisement panels
     - transit_ad: Advertisements on vehicles/buses
   ```
3. **Tools**: Label Studio (free), CVAT (free), or Roboflow (cloud)

## Performance Targets

| Target | CPU Mode | +Coral TPU | +Hailo-8 |
|--------|:---:|:---:|:---:|
| 480p FPS | 5-10 ✅ | 20-30 ✅ | 25-35 ✅ |
| 1080p FPS | 2-4 ⚠️ | 8-12 ✅ | 10-15 ✅ |
| Latency (480p) | <150ms ✅ | <50ms ✅ | <40ms ✅ |
| Power | 5-7W | 7-9W | 7-9W |

## CLI Reference

```bash
# Run pipeline
pi-ad-blocker run --video <path> --output <path> [--config <path>]
pi-ad-blocker run --camera <id> [--output <path>]

# Train model
pi-ad-blocker train --data <yaml> --model <name> [options]

# Evaluate
pi-ad-blocker evaluate --video <path> --model <path> [--gt <path>]

# Benchmark
pi-ad-blocker benchmark --video <path> --model <path>

# Export model
pi-ad-blocker export --model <path> --format onnx|int8

# Manage config
pi-ad-blocker config --show
pi-ad-blocker config --edit

# Install dependencies
pi-ad-blocker install [--accelerator coral|hailo]
```

## Python API

```python
from pi_ad_blocker.config import AppConfig
from pi_ad_blocker.pipeline import AdBlockPipeline

# Configure
config = AppConfig()
config.detector.confidence_threshold = 0.4
config.masking.mode = "blur"
config.video_io.input_source = "video.mp4"
config.video_io.output_path = "output.mp4"

# Run
pipeline = AdBlockPipeline(config)
stats = pipeline.run()
print(f"FPS: {stats['avg_fps']}")

# Programmatic frame-by-frame
reader = pipeline.reader
reader.open()
for frame in reader:
    detections = pipeline.detector.detect(frame.data)
    tracks = pipeline.tracker.update(detections)
    masked = pipeline.masker.apply(frame.data, detections)
    # ... your custom logic
reader.close()
```

## Model Optimization

To export and quantize for Raspberry Pi:

```bash
# 1. Train or download .pt model
# 2. Export to ONNX
pi-ad-blocker export --model yolo11n.pt --format onnx

# 3. Quantize to INT8
pi-ad-blocker export --model yolo11n.pt --format int8

# Results:
#   yolo11n.pt       →  5.5 MB (PyTorch)
#   yolo11n.onnx     →  6.2 MB (FP32 ONNX)
#   yolo11n_int8.onnx →  3.1 MB (INT8 ONNX) ✅
```

INT8 quantization typically provides:
- **2-3x** inference speedup
- **4x** smaller model size
- **<1%** accuracy degradation

## Evaluation & Benchmarks

Run the full evaluation pipeline:

```bash
# On a video file
pi-ad-blocker evaluate --video test_video.mp4 --model data/models/yolo11n_ad.onnx

# With custom benchmark script
python scripts/benchmark.py --video test_video.mp4 --output results.json
```

Output metrics: precision, recall, mAP50, FPS, latency (ms), CPU %, memory (MB)

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default detector | YOLO11n INT8 ONNX | Best accuracy/speed tradeoff on RPi 5 CPU |
| Default tracker | ByteTrack | Zero appearance overhead; ads don't need re-ID |
| Default mask | Solid black rectangle | Minimal cognitive load, lowest compute cost |
| Inference runtime | ONNX Runtime | Cross-platform, INT8 quantization, edge optimized |
| Config format | YAML + Pydantic | Type-safe, self-validating, human-editable |
| Video backend | OpenCV (primary) | Universal, Picamera2 for camera module |
| Training framework | Ultralytics YOLO | Mature, well-tested, easy export pipeline |
| Accelerator support | Coral + Hailo | Most common USB accelerators on RPi |
| Resolution target | 640×480 | Best balance of detection quality and FPS on RPi |

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{pi_ad_blocker_2025,
  title={Pi Ad Blocker: Raspberry Pi Advertisement Blocking System},
  author={Nous Research},
  year={2025},
  url={https://github.com/nousresearch/pi-ad-blocker}
}
```

---

*Built for Raspberry Pi 5. Detects and masks advertisements in real-time.*
