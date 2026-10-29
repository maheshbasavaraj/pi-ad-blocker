# Raspberry Pi 5 Deployment Guide

## Hardware Requirements

### Minimum
- Raspberry Pi 5 (4GB RAM recommended, 8GB preferred)
- 32GB+ microSD card (or USB SSD)
- USB camera or Raspberry Pi Camera Module 3

### Recommended for Real-Time
- Raspberry Pi 5 (8GB)
- USB SSD boot drive
- **Optional**: Coral TPU M.2 Accelerator or Hailo-8L NPU HAT
- Raspberry Pi Camera Module 3 (for camera mode)

### Networking (optional)
- Ethernet or Wi-Fi for downloading models and updates

## Step 1: Install Raspberry Pi OS

```bash
# Download Raspberry Pi Imager from:
# https://www.raspberrypi.com/software/

# Select:
#   Raspberry Pi Device: Raspberry Pi 5
#   Operating System: Raspberry Pi OS Lite (64-bit, Debian Bookworm)
#   Storage: your SD card or SSD

# Enable SSH, set username/password in advanced settings
```

## Step 2: Initial Setup

```bash
# Connect via SSH
ssh pi@raspberry-pi-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-venv \
    libgl1-mesa-glx \
    libgomp1 \
    cmake \
    build-essential

# Reboot
sudo reboot
```

## Step 3: Install Pi Ad Blocker

```bash
# Option A: From source (recommended for bleeding-edge)
cd ~
git clone https://github.com/nousresearch/pi-ad-blocker.git
cd pi-ad-blocker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with RPi optimizations
pip install --upgrade pip
pip install -e ".[webcam]"

# Option B: Via pip
# pip install pi-ad-blocker[webcam]
```

## Step 4: Download and Optimize Model

```bash
# Activate virtual environment if not already
source ~/pi-ad-blocker/venv/bin/activate

# Download YOLO11n
python scripts/download_models.py --model yolo11n

# Export to ONNX (FP32)
python scripts/download_models.py --model yolo11n --export-onnx

# Quantize to INT8 for 2-3x speedup
pi-ad-blocker export --model data/models/yolo11n.pt --format int8

# Verify model files
ls -lh data/models/
# Expected:
# -rw-r--r--  yolo11n.pt         (5.5 MB)  PyTorch
# -rw-r--r--  yolo11n.onnx       (6.2 MB)  FP32 ONNX
# -rw-r--r--  yolo11n_int8.onnx  (3.1 MB)  INT8 ONNX ✅
```

## Step 5: Configure for RPi 5

Edit `configs/default.yaml`:

```yaml
detector:
  model_path: "data/models/yolo11n_int8.onnx"  # Use INT8 model
  model_type: "yolo11"
  confidence_threshold: 0.35
  input_width: 640
  input_height: 640
  providers: ["CPUExecutionProvider"]
  use_quantized: true

video_io:
  input_source: "0"           # /dev/video0 for USB camera
  resize_width: 640
  resize_height: 480

masking:
  mode: "black"               # lowest CPU cost
```

## Step 6: Run

### Test with a sample video

```bash
# Download a test video or use your own
# Process video file
pi-ad-blocker run --video test_video.mp4 --output masked_result.mp4
```

### Run with USB camera

```bash
# Default camera
pi-ad-blocker run --camera 0

# With output recording
pi-ad-blocker run --camera 0 --output recorded_masked.mp4
```

### Run with Pi Camera Module

Update config:
```yaml
video_io:
  camera_backend: "picamera2"
  input_source: "0"
```

```bash
pi-ad-blocker run --camera 0
```

## Step 7 (Optional): Enable Hardware Accelerator

### Coral TPU (M.2 or USB)

```bash
# Install Coral TPU runtime
pip install pi-ad-blocker[tpu]

# Update config
# detector.accelerator: "coral"
# detector.model_path: "data/models/yolo11n_edgetpu.tflite"

# Re-run
pi-ad-blocker run --camera 0
```

### Hailo-8L NPU HAT

```bash
# Install Hailo software
# Follow: https://hailo.ai/developer-zone/

pip install pi-ad-blocker[hailo]

# Update config
# detector.accelerator: "hailo"
# detector.model_path: "data/models/yolo11n.hef"

# Re-run
pi-ad-blocker run --camera 0
```

## Step 8: Auto-Start at Boot

Create systemd service:

```bash
sudo nano /etc/systemd/system/pi-ad-blocker.service
```

```ini
[Unit]
Description=Pi Ad Blocker - Advertisement Masking Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pi-ad-blocker
ExecStart=/home/pi/pi-ad-blocker/venv/bin/pi-ad-blocker run --camera 0 --output /home/pi/recordings/$(date +%%Y%%m%%d_%%H%%M%%S).mp4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-ad-blocker
sudo systemctl start pi-ad-blocker
sudo systemctl status pi-ad-blocker
```

## Performance Verification

```bash
# Run benchmark
python scripts/benchmark.py --video test_video.mp4

# Expected output (CPU only, 480p):
#   YOLO11n INT8:    8-12 FPS ✅
#   YOLO11n FP32:    5-8 FPS
#   MobileNet SSD:   10-15 FPS

# With Coral TPU:
#   YOLO11n INT8:    20-30 FPS ✅

# With Hailo-8L:
#   YOLO11n:         25-35 FPS ✅
```

## Optimization Tips

1. **Use INT8 model**: 2-3x faster than FP32 with <1% accuracy loss
2. **Reduce resolution**: 480p (640×480) is optimal for RPi 5
3. **Disable preview**: `enable_preview: false` saves 2-3 FPS
4. **Skip tracking**: Set `tracker.type: "none"` for ~5% speedup
5. **Overclock**: RPi 5 can safely run at 2.8 GHz with adequate cooling
6. **USB SSD boot**: Much faster model loading than microSD
7. **Reduce confidence**: Lower from 0.35 to 0.25 for more detections (trade accuracy)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low FPS (<5) | Use INT8 model, reduce to 480p, disable preview |
| Camera not found | Check `ls /dev/video*`, try different index |
| ONNX Runtime error | `pip install --upgrade onnxruntime` |
| "Model not found" | Run `python scripts/download_models.py` |
| Picamera2 import error | `pip install pi-ad-blocker[webcam]` |
| Memory error | Use smaller model (yolo11n not yolo11m) |
| CUDA/GPU not needed | RPi has no GPU acceleration for ML |

## Power Requirements

- RPi 5 alone: 5V/3A (15W USB-C PD)
- +Coral TPU: Add 1.5W
- +Hailo-8L: Add 2.5W
- +Camera: Add 0.5W
- Total typical: 10-18W

## Heat Management

- RPi 5 CPU at full load: ~80°C without heatsink
- **Required**: Heatsink + fan case
- Throttling starts at 85°C
- Active cooler recommended for sustained 10+ FPS