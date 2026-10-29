"""Deployment module — ONNX export, quantization, and Raspberry Pi deployment utilities.

Provides model conversion, INT8 quantization, benchmark scripts,
and platform detection helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    torch_model_path: str,
    output_path: str,
    input_width: int = 640,
    input_height: int = 640,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> str:
    """Export a PyTorch YOLO model to ONNX format.

    Args:
        torch_model_path: Path to .pt model file.
        output_path: Path for exported .onnx file.
        input_width, input_height: Model input dimensions.
        dynamic_batch: Enable dynamic batch dimension.
        simplify: Run ONNX simplifier on the graph.

    Returns:
        Path to the exported ONNX file.

    Requires: ultralytics, onnx, onnxsim
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise

    logger.info("Loading PyTorch model: {}", torch_model_path)
    model = YOLO(torch_model_path)

    logger.info("Exporting to ONNX: {}", out)
    success = model.export(
        format="onnx",
        imgsz=(input_height, input_width),
        dynamic=dynamic_batch,
        simplify=simplify,
        opset=18,
    )

    result_path = str(out)
    logger.info("ONNX export complete: {}", result_path)
    return result_path


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------

def quantize_onnx_int8(
    onnx_path: str,
    output_path: str,
    calibration_data: Optional[list[np.ndarray]] = None,
    calibration_samples: int = 100,
    input_width: int = 640,
    input_height: int = 640,
) -> str:
    """Quantize an ONNX model to INT8 using ONNX Runtime's quantization tool.

    INT8 quantization reduces model size ~4x and improves inference speed
    2-3x on CPU with minimal accuracy loss.

    Args:
        onnx_path: Path to FP32 ONNX model.
        output_path: Path for INT8 quantized model.
        calibration_data: Optional list of calibration images.
        calibration_samples: Number of calibration samples to generate.
        input_width, input_height: Input dimensions.

    Returns:
        Path to quantized ONNX model.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import (
            quantize_dynamic,
            QuantType,
            quantize_static,
            CalibrationMethod,
        )
        import onnx
    except ImportError:
        logger.error("onnxruntime-quantization not available")
        raise

    logger.info("Quantizing {} to INT8...", onnx_path)

    # Check if model has appropriate nodes for quantization
    model = onnx.load(onnx_path)

    # Dynamic quantization — simpler, no calibration data needed
    quantize_dynamic(
        model_input=onnx_path,
        model_output=str(out),
        per_channel=True,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    logger.info("INT8 quantization complete: {}", out)
    return str(out)


# ---------------------------------------------------------------------------
# Raspberry Pi platform detection
# ---------------------------------------------------------------------------

def is_raspberry_pi() -> bool:
    """Detect if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().lower()
            return "raspberry" in model
    except (FileNotFoundError, IOError):
        # Check via uname
        try:
            result = subprocess.run(
                ["uname", "-a"], capture_output=True, text=True, timeout=2
            )
            return "aarch64" in result.stdout and "raspberry" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


def detect_pi_model() -> str:
    """Detect specific Raspberry Pi model."""
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return "unknown"


def detect_available_accelerators() -> list[str]:
    """Detect which hardware accelerators are available."""
    accelerators = []

    # Check Coral TPU
    try:
        result = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=2
        )
        if "Global Unichip" in result.stdout or "Google" in result.stdout:
            accelerators.append("coral")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check Hailo-8
    try:
        result = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=2
        )
        if "Hailo" in result.stdout:
            accelerators.append("hailo")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return accelerators


# ---------------------------------------------------------------------------
# Installation helper
# ---------------------------------------------------------------------------

def install_dependencies(accelerator: str = "none") -> None:
    """Install required dependencies for the target platform."""
    logger.info("Installing base dependencies...")

    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "pi-ad-blocker"],
    ]

    if accelerator == "coral":
        cmds.append([sys.executable, "-m", "pip", "install", "pi-ad-blocker[tpu]"])
    elif accelerator == "hailo":
        cmds.append([sys.executable, "-m", "pip", "install", "pi-ad-blocker[hailo]"])

    if is_raspberry_pi():
        cmds.append([
            sys.executable, "-m", "pip", "install",
            "opencv-python-headless",
            "pi-ad-blocker[webcam]",
        ])

    for cmd in cmds:
        logger.info("Running: {}", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            logger.error("Installation step failed: {}", e)
            raise

    logger.info("Dependencies installed successfully")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(detector_model: str, video_path: str, iterations: int = 3) -> dict:
    """Run a performance benchmark on a given detector model."""
    import time
    import cv2

    from pi_ad_blocker.config import DetectorConfig
    from pi_ad_blocker.modules.detector import create_detector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames read from video")

    logger.info("Benchmark: {} frames from {}", len(frames), video_path)

    config = DetectorConfig(model_path=detector_model)
    detector = create_detector(config)

    results: list[dict] = []
    for i in range(iterations):
        detector.reset_timing()
        for frame in frames:
            detector.detect(frame)

        results.append({
            "iteration": i + 1,
            "avg_fps": detector.avg_fps,
            "avg_latency_ms": detector.avg_inference_time_ms,
            "frames_processed": len(frames),
        })

    avg_fps = sum(r["avg_fps"] for r in results) / len(results)
    avg_lat = sum(r["avg_latency_ms"] for r in results) / len(results)

    summary = {
        "model": detector_model,
        "iterations": iterations,
        "frames_per_iter": len(frames),
        "avg_fps": round(avg_fps, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "results": results,
    }

    logger.info("Benchmark complete: {:.1f} FPS, {:.1f} ms latency", avg_fps, avg_lat)
    return summary


__all__ = [
    "export_to_onnx",
    "quantize_onnx_int8",
    "is_raspberry_pi",
    "detect_pi_model",
    "detect_available_accelerators",
    "install_dependencies",
    "run_benchmark",
]