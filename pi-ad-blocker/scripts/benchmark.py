#!/usr/bin/env python3
"""Performance benchmark script for Pi Ad Blocker detectors.

Tests all available detectors on a given video and reports FPS,
latency, CPU and memory usage.
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pi_ad_blocker.modules.deployment import detect_available_accelerators, is_raspberry_pi


def benchmark_detector(model_path: str, video_path: str, label: str):
    """Benchmark a single detector model."""
    from pi_ad_blocker.config import DetectorConfig
    from pi_ad_blocker.modules.detector import create_detector
    import cv2
    import psutil

    process = psutil.Process()

    # Open video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"  [ERROR] No frames read from {video_path}")
        return None

    # Warmup
    config = DetectorConfig(model_path=model_path, use_quantized="int8" in model_path.lower())
    detector = create_detector(config)

    for _ in range(5):
        for f in frames[:10]:
            detector.detect(f)

    # Timed run
    detector.reset_timing()
    cpu_start = process.cpu_percent(interval=0.0)
    mem_start = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    for f in frames:
        detector.detect(f)

    elapsed = time.time() - t0
    cpu_usage = process.cpu_percent(interval=0.1)
    mem_usage = process.memory_info().rss / (1024 * 1024)

    fps = len(frames) / elapsed
    latency = detector.avg_inference_time_ms

    return {
        "detector": label,
        "model": model_path,
        "fps": round(fps, 2),
        "latency_ms": round(latency, 2),
        "cpu_percent": round(cpu_usage, 1),
        "memory_mb": round(mem_usage, 1),
        "frames": len(frames),
        "total_time_s": round(elapsed, 3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pi Ad Blocker Benchmark")
    parser.add_argument("--video", required=True, help="Video file for benchmarking")
    parser.add_argument("--models", nargs="+", default=[],
                        help="Specific model paths (default: auto-detect)")
    parser.add_argument("--output", help="JSON output path")
    args = parser.parse_args()

    print("=" * 60)
    print("Pi Ad Blocker — Performance Benchmark")
    print("=" * 60)

    # Platform info
    print(f"\nPlatform:")
    print(f"  Raspberry Pi: {is_raspberry_pi()}")
    print(f"  Accelerators: {detect_available_accelerators()}")
    print(f"  Video: {args.video}")

    # Find models
    models_dir = Path(__file__).resolve().parent.parent / "data" / "models"
    if args.models:
        model_paths = args.models
    else:
        model_paths = sorted([
            str(p) for p in models_dir.glob("*.*")
            if p.suffix in (".onnx", ".pt")
        ])

    if not model_paths:
        print("\n[!] No models found. Run: python scripts/download_models.py --export-onnx")
        sys.exit(1)

    results = []
    for mp in model_paths:
        label = Path(mp).name
        print(f"\nBenchmarking: {label}")
        print("-" * 40)
        try:
            result = benchmark_detector(mp, args.video, label)
            if result:
                results.append(result)
                print(f"  FPS:      {result['fps']}")
                print(f"  Latency:  {result['latency_ms']:.1f} ms")
                print(f"  CPU:      {result['cpu_percent']}%")
                print(f"  Memory:   {result['memory_mb']:.0f} MB")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["fps"], reverse=True):
        print(f"  {r['detector']:30s}  {r['fps']:>6.1f} FPS  {r['latency_ms']:>6.1f} ms")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()