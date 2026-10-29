#!/usr/bin/env python3
"""Download pre-trained models for Pi Ad Blocker.

Downloads YOLO models and exports to ONNX with optional INT8 quantization.
"""

import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

MODELS = {
    "yolo11n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "description": "YOLO11 nano — fastest, best for RPi 5 CPU mode",
        "size_mb": 5.5,
    },
    "yolo11n-int8": {
        "url": None,  # Generated via quantization
        "description": "YOLO11n INT8 quantized ONNX — 2-3x faster than FP32",
        "size_mb": 3.0,
    },
    "yolov8n": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "description": "YOLOv8 nano — good baseline",
        "size_mb": 6.3,
    },
}


def download_pt_model(name: str) -> Path:
    """Download a .pt model from Ultralytics assets."""
    import urllib.request

    info = MODELS[name]
    if info["url"] is None:
        raise ValueError(f"No direct URL for {name} — use export command instead")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / f"{name}.pt"

    if output_path.exists():
        print(f"[✓] Already exists: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    print(f"[ ] Downloading {info['url']} → {output_path}")
    urllib.request.urlretrieve(info["url"], output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"[✓] Downloaded: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_to_onnx(pt_path: Path) -> Path:
    """Export .pt to ONNX using Ultralytics."""
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[✓] ONNX already exists: {onnx_path}")
        return onnx_path

    print(f"[ ] Exporting {pt_path} → ONNX...")
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=640, simplify=True, opset=18)
    print(f"[✓] Exported: {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Download models for Pi Ad Blocker")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="yolo11n", help="Model to download")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export to ONNX after download")
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        if model_name.endswith("-int8"):
            print(f"[ ] Skipping {model_name} (generated via quantization)")
            continue
        pt_path = download_pt_model(model_name)
        if args.export_onnx:
            export_to_onnx(pt_path)

    print("\nDone! Models are in:", MODELS_DIR)


if __name__ == "__main__":
    main()