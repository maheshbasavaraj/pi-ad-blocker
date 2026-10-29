#!/usr/bin/env python3
"""Download detection models for Pi Ad Blocker.

Usage:
    python download_model.py            # downloads YOLO11n and exports to ONNX
    python download_model.py --model yolov8n
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Reuse the existing download script
from scripts.download_models import download_pt_model, export_to_onnx, MODELS


def download_model(name: str, export_onnx: bool = True) -> None:
    """Download a model and optionally export to ONNX."""
    pt_path = download_pt_model(name)
    if export_onnx:
        export_to_onnx(pt_path)


def main():
    parser = argparse.ArgumentParser(description="Download models for Pi Ad Blocker")
    parser.add_argument("--model", default="yolo11n", choices=list(MODELS.keys()),
                        help="Model to download (default: yolo11n)")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Export to ONNX format")
    parser.add_argument("--no-onnx", action="store_false", dest="export_onnx",
                        help="Skip ONNX export")
    args = parser.parse_args()

    print(f"Downloading {args.model}...")
    download_model(args.model, export_onnx=args.export_onnx)
    print("Done ✅")


if __name__ == "__main__":
    main()