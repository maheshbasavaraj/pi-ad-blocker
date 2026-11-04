"""Export the trained ad segmenter to ONNX (FP32 + INT8) and measure it.

    python scripts/export_ads.py --weights runs/seg/ads_nano_seg/weights/best.pt \
        --imgsz 640 --video data/outputs/test_ad_video.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/seg/ads_nano_seg/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--video", default="data/outputs/test_ad_video.mp4")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out-dir", default="data/models")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    fp32 = model.export(
        format="onnx", imgsz=args.imgsz, opset=18, simplify=True, half=False
    )
    fp32_path = Path(fp32)
    print(f"fp32 onnx: {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")

    int8_path = out_dir / (fp32_path.stem + "_int8.onnx")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        print(f"int8 onnx: {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB)")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"int8 quantization skipped: {exc}")
        int8_path = None

    frames = []
    cap = cv2.VideoCapture(args.video)
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    import onnxruntime as ort

    for path in [fp32_path] + ([int8_path] if int8_path else []):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape  # [1, 3, H, W]
        h, w = int(shape[2]), int(shape[3])
        t0 = time.perf_counter()
        runs = 0
        for f in frames:
            img = cv2.resize(f, (w, h))
            blob = cv2.dnn.blobFromImage(
                img, 1 / 255.0, (w, h), swapRB=True, crop=False
            ).astype(np.float32)
            sess.run(None, {name: blob})
            runs += 1
        dt = time.perf_counter() - t0
        print(
            f"{path.name}: {w}x{h} {runs} runs, {dt / runs * 1000:.1f} ms/frame, "
            f"{runs / dt:.2f} fps"
        )


if __name__ == "__main__":
    main()
