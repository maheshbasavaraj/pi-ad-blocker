"""Static (QDQ) INT8 quantization for the ad segmenter, with calibration.

`quantize_dynamic` was measured as a dead end on this net (225 ms/frame vs
33 ms FP32): it leaves activations in float and inserts dynamic weight
quantization on the fly, which costs more than it saves on a small conv net.
Static QDQ pre-computes activation ranges from a calibration set, so the
runtime graph is all-integer.

    python scripts/quantize_static.py \
        --model runs/seg/ads_nano_seg/weights/best.onnx \
        --calib data/dataset_ads/images/train --n-calib 120 \
        --out data/models/ads_nano_seg_int8_static.onnx

Then verify parity against FP32 (raw latency + mask IoU):

    python scripts/quantize_static.py --verify-only \
        --model runs/seg/ads_nano_seg/weights/best.onnx \
        --int8 data/models/ads_nano_seg_int8_static.onnx \
        --video data/outputs/test_ad_video.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def letterbox_blob(image: np.ndarray, w: int, h: int) -> np.ndarray:
    """Same preprocessing as ONNXSegDetector: letterbox 114, BGR→RGB, /255."""
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None])


class CalibReader:
    """Feeds letterboxed calibration images to onnxruntime's calibrator."""

    def __init__(self, files: list[Path], w: int, h: int, name: str = "images") -> None:
        import onnxruntime.quantization as q

        self._q = q
        self.files = files
        self.w, self.h, self.name = w, h, name
        self.reset()

    def reset(self) -> None:
        self._i = 0

    def get_next(self):
        if self._i >= len(self.files):
            return None
        img = cv2.imread(str(self.files[self._i]))
        self._i += 1
        if img is None:
            return self.get_next()
        return {self.name: letterbox_blob(img, self.w, self.h)}

    def __len__(self) -> int:
        return len(self.files)


def _head_nodes(model: Path, prefix: str) -> list[str]:
    """Names of the detection-head nodes (Ultralytics: everything under /model.23).

    The head decodes boxes, class scores and 32 mask coefficients from siamese
    conv branches. Its outputs span 0..1 (sigmoid) and 0..640 (decoded boxes)
    in the same tensors, so a single per-tensor activation scale collapses the
    class scores to zero — measured: every score became exactly 0.0 after
    calibration. Keeping the head in FP32 costs ~15% of a nano model and
    removes the whole failure class.
    """
    import onnx

    g = onnx.load(str(model)).graph
    names = [n.name for n in g.node if n.name.startswith(prefix)]
    return names


def calibrate(model: Path, files: list[Path], out: Path, n: int,
              exclude_head: bool = True, quant_format: str = "qdq") -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,  # noqa: F401  (kept for docs/typing)
        QuantFormat,
        QuantType,
        quantize_static,
    )

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    name = sess.get_inputs()[0].name
    h, w = int(shape[2]), int(shape[3])
    files = files[:n] if n else files
    if not files:
        raise SystemExit("no calibration images found")

    exclude = _head_nodes(model, "/model.23") if exclude_head else []
    qfmt = QuantFormat.QOperator if quant_format == "qoperator" else QuantFormat.QDQ
    print(f"calibrating {model.name} ({w}x{h}) on {len(files)} images, "
          f"format={quant_format}, {len(exclude)} head nodes kept in FP32",
          flush=True)
    reader = CalibReader(files, w, h, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=qfmt,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        op_types_to_quantize=["Conv", "MatMul"],
        nodes_to_exclude=exclude,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)", flush=True)
    return out


def _masks(sess, blob: np.ndarray, det) -> list[tuple[int, np.ndarray]]:
    """Run one detector pass and return (area_px, bbox) per kept detection."""
    from pi_ad_blocker.config import DetectorConfig

    name = sess.get_inputs()[0].name
    raw = sess.run(None, {name: blob})
    out = det.postprocess(raw, blob.shape[2:])
    return [(d.width * d.height, (d.x1, d.y1, d.x2, d.y2)) for d in out]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def verify(fp32: Path, int8: Path, video: Path, frames: int, conf: float) -> dict:
    import onnxruntime as ort

    from pi_ad_blocker.config import DetectorConfig
    from pi_ad_blocker.modules.detector.seg_detector import ONNXSegDetector

    cap = cv2.VideoCapture(str(video))
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    cap.release()
    if not imgs:
        raise SystemExit(f"no frames read from {video}")

    def build(path: Path):
        cfg = DetectorConfig(model_path=str(path), model_type="onnx_seg",
                             confidence_threshold=conf)
        det = ONNXSegDetector(cfg)
        det.load_model()
        return det

    det_fp, det_i8 = build(fp32), build(int8)
    sess_fp = det_fp._session
    sess_i8 = det_i8._session
    h, w = det_fp._in_h, det_fp._in_w

    blobs = [letterbox_blob(f, w, h) for f in imgs]

    t0 = time.perf_counter()
    res_fp = [_masks(sess_fp, b, det_fp) for b in blobs]
    ms_fp = (time.perf_counter() - t0) / len(blobs) * 1000

    t0 = time.perf_counter()
    res_i8 = [_masks(sess_i8, b, det_i8) for b in blobs]
    ms_i8 = (time.perf_counter() - t0) / len(blobs) * 1000

    ious, count_fp, count_i8 = [], 0, 0
    for a, b in zip(res_fp, res_i8):
        count_fp += len(a)
        count_i8 += len(b)
        for _, box_a in a:
            best = max((_iou(box_a, box_b) for _, box_b in b), default=0.0)
            ious.append(best)

    arr = np.array(ious) if ious else np.array([0.0])
    return {
        "fp32_ms": round(ms_fp, 1),
        "int8_ms": round(ms_i8, 1),
        "speedup": round(ms_fp / ms_i8, 2) if ms_i8 else 0.0,
        "det_fp32": count_fp,
        "det_int8": count_i8,
        "box_iou_mean": round(float(arr.mean()), 3),
        "box_iou_p10": round(float(np.percentile(arr, 10)), 3),
        "frames": len(imgs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/seg/ads_nano_seg/weights/best.onnx")
    ap.add_argument("--calib", default="data/dataset_ads/images/train")
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--out", default="data/models/ads_nano_seg_int8_static.onnx")
    ap.add_argument("--quant-format", default="qdq", choices=["qdq", "qoperator"])
    ap.add_argument("--keep-head-fp32", action="store_true", default=True)
    ap.add_argument("--quantize-head", dest="keep_head_fp32", action="store_false")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--int8", default="")
    ap.add_argument("--video", default="data/outputs/test_ad_video.mp4")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    fp32 = Path(args.model)
    if not fp32.exists():
        raise SystemExit(f"FP32 model not found: {fp32}")

    int8 = Path(args.int8) if args.int8 else Path(args.out)
    if not args.verify_only:
        files = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in Path(args.calib).glob(ext)
        )
        int8 = calibrate(fp32, files, int8, args.n_calib,
                         exclude_head=args.keep_head_fp32,
                         quant_format=args.quant_format)

    if int8.exists():
        rep = verify(fp32, int8, Path(args.video), args.frames, args.conf)
        print("\n| model | ms/frame | detections | box IoU mean | box IoU p10 |")
        print("|---|---|---|---|---|")
        print(f"| FP32 | {rep['fp32_ms']} | {rep['det_fp32']} | — | — |")
        print(f"| INT8 static | {rep['int8_ms']} | {rep['det_int8']} | "
              f"{rep['box_iou_mean']} | {rep['box_iou_p10']} |")
        print(f"\nspeedup: {rep['speedup']}x over {rep['frames']} frames")


if __name__ == "__main__":
    main()
