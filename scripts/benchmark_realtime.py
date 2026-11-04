"""Benchmark real-time configurations on a video.

Examples
--------
    python scripts/benchmark_realtime.py --video data/outputs/test_ad_video.mp4
    python scripts/benchmark_realtime.py --video clip.mp4 --frames 600 --json out.json

The harness measures what matters for glasses: rendered FPS, detector cost,
per-frame latency, and how many frames actually carry a mask. It runs each
variant in the same process so the comparison shares the machine state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pi_ad_blocker.config import (  # noqa: E402
    AppConfig,
    DetectorConfig,
    MaskingConfig,
    RealtimeConfig,
    VideoIOConfig,
)
from pi_ad_blocker.realtime import RealtimeAdBlocker  # noqa: E402

DEFAULT_MODELS = {
    "bbox": ROOT / "data/models/yolo11n_ad.onnx",
    "seg": ROOT / "data/models/yolo11s-seg.onnx",
}


def _changed_ratio(original, masked) -> float:
    """Fraction of pixels the masker actually altered (not just dark video)."""
    diff = cv2.absdiff(original, masked).max(axis=2)
    return float((diff > 20).mean())


def run_variant(name, cfg, video, frames, pace=True):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    blocker = RealtimeAdBlocker(cfg)
    blocker.start()
    masked = 0
    t_start = time.perf_counter()
    done = 0
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        out = blocker.process_frame(frame, i)
        done += 1
        if _changed_ratio(frame, out) > 0.005:
            masked += 1
        if pace:
            target = done / src_fps
            elapsed = time.perf_counter() - t_start
            if elapsed < target:
                time.sleep(target - elapsed)
    elapsed = time.perf_counter() - t_start
    blocker.stop()
    stats = blocker.stats(elapsed, done)
    cap.release()
    stats["variant"] = name
    stats["masked_frame_ratio"] = round(masked / max(done, 1), 3)
    return stats


def build_variants(args):
    common_io = VideoIOConfig(enable_preview=False)
    variants = {}

    if DEFAULT_MODELS["bbox"].exists():
        variants["A_bbox_sequential"] = AppConfig(
            detector=DetectorConfig(
                model_path=str(DEFAULT_MODELS["bbox"]),
                model_type="yolo11",
                confidence_threshold=0.35,
            ),
            masking=MaskingConfig(mode="black", use_polygon=False),
            realtime=RealtimeConfig(async_detection=False, detect_interval=1),
            video_io=common_io,
        )

    if DEFAULT_MODELS["seg"].exists():
        # Throughput variant: every COCO class is masked, so this is a SPEED
        # ceiling only — a product build must filter to ad classes (variant D).
        variants["B_seg_async_all_classes"] = AppConfig(
            detector=DetectorConfig(
                model_path=str(DEFAULT_MODELS["seg"]),
                model_type="onnx_seg",
                confidence_threshold=args.conf,
                input_width=480,
                input_height=480,
                filter_to_ad_classes=False,
            ),
            masking=MaskingConfig(mode="black", use_polygon=True, edge_feather_px=2),
            realtime=RealtimeConfig(
                async_detection=True,
                detect_interval=args.detect_interval,
                max_propagation_frames=args.max_propagation,
            ),
            video_io=common_io,
        )

    variants["C_black_boxes_baseline"] = AppConfig(
        detector=DetectorConfig(
            model_path=str(DEFAULT_MODELS["bbox"]),
            model_type="yolo11",
            confidence_threshold=0.25,
        ),
        masking=MaskingConfig(mode="black", use_polygon=False),
        realtime=RealtimeConfig(async_detection=False, detect_interval=1),
        video_io=common_io,
    )

    if DEFAULT_MODELS["seg"].exists():
        variants["D_seg_async_adclass"] = AppConfig(
            detector=DetectorConfig(
                model_path=str(DEFAULT_MODELS["seg"]),
                model_type="onnx_seg",
                confidence_threshold=args.conf,
                input_width=480,
                input_height=480,
                filter_to_ad_classes=True,
            ),
            masking=MaskingConfig(mode="black", use_polygon=True, edge_feather_px=2),
            realtime=RealtimeConfig(
                async_detection=True,
                detect_interval=args.detect_interval,
                max_propagation_frames=args.max_propagation,
            ),
            video_io=common_io,
        )
    nano = Path("runs/seg/ads_nano_seg/weights/best.pt")
    if nano.exists():
        for tag, imgsz in (("E_nano_seg_640", 640), ("F_nano_seg_480", 480)):
            variants[tag] = AppConfig(
                detector=DetectorConfig(
                    model_path=str(nano),
                    model_type="yolo_seg",
                    confidence_threshold=args.conf,
                    input_width=imgsz,
                    input_height=imgsz,
                    filter_to_ad_classes=False,
                ),
                masking=MaskingConfig(
                    mode="black", use_polygon=True, edge_feather_px=2
                ),
                realtime=RealtimeConfig(
                    async_detection=True,
                    detect_interval=args.detect_interval,
                    max_propagation_frames=args.max_propagation,
                ),
                video_io=common_io,
            )
    onnx_nano = Path("runs/seg/ads_nano_seg/weights/best.onnx")
    if onnx_nano.exists():
        variants["G_onnx_nano_640"] = AppConfig(
            detector=DetectorConfig(
                model_path=str(onnx_nano),
                model_type="onnx_seg",
                confidence_threshold=args.conf,
                input_width=640,
                input_height=640,
                filter_to_ad_classes=False,
            ),
            masking=MaskingConfig(mode="black", use_polygon=True, edge_feather_px=2),
            realtime=RealtimeConfig(
                async_detection=True,
                detect_interval=args.detect_interval,
                max_propagation_frames=args.max_propagation,
            ),
            video_io=common_io,
        )

    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--detect-interval", type=int, default=8)
    ap.add_argument("--max-propagation", type=int, default=30)
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--only", nargs="*", default=None, help="Run only these variant names")
    args = ap.parse_args()

    rows = []
    variants = build_variants(args)
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}
    for name, cfg in variants.items():
        try:
            rows.append(run_variant(name, cfg, args.video, args.frames, not args.no_pace))
        except Exception as exc:  # a missing model must not kill the other rows
            rows.append({"variant": name, "error": str(exc)})
        print(f"done: {name}", flush=True)

    print("\n| variant | render fps | detector fps | frame ms p95 | masked frames |")
    print("|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['variant']} | error: {r['error'][:40]} | | | |")
            continue
        print(
            f"| {r['variant']} | {r['render_fps']} | {r['detector_fps']} | "
            f"{r['frame_ms_p95']} | {r['masked_frame_ratio']:.0%} |"
        )

    payload = {"video": args.video, "frames": args.frames, "results": rows}
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
