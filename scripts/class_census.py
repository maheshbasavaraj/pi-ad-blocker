"""Class census gate: what is the detector actually masking?

    python scripts/class_census.py --video data/outputs/test_ad_video.mp4 \
        --model runs/seg/ads_nano_seg/weights/best.pt --model-type yolo_seg --every 15

Prints a class -> detection-count table. A product build must show ad classes
only; pedestrians, cars or traffic lights mean over-masking.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_ad_blocker.config import DetectorConfig  # noqa: E402
from pi_ad_blocker.modules.detector import create_detector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="yolo_seg")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    cfg = DetectorConfig(
        model_path=args.model,
        model_type=args.model_type,
        confidence_threshold=args.conf,
        input_width=args.imgsz,
        input_height=args.imgsz,
        filter_to_ad_classes=False,
    )
    det = create_detector(cfg)

    census: Counter[str] = Counter()
    frames_seen = frames_scanned = 0
    cap = cv2.VideoCapture(args.video)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_seen += 1
        if frames_seen % max(args.every, 1):
            continue
        if args.frames and frames_scanned >= args.frames:
            break
        frames_scanned += 1
        for d in det.detect(frame):
            census[d.class_name] += 1
    cap.release()

    print(f"scanned {frames_scanned} frames (every {args.every} of {frames_seen})")
    if not census:
        print("census: EMPTY — detector found nothing")

    # A single-class model has no real class names; ultralytics reports "item".
    # Treat the only class of a single-class model as the ad class.
    model_names = getattr(det, "class_names", None) or []
    if len(model_names) == 1:
        allowed = {str(model_names[0])}
    else:
        allowed = set()

    for name, n in census.most_common():
        flag = "" if (name in allowed or name in ("ad", "billboard")) else "  <- non-ad"
        print(f"  {name:<24} {n}{flag}")
    non_ad = {
        k: v
        for k, v in census.items()
        if k not in allowed and k not in ("ad", "billboard")
    }
    if non_ad:
        print(f"GATE FAIL: non-ad classes present -> {non_ad}")
        raise SystemExit(1)
    print("GATE PASS: ad classes only")


if __name__ == "__main__":
    main()
