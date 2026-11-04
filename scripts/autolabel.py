"""Auto-label ad segmentation with a YOLOE teacher.

    python scripts/autolabel.py --video data/outputs/test_ad_video.mp4 \
        --out data/dataset_ads --every 15 --imgsz 640 --conf 0.10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def polygon_to_label(poly: np.ndarray, w: int, h: int) -> str:
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return ""
    pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
    if len(pts) > 200:
        step = int(np.ceil(len(pts) / 200))
        pts = pts[::step]
    return "0 " + " ".join(f"{v:.5f}" for v in pts.reshape(-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/dataset_ads")
    ap.add_argument("--model", default="yoloe-11s-seg.pt")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument(
        "--prompts",
        nargs="*",
        default=["billboard", "advertisement poster", "store sign", "digital screen"],
    )
    args = ap.parse_args()

    from ultralytics import YOLOE

    model = YOLOE(args.model)
    model.set_classes(args.prompts, model.get_text_pe(args.prompts))

    root = Path(args.out)
    dirs = {
        "img_tr": root / "images/train",
        "img_va": root / "images/val",
        "lab_tr": root / "labels/train",
        "lab_va": root / "labels/val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seen = 0
    kept_tr = kept_va = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if seen < args.start:
            seen += 1
            continue
        if seen % max(args.every, 1) != 0:
            seen += 1
            continue
        seen += 1
        if args.max_frames and processed >= args.max_frames:
            break
        processed += 1

        res = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, agnostic_nms=True, verbose=False
        )[0]
        if res.masks is None or res.boxes is None or not len(res.boxes):
            continue
        h, w = frame.shape[:2]
        lines = [polygon_to_label(p, w, h) for p in res.masks.xy]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue

        use_val = kept_tr + kept_va > 0 and (kept_tr + kept_va) % args.val_every == 0
        name = f"frame_{seen:06d}"
        img_dir = dirs["img_va"] if use_val else dirs["img_tr"]
        lab_dir = dirs["lab_va"] if use_val else dirs["lab_tr"]
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        (lab_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        if use_val:
            kept_va += 1
        else:
            kept_tr += 1
        if (kept_tr + kept_va) % 25 == 0:
            print(f"labelled {kept_tr + kept_va} (train {kept_tr} val {kept_va})", flush=True)

    cap.release()
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: ad\n"
    )
    print(f"done: train={kept_tr} val={kept_va} of {total} frames -> {yaml_path}")


if __name__ == "__main__":
    main()
