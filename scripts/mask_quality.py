"""Measure real mask quality from an original + masked video pair.

    python scripts/mask_quality.py --source orig.mp4 --masked out.mp4 [--csv q.csv]

Prints active frames, mask area percentiles, and consecutive-mask IoU
(mean and p10). p10 IoU is the flicker gate: < 0.85 means the mask blinks.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--masked", required=True)
    ap.add_argument("--thresh", type=int, default=20)
    ap.add_argument("--active", type=float, default=0.005)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    a = cv2.VideoCapture(args.source)
    b = cv2.VideoCapture(args.masked)
    areas, ious, rows = [], [], []
    prev = None
    idx = 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        m = (cv2.absdiff(f1, f2).max(axis=2) > args.thresh).astype(np.uint8)
        areas.append(float(m.mean()))
        iou = float("nan")
        if prev is not None and m.sum() > 200 and prev.sum() > 200:
            inter = float(np.logical_and(m, prev).sum())
            union = float(np.logical_or(m, prev).sum())
            iou = inter / max(union, 1.0)
            ious.append(iou)
        rows.append((idx, round(100 * areas[-1], 3), round(iou, 3)))
        prev = m
        idx += 1
    a.release()
    b.release()

    if not areas:
        raise SystemExit("no frames read — check paths")
    active = sum(1 for v in areas if v > args.active)
    print(f"frames={len(areas)} active={active} ({100.0 * active / len(areas):.1f}%)")
    print(
        "area_pct mean={:.2f} p50={:.2f} p95={:.2f} max={:.2f}".format(
            100 * float(np.mean(areas)),
            100 * float(np.percentile(areas, 50)),
            100 * float(np.percentile(areas, 95)),
            100 * float(np.max(areas)),
        )
    )
    if ious:
        print(
            "jitter_iou mean={:.3f} p10={:.3f} min={:.3f}".format(
                float(np.mean(ious)),
                float(np.percentile(ious, 10)),
                float(np.min(ious)),
            )
        )
        print("gate: p10 >= 0.85 passes; below = flicker")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("frame,area_pct,iou\n")
            fh.writelines(f"{i},{a_},{o}\n" for i, a_, o in rows)
        print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
