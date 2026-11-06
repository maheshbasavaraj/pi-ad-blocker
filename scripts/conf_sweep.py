"""Sweep the confidence threshold and report precision / recall / F1.

The detector's conf threshold is the only precision knob the pipeline exposes,
and it was never chosen with numbers. This measures it against the labelled val
split: for each threshold, boxes are matched to ground truth at IoU 0.5 and the
counts are reported.

    python scripts/conf_sweep.py --model runs/seg/ads_nano_seg/weights/best.pt \
        --data data/dataset_ads/data.yaml --imgsz 640 \
        --confs 0.15 0.20 0.25 0.30 0.35 0.45 0.55

F1 is the shipping criterion: a missed billboard leaves an ad visible, a false
positive blackens something that is not an ad (the worse failure per the
glasses plan), so the chosen threshold is the smallest one within 2% of peak F1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_yolo_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """Read YOLO polygon labels → array of xyxy boxes in pixels."""
    if not label_path.exists():
        return np.zeros((0, 4), np.float32)
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        xy = np.array([float(v) for v in parts[1:]], np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
    return np.array(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/seg/ads_nano_seg/weights/best.pt")
    ap.add_argument("--data", default="data/dataset_ads/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--confs", type=float, nargs="+",
                    default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="NMS IoU; 0.7 (Ultralytics default) leaves ~18%% of "
                         "boxes overlapping each other at conf 0.25")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import yaml
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load(Path(args.data).read_text())
    base = Path(data.get("path", root))
    val_dir = base / data.get("val", "images/val")
    lbl_dir = Path(str(val_dir).replace("images", "labels"))

    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                    for p in val_dir.glob(ext))
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no validation images in {val_dir}")

    model = YOLO(args.model)
    # One low-conf pass per image; thresholds are applied to the raw results so
    # the sweep costs one inference per frame instead of one per threshold.
    conf_min = min(args.confs)
    preds, gts = [], []
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        r = model.predict(img, conf=conf_min, iou=args.nms_iou, imgsz=args.imgsz,
                          verbose=False, agnostic_nms=True)[0]
        boxes = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                 else np.zeros((0, 4), np.float32))
        confs = (r.boxes.conf.cpu().numpy() if r.boxes is not None
                 else np.zeros((0,), np.float32))
        preds.append((boxes, confs))
        gts.append(_load_yolo_gt(lbl_dir / (img_path.stem + ".txt"), w, h))

    print(f"val images: {len(images)}  gt boxes: {sum(len(g) for g in gts)}")
    print("\n| conf | preds | TP | FP | FN | precision | recall | F1 |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for conf in args.confs:
        tp = fp = fn = n_pred = 0
        for (boxes, confs), gt in zip(preds, gts):
            keep = confs >= conf
            pb = boxes[keep]
            n_pred += len(pb)
            if len(gt) == 0:
                fp += len(pb)
                continue
            if len(pb) == 0:
                fn += len(gt)
                continue
            ious = _iou(pb, gt)
            matched_gt: set[int] = set()
            for pi in np.argsort(-ious.max(axis=1)):
                gi = int(np.argmax(ious[pi]))
                if ious[pi, gi] >= args.iou and gi not in matched_gt:
                    matched_gt.add(gi)
                    tp += 1
                else:
                    fp += 1
            fn += len(gt) - len(matched_gt)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append((conf, n_pred, tp, fp, fn, p, r, f1))
        print(f"| {conf:.2f} | {n_pred} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f1:.3f} |")

    best = max(rows, key=lambda row: row[7])
    within = [row for row in rows if row[7] >= best[7] - 0.02]
    pick = min(within, key=lambda row: row[0])
    print(f"\npeak F1 {best[7]:.3f} at conf {best[0]:.2f}; "
          f"shipping pick (lowest conf within 2% of peak): {pick[0]:.2f} "
          f"(P {pick[5]:.3f}, R {pick[6]:.3f})")


if __name__ == "__main__":
    main()
