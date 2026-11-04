"""Fine-tune a nano segmentation model on YOLOE-labelled ad frames.

    python scripts/train_ads.py --data data/dataset_ads/data.yaml \
        --epochs 12 --imgsz 512 --batch 8
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_ads/data.yaml")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="ads_nano_seg")
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(Path("runs/seg").resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
        single_cls=True,
        verbose=True,
        plots=False,
    )


if __name__ == "__main__":
    main()
