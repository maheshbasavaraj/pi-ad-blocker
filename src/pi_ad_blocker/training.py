"""Training pipeline — fine-tune YOLO models on advertisement datasets.

Supports Ultralytics YOLO training with configurable hyperparameters,
dataset splits, augmentation, and validation. Designed for fine-tuning
on custom ad datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Dataset YAML generation
# ---------------------------------------------------------------------------

def create_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: Optional[str],
    class_names: list[str],
    output_path: str,
) -> str:
    """Create a YOLO-compatible dataset YAML file.

    Args:
        train_path: Path to training images directory.
        val_path: Path to validation images directory.
        test_path: Optional path to test images directory.
        class_names: List of class names (index = class_id).
        output_path: Path to save the dataset YAML.

    Returns:
        Path to the generated YAML file.
    """
    data = {
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }

    if test_path:
        data["test"] = test_path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    logger.info("Dataset YAML saved: {}", out)
    return str(out)


# ---------------------------------------------------------------------------
# Ad-specific dataset descriptions
# ---------------------------------------------------------------------------

AD_DATASET_CLASSES = {
    "advertisement": ["billboard", "poster", "signboard", "digital_display",
                       "shop_sign", "banner", "hoarding", "advertisement"],
    "default": ["billboard", "poster", "sign", "digital_ad", "shop_sign",
                 "banner", "hoarding", "advertisement"],
}


# ---------------------------------------------------------------------------
# YOLO training wrapper
# ---------------------------------------------------------------------------

def train_yolo(
    dataset_yaml: str,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    project: str = "ad_blocker_training",
    name: str = "yolo11n_ad",
    lr0: float = 0.01,
    lrf: float = 0.01,
    warmup_epochs: int = 3,
    patience: int = 20,
    augment: bool = True,
    mosaic: float = 1.0,
    mixup: float = 0.1,
    copy_paste: float = 0.1,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    cache: bool = True,
    workers: int = 4,
    deterministic: bool = True,
    seed: int = 42,
) -> str:
    """Train a YOLO model on advertisement dataset.

    Args:
        dataset_yaml: Path to dataset YAML.
        model_name: Base model (yolo11n.pt, yolov8n.pt, etc.).
        epochs: Number of training epochs.
        batch_size: Batch size (reduce to 8 for RPi training).
        imgsz: Input image size.
        device: 'cpu', 'cuda:0', etc.
        project: Project directory name.
        name: Run name.
        lr0: Initial learning rate.
        lrf: Final learning rate factor.
        warmup_epochs: Warmup epochs.
        patience: Early stopping patience.
        augment: Enable data augmentation.
        mosaic: Mosaic augmentation probability.
        mixup: Mixup augmentation probability.
        copy_paste: Copy-paste augmentation probability.
        hsv_h, hsv_s, hsv_v: HSV augmentation ranges.
        degrees: Rotation augmentation degrees.
        translate: Translation augmentation fraction.
        scale: Scaling augmentation gain.
        shear: Shear augmentation degrees.
        perspective: Perspective augmentation.
        flipud: Vertical flip probability.
        fliplr: Horizontal flip probability.
        cache: Cache images in RAM.
        workers: Data loading workers.
        deterministic: Set deterministic training.
        seed: Random seed.

    Returns:
        Path to the best trained model weights.
    """
    logger.info("Starting YOLO training: model={}, epochs={}, batch={}",
                model_name, epochs, batch_size)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        raise

    model = YOLO(model_name)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        patience=patience,
        augment=augment,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        cache=cache,
        workers=workers,
        deterministic=deterministic,
        seed=seed,
    )

    best_path = str(Path(project) / name / "weights" / "best.pt")
    logger.info("Training complete. Best model: {}", best_path)

    # Save training metrics
    metrics_path = Path(project) / name / "training_metrics.json"
    if hasattr(results, "results_dict"):
        with open(metrics_path, "w") as f:
            json.dump(results.results_dict, f, indent=2)
        logger.info("Training metrics saved: {}", metrics_path)

    return best_path


# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_yaml_path: str) -> dict:
    """Analyze a YOLO dataset and return class distribution statistics."""
    with open(dataset_yaml_path) as f:
        data = yaml.safe_load(f)

    analysis = {
        "num_classes": data.get("nc", 0),
        "class_names": data.get("names", []),
        "train_path": data.get("train", ""),
        "val_path": data.get("val", ""),
        "test_path": data.get("test", None),
    }

    # Count images per split
    for split in ["train", "val", "test"]:
        path = data.get(split)
        if not path:
            continue
        p = Path(path)
        if p.exists():
            image_files = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
            analysis[f"{split}_images"] = len(image_files)

    logger.info("Dataset analysis: {}", analysis)
    return analysis


__all__ = [
    "create_dataset_yaml",
    "train_yolo",
    "analyze_dataset",
    "AD_DATASET_CLASSES",
]