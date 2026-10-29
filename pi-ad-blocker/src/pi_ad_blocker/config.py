"""Global configuration for the Pi Ad Blocker system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from yaml import safe_load, safe_dump

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "outputs"


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class DetectorConfig(BaseModel):
    """Detector settings."""

    model_path: str = str(DEFAULT_MODEL_DIR / "yolo11n_ad.onnx")
    model_type: Literal["yolov8", "yolo11", "mobilenet_ssd", "rtdetr"] = "yolo11"
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    input_width: int = 640
    input_height: int = 640
    device: str = "cpu"
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    num_classes: int = 80  # COCO classes; will be reduced for fine-tuned model
    use_quantized: bool = True
    enable_trace: bool = False

    # Ad-related COCO class IDs (billboard, signboard, etc.)
    ad_class_ids: list[int] = Field(default_factory=lambda: [79, 83, 84])
    custom_ad_classes: Optional[list[str]] = None

    # Accelerator
    accelerator: Literal["none", "coral", "hailo"] = "none"


class TrackerConfig(BaseModel):
    """Object tracker settings."""

    type: Literal["bytetrack", "ocsort", "botsort"] = "bytetrack"
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.50
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.25


class MaskingConfig(BaseModel):
    """Masking strategy settings."""

    mode: Literal["black", "blur", "pixelate", "custom"] = "black"
    mask_color: tuple[int, int, int] = (0, 0, 0)  # BGR default black
    blur_kernel_size: int = 31
    pixelation_blocks: int = 16
    mask_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    custom_mask_function: Optional[str] = None  # dotted import path
    padding_pixels: int = 4  # extra padding around bounding boxes


class VideoIOConfig(BaseModel):
    """Video input / output settings."""

    input_source: str = ""  # path or camera index or "0"
    output_path: Optional[str] = None
    output_fps: Optional[int] = None
    output_fourcc: str = "mp4v"
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    enable_preview: bool = True
    preview_window_name: str = "Pi Ad Blocker"
    camera_backend: Literal["opencv", "picamera2"] = "opencv"


class EvaluationConfig(BaseModel):
    """Evaluation pipeline settings."""

    ground_truth_path: str = ""
    iou_threshold: float = 0.50
    confidence_thresholds: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.35, 0.5])
    save_plots: bool = True
    output_dir: str = str(DEFAULT_OUTPUT_DIR / "eval")


class AppConfig(BaseModel):
    """Top-level application configuration."""

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    video_io: VideoIOConfig = Field(default_factory=VideoIOConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    log_level: str = "INFO"
    log_file: Optional[str] = None
    profile_mode: bool = False
    benchmark_mode: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Load / Save helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """Load application config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = safe_load(f)
    return AppConfig(**data)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Save application config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        safe_dump(config.model_dump(), f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Default config YAML
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_YAML = """
detector:
  model_path: "data/models/yolo11n_ad.onnx"
  model_type: "yolo11"
  confidence_threshold: 0.35
  iou_threshold: 0.45
  input_width: 640
  input_height: 640
  device: "cpu"
  providers: ["CPUExecutionProvider"]
  use_quantized: true
  ad_class_ids: [79, 83, 84]
  accelerator: "none"

tracker:
  type: "bytetrack"
  track_high_thresh: 0.35
  track_low_thresh: 0.10
  new_track_thresh: 0.50
  track_buffer: 30
  match_thresh: 0.80

masking:
  mode: "black"
  mask_color: [0, 0, 0]
  blur_kernel_size: 31
  pixelation_blocks: 16
  mask_opacity: 1.0
  padding_pixels: 4

video_io:
  input_source: "0"
  output_path: null
  output_fps: null
  output_fourcc: "mp4v"
  enable_preview: true
  preview_window_name: "Pi Ad Blocker"
  camera_backend: "opencv"

evaluation:
  ground_truth_path: ""
  iou_threshold: 0.50
  confidence_thresholds: [0.1, 0.25, 0.35, 0.5]
  save_plots: true
  output_dir: "data/outputs/eval"

log_level: "INFO"
seed: 42
"""