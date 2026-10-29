"""Evaluation module — benchmarks detection accuracy and performance metrics.

Computes precision, recall, mAP, FPS, latency, CPU/memory usage.
Generates evaluation reports with plots.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from pi_ad_blocker.config import EvaluationConfig, DetectorConfig
from pi_ad_blocker.modules.detector import BaseDetector, Detection, create_detector
from pi_ad_blocker.modules.video_io import BaseReader, create_reader


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One evaluation sample (frame-level ground truth + predictions)."""

    frame_number: int
    ground_truth: list[Detection]
    predictions: list[Detection]


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mAP50: float = 0.0
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    total_frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_class_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def compute_iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

class Evaluator:
    """Full evaluation pipeline: ground truth data → metrics."""

    def __init__(self, config: EvaluationConfig, detector: BaseDetector) -> None:
        self.config = config
        self.detector = detector
        self._samples: list[EvalSample] = []

    def load_ground_truth(self, gt_path: str) -> list[EvalSample]:
        """Load ground truth from JSON or YOLO-format annotations."""
        path = Path(gt_path)
        if not path.exists():
            logger.warning("Ground truth file not found: {}", gt_path)
            return []

        if path.suffix == ".json":
            return self._load_json_gt(path)
        elif path.suffix == ".txt":
            return self._load_yolo_gt(path)
        else:
            raise ValueError(f"Unsupported ground truth format: {path.suffix}")

    def _load_json_gt(self, path: Path) -> list[EvalSample]:
        with open(path) as f:
            data = json.load(f)

        samples: list[EvalSample] = []
        for item in data:
            detections = [
                Detection(
                    class_id=d["class_id"],
                    class_name=d.get("class_name", ""),
                    confidence=d.get("confidence", 1.0),
                    bbox=tuple(d["bbox"]),
                )
                for d in item.get("detections", [])
            ]
            samples.append(
                EvalSample(
                    frame_number=item["frame_number"],
                    ground_truth=detections,
                    predictions=[],
                )
            )
        return samples

    def _load_yolo_gt(self, path: Path) -> list[EvalSample]:
        # YOLO format: class_id cx cy w h (normalized)
        samples: list[EvalSample] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                samples.append(
                    EvalSample(
                        frame_number=0,
                        ground_truth=[
                            Detection(
                                class_id=class_id,
                                class_name="",
                                confidence=1.0,
                                bbox=(cx, cy, w, h),
                            )
                        ],
                        predictions=[],
                    )
                )
        return samples

    def evaluate_video(self, video_path: str) -> EvalMetrics:
        """Run evaluation on a video file."""
        logger.info("Evaluating on video: {}", video_path)

        # Configure reader
        from pi_ad_blocker.config import VideoIOConfig
        io_config = VideoIOConfig(input_source=video_path)
        reader = create_reader(io_config)
        reader.open()

        total_frames = 0
        latencies: list[float] = []
        all_predictions: list[list[Detection]] = []

        import psutil
        process = psutil.Process()
        cpu_samples: list[float] = []
        mem_samples: list[float] = []

        for frame in reader:
            total_frames += 1
            t0 = time.perf_counter()
            detections = self.detector.detect(frame.data)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            all_predictions.append(detections)

            if total_frames % 100 == 0:
                cpu_samples.append(process.cpu_percent(interval=0.1))
                mem_samples.append(process.memory_info().rss / (1024 * 1024))

        reader.close()

        # Compute metrics
        fps = total_frames / (sum(latencies) / 1000) if latencies else 0
        avg_lat = sum(latencies) / max(len(latencies), 1)
        avg_cpu = sum(cpu_samples) / max(len(cpu_samples), 1)
        avg_mem = sum(mem_samples) / max(len(mem_samples), 1)

        return EvalMetrics(
            fps=fps,
            avg_latency_ms=avg_lat,
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_mem,
            total_frames=total_frames,
        )

    def compute_metrics(
        self,
        predictions: list[list[Detection]],
        ground_truth: list[list[Detection]],
    ) -> EvalMetrics:
        """Compute precision, recall, mAP from prediction/ground truth pairs."""
        iou_thresh = self.config.iou_threshold

        tp = 0
        fp = 0
        fn = 0

        for preds, gts in zip(predictions, ground_truth):
            matched_gt = set()
            matched_pred = set()

            for pi, pred in enumerate(preds):
                best_iou = 0.0
                best_gt = -1
                for gi, gt in enumerate(gts):
                    if gi in matched_gt:
                        continue
                    iou = compute_iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                if best_iou >= iou_thresh and pred.class_id == gts[best_gt].class_id:
                    tp += 1
                    matched_gt.add(best_gt)
                    matched_pred.add(pi)
                else:
                    fp += 1

            fn += len(gts) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvalMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    @staticmethod
    def generate_report(metrics: EvalMetrics, output_path: str) -> None:
        """Write evaluation report as JSON + optional plots."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1_score": round(metrics.f1_score, 4),
            "mAP50": round(metrics.mAP50, 4),
            "fps": round(metrics.fps, 2),
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "cpu_usage_percent": round(metrics.cpu_usage_percent, 2),
            "memory_usage_mb": round(metrics.memory_usage_mb, 2),
            "total_frames": metrics.total_frames,
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "per_class": metrics.per_class_metrics,
        }

        report_path = path / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved: {}", report_path)

        # Generate plots if matplotlib available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle("Pi Ad Blocker — Evaluation Report")

            # Precision / Recall bar
            axes[0, 0].bar(["Precision", "Recall", "F1"],
                           [metrics.precision, metrics.recall, metrics.f1_score])
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_ylabel("Score")

            # FPS
            axes[0, 1].bar(["FPS"], [metrics.fps])
            axes[0, 1].set_ylabel("Frames / sec")

            # Latency
            axes[1, 0].bar(["Latency"], [metrics.avg_latency_ms])
            axes[1, 0].set_ylabel("ms")

            # Resource usage
            axes[1, 1].bar(["CPU %", "Memory MB"],
                           [metrics.cpu_usage_percent, metrics.memory_usage_mb])
            axes[1, 1].set_ylabel("Usage")

            plt.tight_layout()
            plot_path = path / "evaluation_plot.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info("Evaluation plot saved: {}", plot_path)
        except ImportError:
            logger.debug("matplotlib not available — skipping plots")

        logger.info(
            "Results: P={:.3f} R={:.3f} F1={:.3f} FPS={:.1f} Lat={:.1f}ms",
            metrics.precision, metrics.recall, metrics.f1_score,
            metrics.fps, metrics.avg_latency_ms,
        )


__all__ = [
    "Evaluator",
    "EvalMetrics",
    "EvalSample",
    "compute_iou",
]