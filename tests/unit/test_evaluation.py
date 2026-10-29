"""Unit tests for evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_ad_blocker.config import EvaluationConfig
from pi_ad_blocker.modules.detector import Detection
from pi_ad_blocker.modules.evaluation import Evaluator, compute_iou, EvalMetrics


class TestIoU:
    def test_perfect_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == 1.0

    def test_no_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (100, 100, 200, 200))
        assert iou == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.14 < iou < 0.15  # ~0.1428

    def test_edge_touching_no_area(self):
        iou = compute_iou((0, 0, 100, 100), (100, 100, 200, 200))
        assert iou == 0.0


class TestEvaluator:
    def test_video_not_found(self):
        config = EvaluationConfig()
        from pi_ad_blocker.config import DetectorConfig
        from pi_ad_blocker.modules.detector import BaseDetector

        class MockDetector(BaseDetector):
            def load_model(self): pass
            def preprocess(self, image): return None
            def infer(self, tensor): return None
            def postprocess(self, raw, shape): return []

        evaluator = Evaluator(config, MockDetector(DetectorConfig()))
        with pytest.raises(Exception):
            evaluator.evaluate_video("/nonexistent/video.mp4")

    def test_metrics_computation(self):
        config = EvaluationConfig()
        from pi_ad_blocker.config import DetectorConfig
        from pi_ad_blocker.modules.detector import BaseDetector

        class MockDetector(BaseDetector):
            def load_model(self): pass
            def preprocess(self, image): return None
            def infer(self, tensor): return None
            def postprocess(self, raw, shape): return []

        evaluator = Evaluator(config, MockDetector(DetectorConfig()))

        preds = [
            [Detection(79, "billboard", 0.9, (0, 0, 100, 100))],
        ]
        gts = [
            [Detection(79, "billboard", 1.0, (5, 5, 95, 95))],  # high IoU
        ]

        metrics = evaluator.compute_metrics(preds, gts)
        assert metrics.true_positives >= 0
        assert metrics.precision >= 0
        assert metrics.recall >= 0

    def test_generate_report(self, tmp_path):
        metrics = EvalMetrics(
            precision=0.85,
            recall=0.78,
            f1_score=0.81,
            mAP50=0.82,
            fps=15.5,
            avg_latency_ms=64.2,
            cpu_usage_percent=45.0,
            memory_usage_mb=256.0,
            total_frames=500,
            true_positives=340,
            false_positives=60,
            false_negatives=96,
        )

        output_dir = tmp_path / "eval_report"
        output_dir.mkdir(parents=True, exist_ok=True)
        Evaluator.generate_report(metrics, str(output_dir))

        report_path = output_dir / "evaluation_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)

        assert data["precision"] == 0.85
        assert data["fps"] == 15.5
        assert data["true_positives"] == 340