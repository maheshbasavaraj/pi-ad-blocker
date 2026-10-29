"""Unit tests for configuration module."""

from __future__ import annotations

import yaml
import pytest

from pi_ad_blocker.config import (
    AppConfig,
    DetectorConfig,
    TrackerConfig,
    MaskingConfig,
    VideoIOConfig,
    EvaluationConfig,
    load_config,
    save_config,
    DEFAULT_CONFIG_YAML,
)


class TestDetectorConfig:
    def test_default_values(self):
        config = DetectorConfig()
        assert config.confidence_threshold == 0.35
        assert config.iou_threshold == 0.45
        assert config.input_width == 640
        assert config.device == "cpu"
        assert config.accelerator == "none"

    def test_custom_values(self):
        config = DetectorConfig(
            model_path="/tmp/test.onnx",
            confidence_threshold=0.5,
            model_type="yolov8",
        )
        assert config.model_path == "/tmp/test.onnx"
        assert config.confidence_threshold == 0.5
        assert config.model_type == "yolov8"

    def test_validation(self):
        with pytest.raises(ValueError):
            DetectorConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError):
            DetectorConfig(confidence_threshold=-0.1)


class TestTrackerConfig:
    def test_default_tracker_type(self):
        config = TrackerConfig()
        assert config.type == "bytetrack"

    def test_custom_tracker(self):
        config = TrackerConfig(type="ocsort", track_buffer=60)
        assert config.type == "ocsort"
        assert config.track_buffer == 60


class TestMaskingConfig:
    def test_default_masking(self):
        config = MaskingConfig()
        assert config.mode == "black"
        assert config.mask_color == (0, 0, 0)
        assert config.mask_opacity == 1.0

    def test_blur_mode(self):
        config = MaskingConfig(mode="blur", blur_kernel_size=51)
        assert config.mode == "blur"
        assert config.blur_kernel_size == 51

    def test_pixelation(self):
        config = MaskingConfig(mode="pixelate", pixelation_blocks=8)
        assert config.mode == "pixelate"
        assert config.pixelation_blocks == 8


class TestVideoIOConfig:
    def test_defaults(self):
        config = VideoIOConfig()
        assert config.input_source == ""
        assert config.camera_backend == "opencv"


class TestAppConfig:
    def test_default_app_config(self):
        config = AppConfig()
        assert config.log_level == "INFO"
        assert config.seed == 42
        assert isinstance(config.detector, DetectorConfig)
        assert isinstance(config.tracker, TrackerConfig)
        assert isinstance(config.masking, MaskingConfig)

    def test_serialization_roundtrip(self, tmp_path):
        config = AppConfig(
            detector=DetectorConfig(confidence_threshold=0.5),
            masking=MaskingConfig(mode="blur"),
            log_level="DEBUG",
        )

        path = tmp_path / "test_config.yaml"
        save_config(config, str(path))
        assert path.exists()

        loaded = load_config(str(path))
        assert loaded.detector.confidence_threshold == 0.5
        assert loaded.masking.mode == "blur"
        assert loaded.log_level == "DEBUG"

    def test_default_yaml_valid(self):
        """The bundled default YAML should parse as valid config."""
        data = yaml.safe_load(DEFAULT_CONFIG_YAML)
        config = AppConfig(**data)
        assert config.detector.confidence_threshold == 0.35


class TestLoadConfig:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{{invalid_yaml}}")
        with pytest.raises(Exception):
            load_config(str(p))