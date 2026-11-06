"""Integration tests for the end-to-end video pipeline.

Uses a synthetic clip and a stub detector so the whole read → detect → track →
mask → write path is exercised without model weights. `pipeline.py` was at 0%
coverage before these.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pi_ad_blocker.config import (  # noqa: E402
    AppConfig,
    DetectorConfig,
    MaskingConfig,
    TrackerConfig,
    VideoIOConfig,
)
from pi_ad_blocker.modules.detector import Detection  # noqa: E402
from pi_ad_blocker.modules.tracker import NoTracker, create_tracker  # noqa: E402
from pi_ad_blocker.pipeline import AdBlockPipeline, process_video  # noqa: E402

W, H = 160, 120
N_FRAMES = 8
AD_BOX = (60, 40, 100, 80)


class StubDetector:
    """Fixed ad over a bright frame; counts calls."""

    def __init__(self, detections: int = 1) -> None:
        self.calls = 0
        self._n = detections

    def detect(self, frame):
        self.calls += 1
        poly = np.array(
            [[60, 40], [100, 40], [100, 80], [60, 80]], np.float32
        )
        return [
            Detection(0, "ad", 0.9, AD_BOX, polygon=poly)
            for _ in range(self._n)
        ]

    def reset_timing(self) -> None:
        pass


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("clip") / "in.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H)
    )
    frame = np.full((H, W, 3), 210, np.uint8)
    for _ in range(N_FRAMES):
        writer.write(frame)
    writer.release()
    return str(path)


def _config(clip_path: str, out_path: str, tracker: str = "bytetrack") -> AppConfig:
    cfg = AppConfig(
        detector=DetectorConfig(model_type="yolo_seg", confidence_threshold=0.25),
        tracker=TrackerConfig(type=tracker),
        masking=MaskingConfig(mode="black", use_polygon=True, edge_feather_px=0),
        video_io=VideoIOConfig(
            input_source=clip_path,
            output_path=out_path,
            enable_preview=False,
        ),
    )
    return cfg


def _read_all(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def test_pipeline_masks_ads_and_writes_output(clip, tmp_path, monkeypatch):
    out = tmp_path / "masked.mp4"
    stub = StubDetector()
    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: stub
    )

    stats = AdBlockPipeline(_config(clip, str(out))).run()

    assert stats["frames_processed"] == N_FRAMES
    assert stats["avg_fps"] > 0
    assert out.exists()

    frames = _read_all(str(out))
    assert len(frames) == N_FRAMES
    centre = frames[0][60, 80]  # inside AD_BOX
    outside = frames[0][5, 5]
    assert int(centre.max()) < 40, f"ad not masked: {centre}"
    assert int(outside.min()) > 150, f"frame wrongly masked outside the ad: {outside}"


def test_max_frames_limits_work(clip, tmp_path, monkeypatch):
    out = tmp_path / "three.mp4"
    stub = StubDetector()
    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: stub
    )

    stats = AdBlockPipeline(_config(clip, str(out))).run(max_frames=3)

    assert stats["frames_processed"] == 3
    assert stub.calls == 3


def test_tracker_none_disables_tracking(clip, tmp_path, monkeypatch):
    out = tmp_path / "notrack.mp4"
    stub = StubDetector()
    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: stub
    )

    pipe = AdBlockPipeline(_config(clip, str(out), tracker="none"))
    assert isinstance(pipe.tracker, NoTracker)
    stats = pipe.run(max_frames=2)
    assert stats["frames_processed"] >= 2
    frames = _read_all(str(out))
    assert int(frames[0][60, 80].max()) < 40


def test_frame_callback_receives_frames(clip, tmp_path, monkeypatch):
    out = tmp_path / "cb.mp4"
    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: StubDetector()
    )

    seen = []
    pipe = AdBlockPipeline(_config(clip, str(out)))
    pipe.on_frame_callback = lambda frame, dets, tracks: seen.append((frame, dets))
    pipe.run(max_frames=3)

    assert len(seen) == 3
    assert all(len(dets) == 1 for _, dets in seen)


def test_empty_source_raises(tmp_path, monkeypatch):
    class EmptyDetector(StubDetector):
        pass

    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: EmptyDetector()
    )
    missing = tmp_path / "missing.mp4"
    with pytest.raises((RuntimeError, Exception)):
        AdBlockPipeline(_config(str(missing), str(tmp_path / "o.mp4"))).run()


def test_process_video_sets_paths_and_preview(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pi_ad_blocker.pipeline.create_detector", lambda cfg: StubDetector()
    )
    out = tmp_path / "api.mp4"
    cfg = AppConfig(
        detector=DetectorConfig(model_type="yolo_seg"),
        masking=MaskingConfig(mode="black", use_polygon=True),
        video_io=VideoIOConfig(enable_preview=False),
    )
    stats = process_video(cfg, clip, str(out), show_preview=False, max_frames=2)
    assert stats["frames_processed"] >= 2
    assert out.exists()


def test_no_tracker_passthrough_keeps_detection_payload():
    det = Detection(0, "ad", 0.8, (1, 2, 30, 40),
                    polygon=np.array([[1, 2], [30, 2], [30, 40]], np.float32))
    tr = NoTracker(TrackerConfig(type="none"))
    tracks = tr.update([det])
    assert len(tracks) == 1
    assert tracks[0].bbox == det.bbox
    assert tracks[0].track_id == 1
    assert tracks[0].polygon is det.polygon


def test_tracker_factory_accepts_none():
    assert isinstance(create_tracker(TrackerConfig(type="none")), NoTracker)
