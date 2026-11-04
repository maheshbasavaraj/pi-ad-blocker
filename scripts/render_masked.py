"""Render a masked video and dump inspection frames."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "src")

from pi_ad_blocker.config import (
    AppConfig,
    DetectorConfig,
    MaskingConfig,
    RealtimeConfig,
)
from pi_ad_blocker.realtime import RealtimeAdBlocker


def build_config(args) -> AppConfig:
    det = DetectorConfig(
        model_path=args.model,
        model_type=args.model_type,
        confidence_threshold=args.conf,
        input_width=args.imgsz,
        input_height=args.imgsz,
        filter_to_ad_classes=args.ad_classes_only,
        prompt_classes=args.prompts or DetectorConfig().prompt_classes,
    )
    return AppConfig(
        detector=det,
        masking=MaskingConfig(
            mode=args.effect,
            use_polygon=args.polygon,
            edge_feather_px=args.feather,
            mask_opacity=args.opacity,
            output_mode=args.output_mode,
        ),
        realtime=RealtimeConfig(
            async_detection=not args.sync,
            detect_interval=args.detect_interval,
            max_propagation_frames=args.max_propagation,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/outputs/masked.mp4")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="onnx_seg")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--effect", default="black")
    ap.add_argument("--output-mode", default="masked")
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--polygon", action="store_true", default=True)
    ap.add_argument("--no-polygon", dest="polygon", action="store_false")
    ap.add_argument("--feather", type=int, default=2)
    ap.add_argument("--detect-interval", type=int, default=1)
    ap.add_argument("--max-propagation", type=int, default=30)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--dump-every", type=int, default=25)
    ap.add_argument("--ad-classes-only", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_dir = out_path.parent / (out_path.stem + "_frames")
    dump_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    blocker = RealtimeAdBlocker(cfg)
    blocker.start()

    changed_frames = 0
    done = 0
    t0 = time.perf_counter()
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        out = blocker.process_frame(frame, i)
        writer.write(out)
        diff = cv2.absdiff(frame, out).max(axis=2)
        ratio = float((diff > 20).mean())
        if ratio > 0.005:
            changed_frames += 1
        if args.dump_every and i % args.dump_every == 0:
            side = cv2.hconcat([frame, out])
            cv2.imwrite(str(dump_dir / f"frame_{i:05d}.png"), side)
        done += 1
    elapsed = time.perf_counter() - t0
    blocker.stop()
    writer.release()
    cap.release()

    st = blocker.stats(elapsed, done)
    print(f"frames={done} out={out_path}")
    print(f"render_fps={st['render_fps']:.2f} detector_fps={st['detector_fps']:.2f}")
    print(f"changed_frames={changed_frames}/{done} "
          f"({100.0 * changed_frames / max(done, 1):.1f}%)")
    print(f"dumps={dump_dir}")


if __name__ == "__main__":
    main()



