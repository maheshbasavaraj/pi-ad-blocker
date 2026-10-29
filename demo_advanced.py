#!/usr/bin/env python3
"""
Pi Ad Blocker — Advanced Demo (OCR + Natural Infill)

Detects advertisements via YOLO + OCR and replaces them with
a seamless neutral background fill instead of a black rectangle.

Usage:
    # OCR text detection + black masking
    python demo_advanced.py --ocr

    # Natural infill replacement (perspective warp + feathering)
    python demo_advanced.py --fill

    # Both combined — maximum coverage
    python demo_advanced.py --ocr --fill

    # Quick test on first 100 frames
    python demo_advanced.py --ocr --fill --quick
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pi_ad_blocker.config import AppConfig
from pi_ad_blocker.pipeline import process_video


def run_ocr_demo(input_video: str, output_video: str, quick: bool = False) -> None:
    """Demo with OCR text detection + black masking."""
    print("=" * 60)
    print("OCR TEXT DETECTION + BLACK MASKING")
    print("Detects ad text on any surface — billboards, signs, posters")
    print("=" * 60)

    config = AppConfig()
    config.detector.ocr_enabled = True
    config.detector.ocr_run_every_n_frames = 5
    config.masking.mode = "black"
    config.video_io.enable_preview = False

    max_frames = 100 if quick else 0
    config.video_io.input_source = input_video
    config.video_io.output_path = output_video

    print(f"Input:  {input_video}")
    stats = process_video(config, input_video, output_video, show_preview=False, max_frames=max_frames)

    print(f"\n✅ OCR DEMO — {stats['frames_processed']} frames @ {stats['avg_fps']:.1f} FPS")
    print(f"   Output: {output_video}\n")


def run_fill_demo(
    input_video: str,
    output_video: str,
    infill_image: str,
    enable_ocr: bool = False,
    quick: bool = False,
) -> None:
    """Demo with perspective-warped natural infill replacement."""
    print("=" * 60)
    print("NATURAL INFILL REPLACEMENT")
    print("Replaces detected regions with seamless background fill")
    print("=" * 60)

    config = AppConfig()
    config.detector.ocr_enabled = enable_ocr
    config.masking.mode = "replacement"
    config.masking.replacement_image_path = infill_image
    config.masking.feather_radius = 15
    config.masking.blend_alpha = 0.95
    config.masking.temporal_smoothing = True
    config.video_io.enable_preview = False

    max_frames = 100 if quick else 0

    print(f"Input:       {input_video}")
    print(f"OCR:         {'on' if enable_ocr else 'off'}")
    print(f"Infill img:  {infill_image}")

    stats = process_video(config, input_video, output_video, show_preview=False, max_frames=max_frames)

    print(f"\n✅ FILL DEMO — {stats['frames_processed']} frames @ {stats['avg_fps']:.1f} FPS")
    print(f"   Output: {output_video}\n")


def main():
    parser = argparse.ArgumentParser(description="Pi Ad Blocker — Advanced Demo")
    parser.add_argument("--input", default=None, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--fill-img", default="data/natural_infill.png",
                        help="Natural infill image path")
    parser.add_argument("--ocr", action="store_true",
                        help="Enable OCR text detection")
    parser.add_argument("--fill", action="store_true",
                        help="Use natural infill replacement instead of black")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (first 100 frames)")
    args = parser.parse_args()

    project = Path(__file__).parent
    default_input = str(project / "data" / "outputs" / "test_ad_video.mp4")
    default_output = str(project / "data" / "outputs" / "advanced_demo_output.mp4")

    input_video = args.input or default_input
    output_video = args.output or default_output

    if args.fill:
        run_fill_demo(input_video, output_video, args.fill_img,
                      enable_ocr=args.ocr, quick=args.quick)
    elif args.ocr:
        run_ocr_demo(input_video, output_video, quick=args.quick)
    else:
        print("Select a mode:")
        print("  --ocr   OCR text detection + black masking")
        print("  --fill  Natural infill replacement")
        print("  --ocr --fill  Both combined")
        print()
        print("Examples:")
        print("  python demo_advanced.py --ocr --fill --quick")


if __name__ == "__main__":
    main()