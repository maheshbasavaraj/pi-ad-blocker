#!/usr/bin/env python3
"""Run the Pi Ad Blocker pipeline on a video file.

Usage:
    python run_pipeline.py                          # uses synthetic test video
    python run_pipeline.py --video path/to/vid.mp4  # your own video
    python run_pipeline.py --video vid.mp4 --output result.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pi_ad_blocker.pipeline import process_video
from pi_ad_blocker.config import AppConfig


def main():
    parser = argparse.ArgumentParser(description="Pi Ad Blocker - detect and mask ads in video")
    parser.add_argument("--video", default=None, help="Path to input video (default: synthetic test video)")
    parser.add_argument("--output", default=None, help="Path for masked output video")
    args = parser.parse_args()

    input_video = args.video or str(Path(__file__).parent / "data" / "outputs" / "test_ad_video.mp4")
    output_video = args.output or str(Path(__file__).parent / "data" / "outputs" / "masked_output.mp4")

    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print()

    config = AppConfig()
    stats = process_video(config, input_video, output_video)

    print()
    print("=" * 50)
    print("PIPELINE COMPLETE ✅")
    print("=" * 50)
    print(f"  Frames processed:  {stats['frames_processed']}")
    print(f"  Average FPS:       {stats['avg_fps']:.1f}")
    print(f"  Average latency:   {stats['avg_latency_ms']:.0f} ms")
    print(f"  Output video:      {output_video}")


if __name__ == "__main__":
    main()