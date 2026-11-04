#!/usr/bin/env python3
"""Command-line interface for Pi Ad Blocker.

Usage:
    pi-ad-blocker run [--video PATH] [--camera ID] [--output PATH] [--config PATH]
    pi-ad-blocker train [--data YAML] [--model NAME] [--epochs N]
    pi-ad-blocker evaluate [--video PATH] [--model PATH] [--gt PATH]
    pi-ad-blocker benchmark [--video PATH] [--model PATH]
    pi-ad-blocker export [--model PATH] [--format onnx|int8]
    pi-ad-blocker config [--show|--edit|--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from pi_ad_blocker import __version__
from pi_ad_blocker.config import (
    AppConfig,
    DEFAULT_CONFIG_YAML,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "pi-ad-blocker" / "config.yaml"


def _load_or_default(config_path: Optional[str]) -> AppConfig:
    if config_path:
        return load_config(config_path)
    if DEFAULT_CONFIG_PATH.exists():
        return load_config(DEFAULT_CONFIG_PATH)
    logger.info("No config found — using defaults")
    return AppConfig()


def _ensure_default_config() -> Path:
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "w") as f:
            f.write(DEFAULT_CONFIG_YAML)
        logger.info("Default config written to {}", DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run the advertisement blocking pipeline."""
    from pi_ad_blocker.pipeline import process_video, process_camera

    config = _load_or_default(args.config)

    if args.display_mode:
        config.masking.mode = args.display_mode

    if args.realtime:
        from pi_ad_blocker.realtime import process_video_realtime

        source = args.video if args.video else str(args.camera)
        output_path = args.output or f"output_rt_{Path(source).stem}.mp4"
        stats = process_video_realtime(
            config,
            source,
            output_path,
            max_frames=args.max_frames,
            show_preview_window=not args.no_preview,
            pace_to_source_fps=not args.unpaced,
        )
        print(f"\n{'='*50}")
        print("Real-time processing complete:")
        print(f"  Frames rendered:  {stats['frames']}")
        print(f"  Render FPS:       {stats['render_fps']}")
        print(f"  Detector FPS:     {stats['detector_fps']}")
        print(f"  Detector latency: {stats['detector_ms_avg']} ms avg / {stats['detector_ms_p95']} ms p95")
        print(f"  Frame latency:    {stats['frame_ms_avg']} ms avg / {stats['frame_ms_p95']} ms p95")
        print(f"{'='*50}")
        return

    if args.video:
        output_path = args.output or f"output_{Path(args.video).stem}.mp4"
        stats = process_video(config, args.video, output_path, show_preview=True)
    elif args.camera is not None:
        output_path = args.output or "camera_output.mp4"
        from pi_ad_blocker.pipeline import process_camera
        config.video_io.input_source = str(args.camera)
        config.video_io.output_path = output_path
        # Can't use process_camera directly, run pipeline
        from pi_ad_blocker.pipeline import AdBlockPipeline
        pipeline = AdBlockPipeline(config)
        stats = pipeline.run()
    else:
        logger.error("Specify --video or --camera")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Processing complete:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Average FPS:      {stats['avg_fps']}")
    print(f"  Avg latency:      {stats['avg_latency_ms']:.1f} ms")
    print(f"  Total time:       {stats['total_time_s']:.1f}s")
    print(f"{'='*50}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train a YOLO model on advertisement dataset."""
    from pi_ad_blocker.training import train_yolo

    model_path = train_yolo(
        dataset_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )
    print(f"Trained model: {model_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a detector on a video with ground truth."""
    from pi_ad_blocker.config import DetectorConfig
    from pi_ad_blocker.modules.detector import create_detector
    from pi_ad_blocker.modules.evaluation import Evaluator, EvaluationConfig

    detector_config = DetectorConfig(model_path=args.model)
    detector = create_detector(detector_config)
    eval_config = EvaluationConfig(ground_truth_path=args.gt or "")
    evaluator = Evaluator(eval_config, detector)
    metrics = evaluator.evaluate_video(args.video)

    output_dir = Path(args.output_dir) if args.output_dir else Path("eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    Evaluator.generate_report(metrics, str(output_dir))

    print(f"\nEvaluation Results:")
    print(f"  FPS:   {metrics.fps:.1f}")
    print(f"  Lat:   {metrics.avg_latency_ms:.1f} ms")
    print(f"  CPU:   {metrics.cpu_usage_percent:.1f}%")
    print(f"  Mem:   {metrics.memory_usage_mb:.1f} MB")
    print(f"  Frames:{metrics.total_frames}")
    print(f"Report: {output_dir}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark detector performance."""
    import json
    from pi_ad_blocker.modules.deployment import run_benchmark

    try:
        results = run_benchmark(args.model, args.video, iterations=args.iterations)
        print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error("Benchmark failed: {}", e)
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    """Export model to ONNX or INT8 quantized format."""
    from pi_ad_blocker.modules.deployment import export_to_onnx, quantize_onnx_int8

    model_path = args.model
    model_stem = Path(model_path).stem

    if args.format == "onnx":
        output = f"{model_stem}.onnx"
        result = export_to_onnx(model_path, output)
        print(f"Exported: {result}")
    elif args.format == "int8":
        if args.input_onnx:
            onnx_path = args.input_onnx
        else:
            # Export to ONNX first
            onnx_path = f"{model_stem}.onnx"
            if not Path(onnx_path).exists():
                onnx_path = export_to_onnx(model_path, onnx_path)
        output = f"{model_stem}_int8.onnx"
        result = quantize_onnx_int8(onnx_path, output)
        print(f"Quantized: {result}")
    else:
        logger.error("Unknown format: {}", args.format)
        sys.exit(1)


def cmd_config(args: argparse.Namespace) -> None:
    """Manage configuration."""
    if args.show or args.reset:
        config_path = _ensure_default_config()
        if args.reset:
            config_path.unlink(missing_ok=True)
            _ensure_default_config()
            print(f"Config reset to defaults: {config_path}")

        with open(config_path) as f:
            print(f.read())
    elif args.edit:
        editor = args.editor or "nano"
        import subprocess
        subprocess.call([editor, str(_ensure_default_config())])
    else:
        print(f"Config path: {DEFAULT_CONFIG_PATH}")
        print(f"Exists: {DEFAULT_CONFIG_PATH.exists()}")


def cmd_install(args: argparse.Namespace) -> None:
    """Install dependencies for Raspberry Pi deployment."""
    from pi_ad_blocker.modules.deployment import install_dependencies
    install_dependencies(accelerator=args.accelerator)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-ad-blocker",
        description="Raspberry Pi Advertisement Blocking System v" + __version__,
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_p = sub.add_parser("run", help="Run the advertisement blocking pipeline")
    run_p.add_argument("--video", help="Input video file path")
    run_p.add_argument("--camera", type=int, default=None, help="Camera device index")
    run_p.add_argument("--output", help="Output video path (optional)")
    run_p.add_argument("--config", help="Config file path (optional)")
    run_p.add_argument(
        "--realtime",
        action="store_true",
        help="Use the low-latency AR pipeline (background detector + optical-flow masks)",
    )
    run_p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    run_p.add_argument("--no-preview", action="store_true", help="Do not open a preview window")
    run_p.add_argument(
        "--unpaced", action="store_true", help="Render as fast as possible instead of matching source FPS"
    )
    run_p.add_argument(
        "--display-mode",
        choices=["black", "blur", "pixelate", "replacement"],
        default=None,
        help="Override masking mode",
    )

    # train
    train_p = sub.add_parser("train", help="Train a YOLO model")
    train_p.add_argument("--data", required=True, help="Dataset YAML path")
    train_p.add_argument("--model", default="yolo11n.pt", help="Base model name")
    train_p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_p.add_argument("--batch", type=int, default=16, help="Batch size")
    train_p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    train_p.add_argument("--device", default="cpu", help="Training device")
    train_p.add_argument("--project", default="ad_blocker_training", help="Project name")
    train_p.add_argument("--name", default="yolo11n_ad", help="Run name")
    train_p.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate detector performance")
    eval_p.add_argument("--video", required=True, help="Video to evaluate on")
    eval_p.add_argument("--model", required=True, help="Model path")
    eval_p.add_argument("--gt", help="Ground truth annotations (optional)")
    eval_p.add_argument("--output-dir", default="eval_results", help="Output directory")

    # benchmark
    bench_p = sub.add_parser("benchmark", help="Benchmark detector performance")
    bench_p.add_argument("--video", required=True, help="Benchmark video")
    bench_p.add_argument("--model", required=True, help="Model path")
    bench_p.add_argument("--iterations", type=int, default=3, help="Benchmark iterations")

    # export
    export_p = sub.add_parser("export", help="Export model to ONNX/INT8")
    export_p.add_argument("--model", required=True, help="Source model (.pt)")
    export_p.add_argument("--format", choices=["onnx", "int8"], default="onnx",
                          help="Export format")
    export_p.add_argument("--input-onnx", help="Input ONNX for INT8 quantization (optional)")

    # config
    config_p = sub.add_parser("config", help="Manage configuration")
    config_p.add_argument("--show", action="store_true", help="Show current config")
    config_p.add_argument("--edit", action="store_true", help="Edit config in editor")
    config_p.add_argument("--editor", default="nano", help="Editor for --edit")
    config_p.add_argument("--reset", action="store_true", help="Reset to defaults")

    # install
    install_p = sub.add_parser("install", help="Install dependencies for RPi")
    install_p.add_argument("--accelerator", choices=["none", "coral", "hailo"],
                           default="none", help="Accelerator type")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Remove default loguru handler for cleaner CLI
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "run": cmd_run,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "benchmark": cmd_benchmark,
        "export": cmd_export,
        "config": cmd_config,
        "install": cmd_install,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()