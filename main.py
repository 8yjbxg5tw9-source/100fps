"""CLI entry point — runs Step 1 (environment + config), then Step 2 (frames).

Examples:
    python main.py --input video/in.mp4 --output video/out.mp4
    python main.py --input video/in.mp4 --output video/out.mp4 --step1-only
    python main.py --config workspace/config.json          # resume at Step 2

Exit codes:
    0 — success (raw frames ready for Step 3)
    1 — input video not found
    2 — any other failure
    3 — FFmpeg/FFprobe missing (required from Step 2 on)
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH, PipelineConfig
from pipeline.exceptions import (
    FFmpegNotFoundError,
    InputVideoNotFoundError,
)
from pipeline.logger import get_logger
from pipeline.step01_environment import setup_environment
from pipeline.step02_frames import Step02Frames

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "720p -> 8K @ 1000 FPS pipeline. "
            "Step 1: environment check, GPU analysis, central config. "
            "Step 2: metadata probe, audio + frame extraction."
        )
    )
    # Step 1 inputs (not needed when resuming via --config).
    parser.add_argument("--input", default=None, help="Input video file (720p).")
    parser.add_argument("--output", default=None, help="Final output video path.")
    parser.add_argument(
        "--config",
        default=None,
        help="Resume from a saved config.json (runs Step 2 only).",
    )
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="Working folder for frames + config.json (default: workspace).",
    )
    parser.add_argument(
        "--target-fps", type=float, default=TARGET_FPS, help="Target FPS (default: 1000)."
    )
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Do not pip-install missing Python packages automatically.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Override the VRAM-derived tile_size (256/512/1024).",
    )
    # Step 1/2 flow control.
    parser.add_argument(
        "--step1-only",
        action="store_true",
        help="Stop after Step 1 (environment + config).",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not write workspace/config.json.",
    )
    # Step 2 options.
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Raw frame format: png (lossless, default) or jpg (smaller).",
    )
    parser.add_argument(
        "--audio-format",
        choices=["wav", "aac"],
        default="wav",
        help="Extracted audio format: wav (lossless, default) or aac.",
    )
    parser.add_argument(
        "--keep-old-frames",
        action="store_true",
        help="Do not delete stale frame_*.* files before extraction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.config:
            config = PipelineConfig.load(args.config)
            log.info("Resuming from saved config: %s", args.config)
        else:
            if not args.input or not args.output:
                log.error("--input and --output are required (or use --config to resume).")
                return 2
            config = setup_environment(
                input_video_path=args.input,
                final_output_path=args.output,
                workspace_root=args.workspace,
                target_fps=args.target_fps,
                target_width=args.target_width,
                target_height=args.target_height,
                auto_install=not args.no_auto_install,
                tile_size_override=args.tile_size,
                save_config=not args.no_save_config,
                logger=log,
            )
            if args.step1_only:
                return 0

        Step02Frames(
            image_format=args.image_format,
            audio_format=args.audio_format,
            clean_frame_dir=not args.keep_old_frames,
            logger=log,
        ).run(config)
    except InputVideoNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except FFmpegNotFoundError as exc:
        log.error("%s", exc)
        return 3
    except Exception as exc:  # noqa: BLE001 - CLI must report, not traceback-spam
        log.error("Pipeline FAILED: %s: %s", type(exc).__name__, exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
