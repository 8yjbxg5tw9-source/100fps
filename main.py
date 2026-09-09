"""CLI entry point — runs pipeline steps 1..N (default: 1-2).

Examples:
    python main.py --input video/in.mp4 --output video/out.mp4
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 1
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 3
    python main.py --config workspace/config.json --from-step 2
    python main.py --config workspace/config.json --from-step 3 --to-step 3

Exit codes:
    0 — success
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
from pipeline.rife.weights import DEFAULT_RIFE_VERSION, RIFE_VERSIONS
from pipeline.step01_environment import setup_environment
from pipeline.step02_frames import Step02Frames
from pipeline.step03_interpolate import Step03Interpolate

log = get_logger("main")

LAST_STEP = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "720p -> 8K @ 1000 FPS pipeline. "
            "Step 1: environment check, GPU analysis, central config. "
            "Step 2: metadata probe, audio + frame extraction. "
            "Step 3: RIFE interpolation to target FPS."
        )
    )
    # Input selection.
    parser.add_argument("--input", default=None, help="Input video file (720p).")
    parser.add_argument("--output", default=None, help="Final output video path.")
    parser.add_argument(
        "--config",
        default=None,
        help="Resume from a saved config.json (required with --from-step > 1).",
    )
    # Step range.
    parser.add_argument(
        "--from-step", type=int, default=1, choices=[1, 2, 3],
        help="First step to run (default: 1; >1 needs --config).",
    )
    parser.add_argument(
        "--to-step", type=int, default=2, choices=[1, 2, 3],
        help="Last step to run (default: 2).",
    )
    parser.add_argument(
        "--step1-only", action="store_true",
        help="Alias for --to-step 1.",
    )
    # General options.
    parser.add_argument(
        "--workspace", default="workspace",
        help="Working folder for frames + config.json (default: workspace).",
    )
    parser.add_argument(
        "--target-fps", type=float, default=TARGET_FPS, help="Target FPS (default: 1000)."
    )
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument(
        "--no-auto-install", action="store_true",
        help="Do not pip-install missing Python packages automatically.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help="Override the VRAM-derived tile_size (256/512/1024).",
    )
    parser.add_argument(
        "--no-save-config", action="store_true",
        help="Do not write workspace/config.json after Step 1.",
    )
    # Step 2 options.
    parser.add_argument(
        "--image-format", choices=["png", "jpg"], default="png",
        help="Raw frame format: png (lossless, default) or jpg (smaller).",
    )
    parser.add_argument(
        "--audio-format", choices=["wav", "aac"], default="wav",
        help="Extracted audio format: wav (lossless, default) or aac.",
    )
    parser.add_argument(
        "--keep-old-frames", action="store_true",
        help="Do not delete stale frame_*.* files before extraction.",
    )
    # Step 3 options.
    parser.add_argument(
        "--backend", choices=["rife", "blend"], default="rife",
        help="'rife': AI interpolation (needs torch); 'blend': non-AI smoke test.",
    )
    parser.add_argument(
        "--rife-version", choices=sorted(RIFE_VERSIONS), default=DEFAULT_RIFE_VERSION,
        help=f"RIFE weights version (default: {DEFAULT_RIFE_VERSION}).",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Explicit RIFE weights file (or dir with flownet.pkl); skips download.",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Inference device (default: auto = cuda if available else cpu).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Frame pairs per GPU forward batch (default: 4).",
    )
    parser.add_argument(
        "--max-exp", type=int, default=6,
        help="Refuse dense factors above 2^N (default: 6 = 64x).",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Force fp32 inference (default: fp16 on CUDA, fp32 on CPU).",
    )
    parser.add_argument(
        "--output-format", choices=["png", "jpg"], default="png",
        help="Interpolated frame format (default: png).",
    )
    parser.add_argument(
        "--no-shortcuts", action="store_true",
        help="Disable static/scene-cut pair shortcuts (always run inference).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.step1_only:
        args.to_step = 1
    if args.from_step > args.to_step:
        log.error("--from-step (%d) > --to-step (%d).", args.from_step, args.to_step)
        return 2
    try:
        if args.from_step > 1:
            if not args.config:
                log.error("--from-step %d requires --config <saved config.json>.", args.from_step)
                return 2
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
            if args.to_step < 2:
                return 0

        if args.from_step <= 2 <= args.to_step:
            Step02Frames(
                image_format=args.image_format,
                audio_format=args.audio_format,
                clean_frame_dir=not args.keep_old_frames,
                logger=log,
            ).run(config)

        if args.from_step <= 3 <= args.to_step:
            Step03Interpolate(
                backend=args.backend,
                rife_version=args.rife_version,
                weights=args.weights,
                weights_root="weights",
                device=args.device,
                fp16=False if args.fp32 else None,
                batch_size=args.batch_size,
                max_exp=args.max_exp,
                output_format=args.output_format,
                static_threshold=None if args.no_shortcuts else 1.0,
                cut_threshold=None if args.no_shortcuts else 60.0,
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
