"""CLI entry point — currently runs Step 1 (environment + config).

Example:
    python main.py --input video/in.mp4 --output video/out.mp4

Exit codes:
    0 — success (config ready for Step 2)
    1 — input video not found
    2 — any other failure
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH
from pipeline.exceptions import InputVideoNotFoundError
from pipeline.logger import get_logger
from pipeline.step01_environment import setup_environment

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "720p -> 8K @ 1000 FPS pipeline. "
            "Step 1: environment check, GPU analysis, central config."
        )
    )
    parser.add_argument("--input", required=True, help="Input video file (720p).")
    parser.add_argument("--output", required=True, help="Final output video path.")
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
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not write workspace/config.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        setup_environment(
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
    except InputVideoNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI must report, not traceback-spam
        log.error("Step 1 FAILED: %s: %s", type(exc).__name__, exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
