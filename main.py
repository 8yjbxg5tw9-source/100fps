"""CLI entry point — runs pipeline steps 1..N (default: 1-2).

Examples:
    python main.py --input video/in.mp4 --output video/out.mp4
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 1
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 3
    python main.py --config workspace/config.json --from-step 2
    python main.py --config workspace/config.json --from-step 3 --to-step 3
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 4
    python main.py --input video/in.mp4 --output video/out.mp4 --to-step 5

Exit codes:
    0 — success
    1 — input video not found
    2 — any other failure
    3 — FFmpeg/FFprobe missing (required from Step 2 on)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pipeline import __version__
from pipeline.checkpoint import CheckpointManager, steps_to_run
from pipeline.config import TARGET_FPS, TARGET_HEIGHT, TARGET_WIDTH, PipelineConfig
from pipeline.exceptions import (
    FFmpegNotFoundError,
    InputVideoNotFoundError,
)
from pipeline.logger import get_logger
from pipeline.perf import Profiler, VramSampler, nvidia_smi_probe
from pipeline.resources import is_frozen, prepare_frozen_environment
from pipeline.webui import parse_resolution, parse_tile_size
from pipeline.esrgan.weights import DEFAULT_ESRGAN_MODEL, ESRGAN_MODELS
from pipeline.rife.weights import DEFAULT_RIFE_VERSION, RIFE_VERSIONS
from pipeline.step01_environment import setup_environment
from pipeline.step02_frames import Step02Frames
from pipeline.step03_interpolate import Step03Interpolate
from pipeline.step04_upscale import Step04Upscale
from pipeline.step05_assemble import Step05Assemble
from pipeline.step06_cleanup import Step06Cleanup

log = get_logger("main")

LAST_STEP = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "720p -> 8K @ 1000 FPS pipeline. "
            "Step 1: environment check, GPU analysis, central config. "
            "Step 2: metadata probe, audio + frame extraction. "
            "Step 3: RIFE interpolation to target FPS. "
            "Step 4: Real-ESRGAN upscale to 8K. "
            "Step 5: FFmpeg assembly + verification. "
            "Step 6: Safe cleanup of temporary data. "
            "Step 7: checkpoint & resume is always on. "
            "Step 8: --resolution / --model shortcuts, --ui for the WebUI."
        )
    )
    # WebUI launcher (Step 8).
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__} (720p -> 8K @ 1000 FPS pipeline)",
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch the Gradio WebUI instead of running the CLI pipeline.",
    )
    parser.add_argument(
        "--ui-port", type=int, default=7860, help="WebUI port (default: 7860).",
    )
    parser.add_argument(
        "--ui-share", action="store_true",
        help="Create a public gradio.live link for the WebUI.",
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
        "--from-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
        help="First step to run (default: 1; >1 needs --config).",
    )
    parser.add_argument(
        "--to-step", type=int, default=2, choices=[1, 2, 3, 4, 5, 6],
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
    parser.add_argument(
        "--resolution", default=None,
        help="Target resolution shortcut: 8K, 4K, 1080p, 720p or WIDTHxHEIGHT "
        "(default: 8K; explicit --target-width/--target-height win).",
    )
    parser.add_argument(
        "--target-width", type=int, default=None,
        help="Exact target width (default: from --resolution, else 7680).",
    )
    parser.add_argument(
        "--target-height", type=int, default=None,
        help="Exact target height (default: from --resolution, else 4320).",
    )
    parser.add_argument(
        "--no-auto-install", action="store_true",
        help="Do not pip-install missing Python packages automatically.",
    )
    parser.add_argument(
        "--tile-size", default=None,
        help="VRAM tile size: 'auto' (default, Step 1 decides) or int (256/512/1024).",
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
    # Step 4 options.
    parser.add_argument(
        "--upscale-backend", choices=["esrgan", "onnx", "resize"], default="esrgan",
        help="'esrgan': AI upscale (needs torch); 'onnx': AI via ONNX Runtime "
        "(needs onnxruntime + .onnx weights); 'resize': non-AI smoke test.",
    )
    parser.add_argument(
        "--esrgan-model", choices=sorted(ESRGAN_MODELS), default=DEFAULT_ESRGAN_MODEL,
        help=f"Real-ESRGAN weights (default: {DEFAULT_ESRGAN_MODEL}).",
    )
    parser.add_argument(
        "--model", dest="esrgan_model", choices=sorted(ESRGAN_MODELS),
        default=argparse.SUPPRESS,
        help="Alias for --esrgan-model (Real-CUGAN planned for later).",
    )
    parser.add_argument(
        "--esrgan-weights", default=None,
        help="Explicit Real-ESRGAN weights file (.pth for torch, .onnx for "
        "ONNX Runtime — auto-selected); skips download.",
    )
    parser.add_argument(
        "--upscale-tile", type=int, default=None,
        help="Tile size for VRAM-safe upscale (default: Step 1 tile_size; 0 = off).",
    )
    parser.add_argument(
        "--tile-pad", type=int, default=10,
        help="Halo around each tile to avoid seams (default: 10).",
    )
    parser.add_argument(
        "--upscale-format", choices=["png", "jpg"], default="png",
        help="8K frame format (default: png).",
    )
    parser.add_argument(
        "--upscale-cache-every", type=int, default=10,
        help="Run torch.cuda.empty_cache() every N frames (default: 10).",
    )
    parser.add_argument(
        "--writer-queue", type=int, default=8,
        help="Async RAM writer FIFO depth for steps 3-4 (default: 8; "
        "0 = synchronous writes, debug only).",
    )
    # Step 5 options.
    parser.add_argument(
        "--video-codec",
        choices=["auto", "hevc_nvenc", "libx265", "av1_nvenc", "libsvtav1"],
        default="auto",
        help="Output encoder: auto = hevc_nvenc on CUDA else libx265.",
    )
    parser.add_argument(
        "--crf", type=float, default=19,
        help="Quality factor 0..63, lower = better (default: 19).",
    )
    parser.add_argument(
        "--encoder-preset", default=None,
        help="Override the encoder preset (e.g. ultrafast, p7, 8).",
    )
    parser.add_argument(
        "--ffmpeg-args", default=None,
        help="Extra raw ffmpeg args, e.g. \"-x265-params log-level=error\".",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip output resolution/FPS verification.",
    )
    # Step 6 options.
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Step 6: report what would be deleted without deleting.",
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="Step 6: keep temp_raw_frames.",
    )
    parser.add_argument(
        "--keep-interpolated", action="store_true",
        help="Step 6: keep interpolated_720p.",
    )
    parser.add_argument(
        "--keep-upscaled", action="store_true", help="Step 6: keep upscaled_8k.",
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="Step 6: keep the extracted audio file.",
    )
    # Step 7 options (checkpoint & resume).
    parser.add_argument(
        "--resume", choices=["ask", "yes", "no"], default="ask",
        help="Unfinished run found: 'ask' prompts (auto-resumes when "
        "non-interactive), 'yes' always resumes, 'no' discards progress.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Record pipeline_state.json every N pairs/frames (default: 10).",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable Step 7 checkpoint tracking entirely.",
    )
    # Step 9 options (GPU acceleration + performance profiling).
    parser.add_argument(
        "--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto",
        help="AI inference precision (default: auto = fp16 on CUDA, fp32 on "
        "CPU; bf16 needs Ampere+; explicit value beats --fp32).",
    )
    parser.add_argument(
        "--accel", choices=["auto", "none", "onnx"], default="auto",
        help="Step 4 acceleration: auto (.onnx weights switch to ONNX "
        "Runtime), none (force PyTorch), onnx (require ONNX Runtime).",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Time every stage (disk read/inference/disk write) and print a "
        "benchmark report (ms/frame, VRAM, processing vs target FPS).",
    )
    parser.add_argument(
        "--cprofile", action="store_true",
        help="Also run a cProfile pass over the run (saved to the workspace).",
    )
    parser.add_argument(
        "--torch-profile", action="store_true",
        help="Also capture torch.profiler kernel tables for steps 3-4.",
    )
    parser.add_argument(
        "--no-async-transfer", action="store_true",
        help="Disable CUDA streams + non-blocking transfers (debug switch).",
    )
    return parser


def _save_and_print_benchmark(
    config: PipelineConfig,
    profiler: Profiler,
    wall_s: float,
    step_wall: dict[int, float],
    vram_summary: dict | None,
    cprof_text: str | None,
    log: logging.Logger,
) -> None:
    """Build the Step 9 benchmark report, save it, print the summary."""
    from pipeline.perf import build_benchmark_report

    report = build_benchmark_report(
        target_fps=config.target_fps,
        target_size=[config.target_width, config.target_height],
        profiler=profiler,
        step_wall=step_wall,
        step_frames={
            3: config.interpolated_frame_count or 0,
            4: config.upscaled_frame_count or 0,
        },
        vram_summary=vram_summary,
    )
    saved = report.save(config.workspace_root / "benchmark.json")
    log.info("Benchmark report saved: %s (run wall time %.1fs)", saved, wall_s)
    print(report.render_text())
    if profiler.torch_tables:
        print()
        print(profiler.render_torch())
    if cprof_text is not None:
        cprof_path = config.workspace_root / "cprofile.txt"
        cprof_path.write_text(cprof_text, encoding="utf-8")
        log.info("cProfile table saved: %s", cprof_path)
        print()
        print(cprof_text)


def main(argv: list[str] | None = None) -> int:
    prepare_frozen_environment()  # no-op on dev runs; PATH/DLLs when frozen
    if argv is None and is_frozen() and len(sys.argv) <= 1:
        # Double-clicked EXE with no arguments: open the GUI, not an error.
        import app

        app.launch_ui(inbrowser=True)
        return 0
    args = build_parser().parse_args(argv)
    if args.ui:
        import app

        app.launch_ui(port=args.ui_port, share=args.ui_share)
        return 0
    try:
        res_width, res_height = (
            parse_resolution(args.resolution) if args.resolution
            else (TARGET_WIDTH, TARGET_HEIGHT)
        )
        target_width = args.target_width or res_width
        target_height = args.target_height or res_height
        tile_override = parse_tile_size(args.tile_size)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if args.step1_only:
        args.to_step = 1
    if args.from_step > args.to_step:
        log.error("--from-step (%d) > --to-step (%d).", args.from_step, args.to_step)
        return 2
    if args.accel == "onnx" and args.upscale_backend == "resize":
        log.error("--accel onnx needs an AI upscale backend (--upscale-backend esrgan|onnx).")
        return 2
    if args.fp32 and args.precision != "auto":
        log.warning("--fp32 is ignored because --precision %s is explicit.", args.precision)
    try:
        use_checkpoint = not args.no_checkpoint
        manager: CheckpointManager | None = None
        decision = None  # ResumeDecision when checkpointing is on
        skip_through = 0  # completed steps to skip (full-run resume only)

        step_wall: dict[int, float] = {}  # Step 9: per-step wall time

        def _tracked(step_num: int, run_step):  # noqa: ANN001, ANN202 - tiny local helper
            started = time.perf_counter()
            try:
                run_step()
            finally:
                step_wall[step_num] = (
                    step_wall.get(step_num, 0.0) + (time.perf_counter() - started)
                )
            if manager is not None:
                manager.mark_step_complete(step_num)
                manager.refresh_snapshot(config)

        if args.from_step > 1:
            if not args.config:
                log.error("--from-step %d requires --config <saved config.json>.", args.from_step)
                return 2
            config = PipelineConfig.load(args.config)
            log.info("Resuming from saved config: %s", args.config)
            if use_checkpoint:
                # Explicit step ranges always run as asked; the checkpoint
                # only tracks progress (no step-skipping here).
                manager = CheckpointManager(config.workspace_root, logger=log)
                snapshot = manager.snapshot_from_config(config)
                decision = manager.decide(snapshot, policy=args.resume)
                log.info("Checkpoint: %s.", decision.reason)
                if decision.action == "overwrite":
                    manager.discard_progress(from_step=args.from_step)
                    manager.begin_run(snapshot, args.from_step)
                elif decision.action == "resume":
                    manager.attach(decision.state)
                else:
                    manager.begin_run(snapshot, args.from_step)
        else:
            if not args.input or not args.output:
                log.error("--input and --output are required (or use --config to resume).")
                return 2
            if use_checkpoint:
                manager = CheckpointManager(Path(args.workspace), logger=log)
                desired = manager.snapshot_from_values(
                    args.input, args.output,
                    args.target_fps, target_width, target_height,
                )
                decision = manager.decide(desired, policy=args.resume)
                log.info("Checkpoint: %s.", decision.reason)
                if decision.action == "overwrite":
                    manager.discard_progress(from_step=1)
            saved_config = Path(args.workspace) / "config.json"
            if (
                use_checkpoint
                and decision.action == "resume"
                and saved_config.is_file()
            ):
                # Resume WITHOUT re-probing: keeps Step 2 metadata intact.
                config = PipelineConfig.load(saved_config)
                manager.attach(decision.state)
                skip_through = decision.skip_through_step
                log.info("Resuming run from saved config: %s", saved_config)
            else:
                config = setup_environment(
                    input_video_path=args.input,
                    final_output_path=args.output,
                    workspace_root=args.workspace,
                    target_fps=args.target_fps,
                    target_width=target_width,
                    target_height=target_height,
                    auto_install=not args.no_auto_install,
                    tile_size_override=tile_override,
                    save_config=not args.no_save_config,
                    logger=log,
                )
                if use_checkpoint:
                    if decision.action == "resume":
                        # config.json was lost; the state file backs up Step 2
                        # probe results — restore them so frame loops can run.
                        manager.attach(decision.state)
                        restored = manager.restore_metadata(config, decision.state)
                        config.save(saved_config)
                        skip_through = decision.skip_through_step
                        log.info(
                            "Restored %d metadata field(s) from the checkpoint: %s.",
                            len(restored), ", ".join(restored) or "none",
                        )
                    else:
                        manager.begin_run(
                            manager.snapshot_from_config(config), args.from_step
                        )
                if args.to_step < 2:
                    if manager is not None:
                        manager.mark_step_complete(1)
                        manager.refresh_snapshot(config)
                    return 0

        planned = (
            steps_to_run(args.from_step, args.to_step, skip_through)
            if use_checkpoint and args.from_step == 1
            else list(range(args.from_step, args.to_step + 1))
        )
        if use_checkpoint and args.from_step == 1 and skip_through:
            skipped = [n for n in range(args.from_step, args.to_step + 1) if n <= skip_through]
            if skipped:
                log.info(
                    "Skipping already-completed step(s) %s (per pipeline_state.json).",
                    skipped,
                )
        if not planned:
            log.info("All requested steps are already complete — nothing to do.")
            return 0

        # Step 9: profiling is opt-in; without flags nothing is sampled
        # and the hot path is untouched (profiler=None => nullcontext spans).
        profiling = args.profile or args.cprofile or args.torch_profile
        profiler = (
            Profiler(torch_profile=args.torch_profile) if profiling else None
        )
        vram = VramSampler(nvidia_smi_probe, interval=1.0)
        vram_started = False
        wall_start = time.perf_counter()
        if profiler is not None:
            if args.cprofile:
                profiler.start_cprofile()
            vram.start()
            vram_started = True
        try:
            if 2 in planned:
                _tracked(2, lambda: Step02Frames(
                    image_format=args.image_format,
                    audio_format=args.audio_format,
                    clean_frame_dir=not args.keep_old_frames,
                    logger=log,
                ).run(config))

            if 3 in planned:
                _tracked(3, lambda: Step03Interpolate(
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
                    checkpoint=manager,
                    checkpoint_every=args.checkpoint_every,
                    writer_queue=args.writer_queue,
                    profiler=profiler,
                    precision=args.precision,
                    async_transfers=not args.no_async_transfer,
                ).run(config))

            if 4 in planned:
                _tracked(4, lambda: Step04Upscale(
                    backend=args.upscale_backend,
                    model=args.esrgan_model,
                    weights=args.esrgan_weights,
                    weights_root="weights",
                    device=args.device,
                    fp16=False if args.fp32 else None,
                    tile=args.upscale_tile,
                    tile_pad=args.tile_pad,
                    output_format=args.upscale_format,
                    empty_cache_every=args.upscale_cache_every,
                    writer_queue=args.writer_queue,
                    logger=log,
                    checkpoint=manager,
                    checkpoint_every=args.checkpoint_every,
                    profiler=profiler,
                    precision=args.precision,
                    async_transfers=not args.no_async_transfer,
                    accel=args.accel,
                ).run(config))

            if 5 in planned:
                _tracked(5, lambda: Step05Assemble(
                    video_codec=args.video_codec,
                    crf=args.crf,
                    encoder_preset=args.encoder_preset,
                    ffmpeg_args=args.ffmpeg_args,
                    verify=not args.skip_verify,
                    logger=log,
                ).run(config))

            if 6 in planned:
                _tracked(6, lambda: Step06Cleanup(
                    dry_run=args.dry_run,
                    keep_raw=args.keep_raw,
                    keep_interpolated=args.keep_interpolated,
                    keep_upscaled=args.keep_upscaled,
                    keep_audio=args.keep_audio,
                    logger=log,
                ).run(config))
        finally:
            wall_s = time.perf_counter() - wall_start
            if profiler is not None:
                cprof_text = profiler.stop_cprofile() if args.cprofile else None
                if vram_started:
                    vram.stop()
                try:
                    _save_and_print_benchmark(
                        config, profiler, wall_s, step_wall,
                        vram.summary(), cprof_text, log,
                    )
                except Exception as exc:  # noqa: BLE001 - report must not mask run errors
                    log.warning("Benchmark report failed: %s", exc)
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
