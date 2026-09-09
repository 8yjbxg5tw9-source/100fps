"""720p 30/60 FPS -> 8K (7680x4320) @ 1000 FPS local video pipeline.

This package hosts the 10-step pipeline. Step 1 (environment init, hardware
analysis, central configuration) is the entry point; Step 2 (analysis, audio
and frame extraction) consumes the config; Step 3 (RIFE interpolation to
1000 FPS) consumes the raw frames; Step 4 (Real-ESRGAN 8K upscale) consumes
the interpolated frames; Step 5 (FFmpeg assembly + verification) produces the
final video; Step 6 (safe cleanup) reclaims workspace disk space; Step 7
(checkpoint & resume) makes every run crash-proof; Step 8 (CLI shortcuts +
Gradio WebUI backend) makes every run one click away; Step 9 (GPU
acceleration + profiling) makes every run fast; Step 10 (packaging) ships
every run as a standalone app.
"""

from pipeline.checkpoint import (
    CheckpointManager,
    PipelineState,
    ResumeDecision,
    StepProgress,
)
from pipeline.config import PipelineConfig
from pipeline.step01_environment import Step01Environment, setup_environment
from pipeline.step02_frames import Step02Frames, Step02Result, VideoMetadata
from pipeline.step03_interpolate import Step03Interpolate, Step03Result
from pipeline.step04_upscale import Step04Upscale, Step04Result
from pipeline.step05_assemble import Step05Assemble, Step05Result
from pipeline.step06_cleanup import Step06Cleanup, Step06Result
from pipeline.webui import ProgressBus, RunOptions, parse_resolution

__version__ = "1.0.0"  # Steps 1-10 complete: first commercial-grade release
__all__ = [
    "CheckpointManager",
    "PipelineConfig",
    "PipelineState",
    "ProgressBus",
    "ResumeDecision",
    "RunOptions",
    "Step01Environment",
    "setup_environment",
    "Step02Frames",
    "Step02Result",
    "StepProgress",
    "VideoMetadata",
    "Step03Interpolate",
    "Step03Result",
    "Step04Upscale",
    "Step04Result",
    "Step05Assemble",
    "Step05Result",
    "Step06Cleanup",
    "Step06Result",
    "parse_resolution",
]
