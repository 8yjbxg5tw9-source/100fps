"""720p 30/60 FPS -> 8K (7680x4320) @ 1000 FPS local video pipeline.

This package hosts the 10-step pipeline. Step 1 (environment init, hardware
analysis, central configuration) is the entry point; Step 2 (analysis, audio
and frame extraction) consumes the config; Step 3 (RIFE interpolation to
1000 FPS) consumes the raw frames; Step 4 (Real-ESRGAN 8K upscale) consumes
the interpolated frames; Step 5 (FFmpeg assembly + verification) produces the
final video; Step 6 (safe cleanup) reclaims workspace disk space; later steps
build on all of them.
"""

from pipeline.config import PipelineConfig
from pipeline.step01_environment import Step01Environment, setup_environment
from pipeline.step02_frames import Step02Frames, Step02Result, VideoMetadata
from pipeline.step03_interpolate import Step03Interpolate, Step03Result
from pipeline.step04_upscale import Step04Upscale, Step04Result
from pipeline.step05_assemble import Step05Assemble, Step05Result
from pipeline.step06_cleanup import Step06Cleanup, Step06Result

__version__ = "0.6.0"  # Steps 1-6 complete
__all__ = [
    "PipelineConfig",
    "Step01Environment",
    "setup_environment",
    "Step02Frames",
    "Step02Result",
    "VideoMetadata",
    "Step03Interpolate",
    "Step03Result",
    "Step04Upscale",
    "Step04Result",
    "Step05Assemble",
    "Step05Result",
    "Step06Cleanup",
    "Step06Result",
]
