"""720p 30/60 FPS -> 8K (7680x4320) @ 1000 FPS local video pipeline.

This package hosts the 10-step pipeline. Step 1 (environment init, hardware
analysis, central configuration) is the entry point; Step 2 (analysis, audio
and frame extraction) consumes the config; later steps build on both.
"""

from pipeline.config import PipelineConfig
from pipeline.step01_environment import Step01Environment, setup_environment
from pipeline.step02_frames import Step02Frames, Step02Result, VideoMetadata

__version__ = "0.2.0"  # Steps 1-2 complete
__all__ = [
    "PipelineConfig",
    "Step01Environment",
    "setup_environment",
    "Step02Frames",
    "Step02Result",
    "VideoMetadata",
]
