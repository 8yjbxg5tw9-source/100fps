"""720p 30/60 FPS -> 8K (7680x4320) @ 1000 FPS local video pipeline.

This package hosts the 10-step pipeline. Step 1 (environment init, hardware
analysis, central configuration) is the entry point; later steps consume the
:class:`pipeline.config.PipelineConfig` object it produces.
"""

from pipeline.config import PipelineConfig
from pipeline.step01_environment import Step01Environment, setup_environment

__version__ = "0.1.0"  # Step 1 complete
__all__ = ["PipelineConfig", "Step01Environment", "setup_environment"]
