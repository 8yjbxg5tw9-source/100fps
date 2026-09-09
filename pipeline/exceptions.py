"""Custom exception hierarchy for the pipeline.

All pipeline errors derive from :class:`PipelineError` so callers can catch a
single type. ``FileNotFoundError`` semantics for a missing input video are
preserved by multiple inheritance.
"""


class PipelineError(Exception):
    """Base class for all pipeline failures."""


class InputVideoNotFoundError(PipelineError, FileNotFoundError):
    """Raised when the input video path does not exist or is not a file."""


class FFmpegNotFoundError(PipelineError):
    """Raised when FFmpeg is required but not installed (used from Step 2 on).

    Step 1 itself only *warns* about a missing FFmpeg so that environment
    probing can still complete; actual frame extraction (Step 2) will raise.
    """


class DependencyInstallError(PipelineError):
    """Raised when an automatic ``pip install`` of a missing package fails."""


class MetadataProbeError(PipelineError):
    """Raised when video metadata (FPS, size, ...) cannot be determined.

    This happens when neither FFprobe nor the OpenCV fallback can read the
    input file (missing tools, corrupt/unsupported container, ...).
    """


class FrameExtractionError(PipelineError):
    """Raised when FFmpeg frame extraction fails (Step 2, critical path).

    Unlike audio extraction (auxiliary -- Step 8 merges audio only if the
    file exists), frames are mandatory for Step 3, so any failure aborts.
    """


class InterpolationError(PipelineError):
    """Raised when Step 3 cannot interpolate (too few frames, exp overflow)."""


class RifeWeightsError(PipelineError):
    """Raised when RIFE weights are missing, invalid or not downloadable."""


class RifeInferenceError(PipelineError):
    """Raised when the interpolation backend fails (no torch/CUDA, OOM, ...)."""


class UpscaleError(PipelineError):
    """Raised when Step 4 cannot upscale (no input frames, bad target, ...)."""


class EsrganWeightsError(PipelineError):
    """Raised when Real-ESRGAN weights are missing, invalid or not downloadable."""


class EsrganInferenceError(PipelineError):
    """Raised when the upscaling backend fails (no torch/CUDA, OOM, ...)."""
