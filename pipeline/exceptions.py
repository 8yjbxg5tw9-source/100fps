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
