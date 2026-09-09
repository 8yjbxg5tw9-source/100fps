"""Central configuration object for the whole 10-step pipeline.

:class:`PipelineConfig` is created by Step 1
(:mod:`pipeline.step01_environment`) and then passed to every later step, so
all steps agree on paths, target resolution/FPS and the VRAM-derived tiling
parameter. It can be serialised to ``workspace/config.json`` for handoff
between processes (e.g. Step 2 reading what Step 1 prepared).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Pipeline-wide constants (targets of the full 10-step program)
# ---------------------------------------------------------------------------
TARGET_FPS: float = 1000.0
TARGET_WIDTH: int = 7680   # 8K UHD width
TARGET_HEIGHT: int = 4320  # 8K UHD height

# Workspace sub-folders (relative to ``workspace_root``). Step 1 creates them,
# later steps use them as fixed handoff points.
DIR_RAW_FRAMES = "temp_raw_frames"    # Step 2 output: original frames from FFmpeg
DIR_INTERPOLATED = "interpolated_720p"  # Step 3 output: e.g. 30fps -> 1000fps frames
DIR_UPSCALED_8K = "upscaled_8k"       # Step 4 output: 8K frames ready for encode

# Step 2 artefacts (relative to ``workspace_root`` unless stated otherwise).
AUDIO_FILENAME_WAV = "input_audio.wav"  # lossless extraction (default)
AUDIO_FILENAME_AAC = "input_audio.aac"  # lossy fallback / explicit choice
CONFIG_FILENAME = "config.json"         # Step N -> Step N+1 handoff file
FRAME_PATTERN_PNG = "frame_%06d.png"    # lossless frames (default)
FRAME_PATTERN_JPG = "frame_%06d.jpg"    # high-quality lossy alternative
INTERPOLATED_FRAME_PATTERN = "frame_%08d.png"  # Step 3 output (8 digits: 1000fps!)
UPSCALED_FRAME_PATTERN = "frame_8k_%08d.png"     # Step 4 output (8K frames)


@dataclass
class PipelineConfig:
    """Single source of truth for paths, targets and hardware decisions."""

    # -- Required paths ------------------------------------------------------
    input_video_path: Path
    final_output_path: Path

    # -- Workspace -----------------------------------------------------------
    workspace_root: Path = field(default=Path("workspace"))

    # -- Global targets ------------------------------------------------------
    target_fps: float = TARGET_FPS
    target_width: int = TARGET_WIDTH
    target_height: int = TARGET_HEIGHT

    # -- Hardware-derived settings (filled in by Step 1) ---------------------
    tile_size: int = 256          # VRAM-based tiling; see Step01Environment
    device: str = "cpu"           # "cuda" or "cpu"
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    ffmpeg_path: Optional[str] = None
    ffmpeg_available: bool = False
    ffmpeg_version: Optional[str] = None

    # -- Source media analysis (filled in by Step 2) ---------------------------
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    original_fps: Optional[float] = None
    total_frames: Optional[int] = None
    duration_sec: Optional[float] = None
    # Multiplier needed to reach target_fps, e.g. 30 -> 1000 = ~33.33x.
    interpolation_factor: Optional[float] = None
    estimated_interpolated_frames: Optional[int] = None
    # -- Step 2 artefacts ------------------------------------------------------
    has_audio: Optional[bool] = None
    audio_path: Optional[Path] = None
    extracted_frame_count: Optional[int] = None
    frame_pattern: Optional[str] = None
    # Full probe dump (ffprobe/OpenCV) for Step 3+ debugging.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Interpolation (filled in by Step 3) ---------------------------------
    interpolation_exp: Optional[int] = None      # N in the 2^N dense pass
    interpolated_frame_count: Optional[int] = None  # exact 1000fps frames
    interpolation_backend: Optional[str] = None  # "rife" | "blend"

    # -- Super-resolution (filled in by Step 4) ------------------------------
    esrgan_model: Optional[str] = None            # "x4plus" | "x4plus-anime"
    esrgan_backend: Optional[str] = None          # "esrgan" | "resize"
    esrgan_tile: Optional[int] = None             # tile actually used (0 = off)
    upscaled_frame_count: Optional[int] = None    # 8K frames written
    upscaled_frame_pattern: Optional[str] = None

    def __post_init__(self) -> None:
        # Accept plain strings for convenience.
        self.input_video_path = Path(self.input_video_path)
        self.final_output_path = Path(self.final_output_path)
        self.workspace_root = Path(self.workspace_root)
        if self.audio_path is not None:
            self.audio_path = Path(self.audio_path)

        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError(
                f"target resolution must be positive, got "
                f"{self.target_width}x{self.target_height}"
            )
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")

    # -- Derived paths --------------------------------------------------------
    @property
    def temp_raw_frames(self) -> Path:
        """Step 2 output / Step 3 input: original decoded frames."""
        return self.workspace_root / DIR_RAW_FRAMES

    @property
    def interpolated_720p(self) -> Path:
        """Step 3 output / Step 4 input: interpolated (1000 FPS) frames."""
        return self.workspace_root / DIR_INTERPOLATED

    @property
    def upscaled_8k(self) -> Path:
        """Step 4 output / Step 7 input: upscaled 8K frames."""
        return self.workspace_root / DIR_UPSCALED_8K

    @property
    def target_resolution(self) -> Tuple[int, int]:
        return (self.target_width, self.target_height)

    @property
    def target_resolution_str(self) -> str:
        return f"{self.target_width}x{self.target_height}"

    # -- Filesystem ------------------------------------------------------------
    def ensure_directories(self, create_output_parent: bool = True) -> "PipelineConfig":
        """Create workspace + temp folders (and output parent) if missing."""
        for folder in (
            self.workspace_root,
            self.temp_raw_frames,
            self.interpolated_720p,
            self.upscaled_8k,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        if create_output_parent and self.final_output_path.parent != Path(""):
            self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    # -- (De)serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("input_video_path", "final_output_path", "workspace_root"):
            data[key] = str(data[key])
        if data.get("audio_path") is not None:
            data["audio_path"] = str(data["audio_path"])
        return data

    def save(self, path: str | Path) -> Path:
        """Write config as JSON (used for Step 1 -> Step 2 handoff)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        """Read a config previously written with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    # -- Pretty printing ---------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "PipelineConfig:",
            f"  input_video_path : {self.input_video_path}",
            f"  final_output_path: {self.final_output_path}",
            f"  workspace_root   : {self.workspace_root}",
            f"  temp_raw_frames  : {self.temp_raw_frames}",
            f"  interpolated_720p: {self.interpolated_720p}",
            f"  upscaled_8k      : {self.upscaled_8k}",
            f"  target           : {self.target_resolution_str} @ {self.target_fps} FPS",
            f"  device           : {self.device}"
            + (f" ({self.gpu_name}, {self.vram_gb:.1f} GB VRAM)" if self.gpu_name else ""),
            f"  tile_size        : {self.tile_size}",
            f"  ffmpeg           : "
            + (
                f"{self.ffmpeg_version} ({self.ffmpeg_path})"
                if self.ffmpeg_available
                else "NOT FOUND"
            ),
        ]
        # Step 2 section (only once Step 2 has run).
        if self.original_fps:
            lines.append(
                f"  source           : {self.source_width}x{self.source_height} "
                f"@ {self.original_fps:.2f} FPS, "
                f"{self.total_frames if self.total_frames is not None else '?'} frames"
                + (
                    f", {self.duration_sec:.2f}s"
                    if self.duration_sec is not None
                    else ""
                )
            )
            if self.interpolation_factor:
                lines.append(
                    f"  interpolation    : {self.interpolation_factor:.2f}x "
                    f"(-> ~{self.estimated_interpolated_frames} frames @ "
                    f"{self.target_fps} FPS)"
                    if self.estimated_interpolated_frames
                    else f"  interpolation    : {self.interpolation_factor:.2f}x"
                )
            lines.append(
                f"  audio            : "
                + (str(self.audio_path) if self.has_audio else "none")
            )
            lines.append(
                f"  raw frames       : "
                + (
                    f"{self.extracted_frame_count} files ({self.frame_pattern})"
                    if self.extracted_frame_count is not None
                    else "not extracted yet"
                )
            )
        # Step 3 section (only once Step 3 has run).
        if self.interpolation_exp is not None:
            lines.append(
                f"  interp (Step 3)  : exp={self.interpolation_exp} "
                f"(2^{self.interpolation_exp}x dense), backend={self.interpolation_backend}, "
                + (
                    f"{self.interpolated_frame_count} frames @ {self.target_fps} FPS"
                    if self.interpolated_frame_count is not None
                    else "not interpolated yet"
                )
            )
        # Step 4 section (only once Step 4 has run).
        if self.upscaled_frame_count is not None:
            lines.append(
                f"  upscale (Step 4)   : {self.esrgan_model} via {self.esrgan_backend}, "
                f"tile={self.esrgan_tile}, "
                f"{self.upscaled_frame_count} frames ({self.upscaled_frame_pattern})"
            )
        return "\n".join(lines)
