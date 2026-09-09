"""STEP 1 — System environment init, hardware analysis, central config.

Running this step probes the machine and produces the :class:`PipelineConfig`
object consumed by Step 2 (FFmpeg frame extraction) and all later steps:

1. Validates the input video exists (``FileNotFoundError`` otherwise).
2. Checks ``FFmpeg`` availability (warns with install guide if missing).
3. Checks Python deps (``torch``, ``torchvision``, ``opencv-python``,
   ``numpy``, ``Pillow``, ``tqdm``) and auto-installs missing ones via pip.
4. Detects NVIDIA CUDA GPU + VRAM size (``torch.cuda`` first,
   ``nvidia-smi`` fallback, CPU mode otherwise).
5. Derives ``tile_size`` from VRAM so 8K frames never overflow GPU memory.
6. Creates workspace folders and fills in the central config.

Typical usage::

    from pipeline.step01_environment import setup_environment

    config = setup_environment("in.mp4", "out.mp4")
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.base import PipelineStep
from pipeline.config import PipelineConfig
from pipeline.exceptions import DependencyInstallError, InputVideoNotFoundError
from pipeline.logger import get_logger

# ---------------------------------------------------------------------------
# Install guide shown when FFmpeg is missing (Step 1 warns, Step 2 requires).
# ---------------------------------------------------------------------------
FFMPEG_INSTALL_GUIDE = (
    "FFmpeg was NOT found on PATH. It is required for Step 2 (frame "
    "extraction) and Step 7 (video encoding). Install it:\n"
    "  - Ubuntu/Debian : sudo apt update && sudo apt install -y ffmpeg\n"
    "  - macOS         : brew install ffmpeg\n"
    "  - Windows       : winget install Gyan.FFmpeg  "
    "(or: choco install ffmpeg)"
)


@dataclass
class FFmpegInfo:
    available: bool
    path: Optional[str] = None
    version: Optional[str] = None


@dataclass
class GPUInfo:
    has_cuda: bool
    device: str  # "cuda" or "cpu"
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    vram_bytes: Optional[int] = None
    torch_available: bool = False
    source: str = "none"  # "torch.cuda" | "nvidia-smi" | "none"


@dataclass
class DependencyStatus:
    import_name: str
    pip_name: str
    installed: bool
    version: Optional[str] = None
    auto_installed: bool = False


class Step01Environment(PipelineStep[PipelineConfig]):
    """Step 1 implementation. See module docstring for the full contract."""

    name = "step01_environment"

    #: import name -> pip package name (checked in this order).
    REQUIRED_PACKAGES: Dict[str, str] = {
        "torch": "torch",
        "torchvision": "torchvision",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "PIL": "Pillow",
        "tqdm": "tqdm",
    }

    # VRAM (GB) -> tile_size mapping for 8K tiled inference.
    TILE_SMALL = 256    # VRAM < 8 GB, or CPU mode (safe fallback)
    TILE_MEDIUM = 512   # 8 GB <= VRAM <= 16 GB
    TILE_LARGE = 1024   # VRAM > 16 GB
    VRAM_SMALL_MAX = 8.0
    VRAM_MEDIUM_MAX = 16.0

    SUPPORTED_VIDEO_EXTENSIONS = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg",
    }

    def __init__(
        self,
        auto_install: bool = True,
        tile_size_override: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.auto_install = auto_install
        self.tile_size_override = tile_size_override
        self.log = logger or get_logger(__name__)
        self.ffmpeg_info: Optional[FFmpegInfo] = None
        self.gpu_info: Optional[GPUInfo] = None
        self.dependencies: List[DependencyStatus] = []

    # -- Step orchestration ---------------------------------------------------
    def run(self, config: PipelineConfig) -> PipelineConfig:
        """Probe the system, fill in ``config`` hardware fields, make dirs."""
        self.log.info("=== Step 1: Environment initialization started ===")

        # 1. Fail fast on a missing input video.
        config.input_video_path = self.validate_input_video(config.input_video_path)

        # 2. FFmpeg availability (warn only -- Step 2 will hard-require it).
        self.ffmpeg_info = self.check_ffmpeg()
        config.ffmpeg_available = self.ffmpeg_info.available
        config.ffmpeg_path = self.ffmpeg_info.path
        config.ffmpeg_version = self.ffmpeg_info.version

        # 3. Python dependencies (auto-install missing ones via pip).
        self.dependencies = self.check_python_dependencies()

        # 4. GPU / VRAM analysis.
        self.gpu_info = self.detect_gpu()
        config.device = self.gpu_info.device
        config.gpu_name = self.gpu_info.gpu_name
        config.vram_gb = self.gpu_info.vram_gb

        # 5. Tiling parameter (manual override wins over the VRAM heuristic).
        if self.tile_size_override is not None:
            if self.tile_size_override <= 0:
                raise ValueError(
                    f"tile_size_override must be positive, "
                    f"got {self.tile_size_override}"
                )
            config.tile_size = self.tile_size_override
            self.log.info("tile_size manually overridden: %d", config.tile_size)
        else:
            config.tile_size = self.determine_tile_size(
                self.gpu_info.vram_gb, self.gpu_info.has_cuda
            )

        # 6. Workspace folders.
        self._prepare_directories(config)

        self.log.info("Step 1 completed. Config ready for Step 2.")
        self.log.info("\n%s", config.summary())
        return config

    # -- Input validation -------------------------------------------------------
    def validate_input_video(self, path: str | Path) -> Path:
        """Return resolved ``path`` or raise :class:`InputVideoNotFoundError`."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise InputVideoNotFoundError(
                f"Input video not found: {resolved}\n"
                f"Check --input / input_video_path and try again."
            )
        if resolved.suffix.lower() not in self.SUPPORTED_VIDEO_EXTENSIONS:
            self.log.warning(
                "Unusual video extension '%s' for %s -- continuing anyway. "
                "Supported: %s",
                resolved.suffix,
                resolved.name,
                sorted(self.SUPPORTED_VIDEO_EXTENSIONS),
            )
        else:
            self.log.info("Input video found: %s", resolved)
        return resolved

    # -- FFmpeg -------------------------------------------------------------------
    def check_ffmpeg(self) -> FFmpegInfo:
        """Detect the ``ffmpeg`` binary and parse its version string."""
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            self.log.warning("FFmpeg NOT found on PATH!")
            for line in FFMPEG_INSTALL_GUIDE.splitlines():
                self.log.warning(line)
            return FFmpegInfo(available=False)

        version = self._read_ffmpeg_version(ffmpeg_bin)
        if version:
            self.log.info("FFmpeg detected: %s (%s)", version, ffmpeg_bin)
        else:
            self.log.info("FFmpeg detected at %s (version unreadable)", ffmpeg_bin)
        return FFmpegInfo(available=True, path=ffmpeg_bin, version=version)

    @staticmethod
    def _read_ffmpeg_version(ffmpeg_bin: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                [ffmpeg_bin, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout:
            return None
        first_line = proc.stdout.splitlines()[0].strip()
        # e.g. "ffmpeg version 6.0 Copyright (c) ..." -> keep the head only.
        return first_line[:80]

    # -- Python dependencies ---------------------------------------------------------
    def check_python_dependencies(self) -> List[DependencyStatus]:
        """Verify imports; ``pip install`` whatever is missing (if enabled)."""
        report: List[DependencyStatus] = []
        for import_name, pip_name in self.REQUIRED_PACKAGES.items():
            if self._is_importable(import_name):
                version = self._installed_version(pip_name)
                self.log.info(
                    "Dependency OK: %s%s",
                    pip_name,
                    f" (v{version})" if version else "",
                )
                report.append(
                    DependencyStatus(import_name, pip_name, True, version, False)
                )
                continue

            self.log.warning("Dependency MISSING: %s (import '%s')", pip_name, import_name)
            auto_installed = False
            if self.auto_install:
                auto_installed = self._install_package(pip_name)
            else:
                self.log.warning(
                    "Auto-install disabled -- run: pip install %s", pip_name
                )

            still_missing = not self._is_importable(import_name)
            if still_missing:
                self.log.error(
                    "Dependency STILL missing: %s. Install manually with "
                    "'pip install %s' before running Step 3/4 (AI stages).",
                    pip_name,
                    pip_name,
                )
            report.append(
                DependencyStatus(
                    import_name,
                    pip_name,
                    installed=not still_missing,
                    version=self._installed_version(pip_name),
                    auto_installed=auto_installed and not still_missing,
                )
            )
        return report

    @staticmethod
    def _is_importable(import_name: str) -> bool:
        return importlib.util.find_spec(import_name) is not None

    @staticmethod
    def _installed_version(pip_name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _install_package(self, pip_name: str) -> bool:
        """Attempt ``python -m pip install <pkg>``; return True on success."""
        self.log.info("Installing missing package via pip: %s ...", pip_name)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True,
                text=True,
                timeout=600,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.error("pip install %s failed to launch: %s", pip_name, exc)
            return False
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = tail[-5:] if tail else ["unknown error"]
            self.log.error("pip install %s FAILED:\n%s", pip_name, "\n".join(tail))
            return False
        self.log.info("Successfully installed: %s", pip_name)
        return True

    # -- GPU / VRAM --------------------------------------------------------------------
    def detect_gpu(self) -> GPUInfo:
        """Detect CUDA GPU via torch, else nvidia-smi, else CPU fallback."""
        info = self._detect_gpu_via_torch()
        if info is not None:
            self._log_gpu_info(info)
            return info
        info = self._detect_gpu_via_nvidia_smi()
        if info is not None:
            self._log_gpu_info(info)
            return info
        fallback = GPUInfo(has_cuda=False, device="cpu", source="none")
        self.log.warning(
            "No CUDA GPU detected -- running in CPU mode. "
            "8K processing on CPU will be extremely slow; "
            "for an NVIDIA GPU install the CUDA-enabled PyTorch build from "
            "https://pytorch.org/get-started/locally/"
        )
        self._log_gpu_info(fallback)
        return fallback

    def _detect_gpu_via_torch(self) -> Optional[GPUInfo]:
        try:
            import torch  # lazy import: torch may not be installed yet
        except ImportError:
            self.log.warning("torch not importable -- skipping CUDA probe via torch.")
            return None
        try:
            if not torch.cuda.is_available():
                self.log.info("torch found but torch.cuda.is_available() is False.")
                return None
            props = torch.cuda.get_device_properties(0)
            total_bytes = int(props.total_memory)
            vram_gb = total_bytes / (1024**3)
            return GPUInfo(
                has_cuda=True,
                device="cuda",
                gpu_name=torch.cuda.get_device_name(0),
                vram_gb=vram_gb,
                vram_bytes=total_bytes,
                torch_available=True,
                source="torch.cuda",
            )
        except Exception as exc:  # noqa: BLE001 - probe must never crash Step 1
            self.log.warning("torch CUDA probe failed (%s) -- trying nvidia-smi.", exc)
            return None

    def _detect_gpu_via_nvidia_smi(self) -> Optional[GPUInfo]:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            return None
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            first = proc.stdout.strip().splitlines()[0]
            name, mem_mib = [part.strip() for part in first.split(",", 1)]
            vram_gb = float(mem_mib) / 1024.0
        except (ValueError, IndexError):
            return None
        self.log.info("torch CUDA unavailable, but nvidia-smi found a GPU.")
        return GPUInfo(
            has_cuda=False,  # torch can't use it, so pipeline device stays CPU
            device="cpu",
            gpu_name=name,
            vram_gb=vram_gb,
            vram_bytes=int(float(mem_mib) * 1024 * 1024),
            torch_available=self._is_importable("torch"),
            source="nvidia-smi",
        )

    def _log_gpu_info(self, info: GPUInfo) -> None:
        if info.has_cuda and info.gpu_name and info.vram_gb is not None:
            tile = self.determine_tile_size(info.vram_gb, True)
            self.log.info(
                "CUDA detected: %s (%.1f GB VRAM) -> tile_size=%d",
                info.gpu_name,
                info.vram_gb,
                tile,
            )
        elif info.gpu_name and info.vram_gb is not None:
            self.log.info(
                "GPU seen via %s: %s (%.1f GB VRAM), but torch cannot use CUDA "
                "-> device=cpu, tile_size=%d",
                info.source,
                info.gpu_name,
                info.vram_gb,
                self.determine_tile_size(info.vram_gb, False),
            )
        else:
            self.log.info(
                "Device=cpu, tile_size=%d (no GPU information available)",
                self.determine_tile_size(None, False),
            )

    # -- Tiling heuristic -----------------------------------------------------------------
    @classmethod
    def determine_tile_size(
        cls, vram_gb: Optional[float], has_cuda: bool
    ) -> int:
        """Map VRAM size to the 8K tiled-inference tile size.

        - VRAM < 8 GB (or unknown / CPU mode) -> 256
        - 8 GB <= VRAM <= 16 GB               -> 512
        - VRAM > 16 GB                        -> 1024
        """
        if not has_cuda or vram_gb is None:
            return cls.TILE_SMALL
        if vram_gb < cls.VRAM_SMALL_MAX:
            return cls.TILE_SMALL
        if vram_gb <= cls.VRAM_MEDIUM_MAX:
            return cls.TILE_MEDIUM
        return cls.TILE_LARGE

    # -- Workspace ---------------------------------------------------------------------------
    def _prepare_directories(self, config: PipelineConfig) -> None:
        for folder in (
            config.workspace_root,
            config.temp_raw_frames,
            config.interpolated_720p,
            config.upscaled_8k,
        ):
            existed = folder.exists()
            folder.mkdir(parents=True, exist_ok=True)
            self.log.info(
                "%s directory: %s",
                "Found" if existed else "Created",
                folder,
            )
        if config.final_output_path.parent != Path(""):
            config.final_output_path.parent.mkdir(parents=True, exist_ok=True)


def setup_environment(
    input_video_path: str | Path,
    final_output_path: str | Path,
    workspace_root: str | Path = "workspace",
    target_fps: float = 1000.0,
    target_width: int = 7680,
    target_height: int = 4320,
    auto_install: bool = True,
    tile_size_override: Optional[int] = None,
    save_config: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PipelineConfig:
    """One-call Step 1 entry point: build config, probe system, make dirs.

    Returns the ready-to-use :class:`PipelineConfig` for Step 2. When
    ``save_config`` is True (default), the config is also written to
    ``<workspace>/config.json`` for cross-process handoff.

    Raises:
        InputVideoNotFoundError: if ``input_video_path`` does not exist.
        DependencyInstallError: never raised directly (install failures are
            logged); kept documented for future ``strict=True`` behaviour.
    """
    _ = DependencyInstallError  # re-export hint for API consumers
    config = PipelineConfig(
        input_video_path=Path(input_video_path),
        final_output_path=Path(final_output_path),
        workspace_root=Path(workspace_root),
        target_fps=target_fps,
        target_width=target_width,
        target_height=target_height,
    )
    step = Step01Environment(
        auto_install=auto_install,
        tile_size_override=tile_size_override,
        logger=logger,
    )
    config = step.run(config)
    if save_config:
        saved = config.save(config.workspace_root / "config.json")
        step.log.info("Config saved for Step 2: %s", saved)
    return config
