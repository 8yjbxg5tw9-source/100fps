"""STEP 4 — Real-ESRGAN super-resolution to 8K UHD (7680x4320).

Reads the interpolated 720p frames from Step 3 (``interpolated_720p``), upscales
each one with Real-ESRGAN x4 on CUDA/fp16, and writes 8K frames to
``upscaled_8k`` for Step 5 (video assembly / export).

## Resolution math

720p (1280x720) x4 = 5120x2880 — not quite 8K. Like upstream's ``outscale``
option, the backend therefore finishes with an exact-size Lanczos resize
(``INTER_LANCZOS4``), i.e. **4x neural upscale + 1.5x classical resize** to
hit precisely ``7680x4320`` (or any configured target).

## VRAM safety

Whole-frame 8K inference overflows smaller GPUs, so frames are processed with
the upstream tiling procedure (haloed tiles, halo cropped after upscale —
seam-free by construction). The tile size defaults to Step 1's VRAM-derived
``tile_size`` (256/512/1024); ``tile=0`` selects whole-image inference.

## Throughput

- I/O never blocks the GPU: an :class:`AsyncFrameWriter` background thread
  persists 8K PNGs/JPGs while the next frame is inferred (bounded FIFO,
  order-preserving, fail-loud).
- ``torch.cuda.empty_cache()`` runs every ``empty_cache_every`` frames.
- Live ``tqdm`` progress shows frames done/total plus sec/frame.

Typical usage::

    from pipeline.config import PipelineConfig
    from pipeline.step04_upscale import Step04Upscale

    config = PipelineConfig.load("workspace/config.json")
    result = Step04Upscale(backend="esrgan", model="x4plus").run(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from pipeline.base import PipelineStep
from pipeline.config import (
    CONFIG_FILENAME,
    UPSCALED_FRAME_PATTERN,
    PipelineConfig,
)
from pipeline.exceptions import UpscaleError
from pipeline.logger import get_logger


@dataclass
class Step04Result:
    source_frames: int          # interpolated frames discovered
    written_count: int          # 8K frames actually written
    target_size: tuple          # (width, height), e.g. (7680, 4320)
    tile_used: int              # 0 = whole-image inference
    out_dir: Path
    frame_pattern: str
    backend_name: str
    model_name: Optional[str]   # None for the non-AI resize backend
    validation_ok: bool


class Step04Upscale(PipelineStep[Step04Result]):
    """Step 4 implementation. See module docstring for the full contract."""

    name = "step04_upscale"

    def __init__(
        self,
        backend: str = "esrgan",          # "esrgan" (AI) | "resize" (smoke)
        model: str = "x4plus",            # "x4plus" | "x4plus-anime"
        weights: Optional[str | Path] = None,
        weights_root: str | Path = "weights",
        device: Optional[str] = None,     # "cuda" | "cpu" | None (auto)
        fp16: Optional[bool] = None,      # None -> auto (True on CUDA)
        tile: Optional[int] = None,       # None -> config.tile_size; 0 = off
        tile_pad: int = 10,               # upstream halo default
        pre_pad: int = 0,                 # upstream inference-script default
        output_format: str = "png",       # "png" (lossless) | "jpg"
        clean_output_dir: bool = True,
        empty_cache_every: int = 10,
        writer_queue: int = 8,            # async writer FIFO depth
        backend_obj: Optional[Any] = None,  # injected backend (tests/users)
        frame_io: Optional[Any] = None,     # injected FrameIO (tests)
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if output_format not in ("png", "jpg"):
            raise ValueError(f"output_format must be 'png' or 'jpg', got {output_format!r}")
        if tile is not None and tile < 0:
            raise ValueError(f"tile must be >= 0, got {tile}.")
        self.backend_name = backend
        self.model_name = model
        self.weights = weights
        self.weights_root = weights_root
        self.device = device
        self.fp16 = fp16
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.output_format = output_format
        self.clean_output_dir = clean_output_dir
        self.empty_cache_every = empty_cache_every
        self.writer_queue = writer_queue
        self._backend = backend_obj
        self._frame_io = frame_io
        self.log = logger or get_logger(__name__)

    # -- Orchestration -----------------------------------------------------------
    def run(self, config: PipelineConfig) -> Step04Result:
        target = (config.target_width, config.target_height)
        self.log.info(
            "=== Step 4: Real-ESRGAN upscale to %dx%d started ===", *target
        )
        if target[0] <= 0 or target[1] <= 0:
            raise UpscaleError(f"Invalid target size: {target}.")

        sources = self._discover_sources(config)
        tile = config.tile_size if self.tile is None else self.tile
        self.log.info(
            "Upscaling %d frame(s) with %s (tile=%s, %s) ...",
            len(sources),
            self.model_name if self.backend_name == "esrgan" else "lanczos-resize",
            tile if tile > 0 else "off",
            self.backend_name,
        )

        frame_io = self._resolve_frame_io()
        backend = self._resolve_backend(tile, target)
        backend.load()

        out_dir = config.upscaled_8k
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.clean_output_dir:
            self._clean_output_dir(out_dir)
        pattern = UPSCALED_FRAME_PATTERN.replace(".png", f".{self.output_format}")

        from pipeline.esrgan.writer import AsyncFrameWriter

        bar = self._make_progress_bar(len(sources), desc="Upscaling")
        try:
            with AsyncFrameWriter(frame_io, self.writer_queue, self.log) as writer:
                with bar:
                    for idx, src in enumerate(sources, start=1):
                        frame = frame_io.read(src)
                        upscaled = backend.upscale(frame)
                        writer.submit(out_dir / (pattern % idx), upscaled)
                        del frame, upscaled
                        if idx % max(1, self.empty_cache_every) == 0:
                            backend.empty_cache()
                        bar.update(1)
                        bar.set_postfix_str(f"{idx}/{len(sources)}")
            written = writer.count
        finally:
            backend.unload()  # release VRAM for Step 5 even on failure

        validation_ok = self._validate(written, len(sources))
        config.esrgan_model = self.model_name if self.backend_name == "esrgan" else None
        config.esrgan_backend = backend.name
        config.esrgan_tile = tile
        config.upscaled_frame_count = written
        config.upscaled_frame_pattern = pattern
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 5: %s", saved)
        self.log.info(
            "Step 4 completed: %d 8K frames (%dx%d) ready for video assembly.",
            written, *target,
        )
        self.log.info("\n%s", config.summary())
        return Step04Result(
            source_frames=len(sources),
            written_count=written,
            target_size=target,
            tile_used=tile,
            out_dir=out_dir,
            frame_pattern=pattern,
            backend_name=backend.name,
            model_name=config.esrgan_model,
            validation_ok=validation_ok,
        )

    # -- Setup helpers ------------------------------------------------------------------
    def _resolve_frame_io(self) -> Any:
        if self._frame_io is not None:
            return self._frame_io
        from pipeline.frame_io import Cv2FrameIO

        return Cv2FrameIO()

    def _resolve_backend(self, tile: int, target: tuple) -> Any:
        if self._backend is not None:
            return self._backend
        from pipeline.esrgan.backends import create_upscaler

        if self.backend_name == "resize":
            return create_upscaler(
                "resize", logger=self.log, target_size=target
            )
        return create_upscaler(
            "esrgan",
            logger=self.log,
            model=self.model_name,
            weights=self.weights,
            weights_root=self.weights_root,
            device=self.device,
            fp16=self.fp16,
            tile=tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            target_size=target,
        )

    def _discover_sources(self, config: PipelineConfig) -> List[Path]:
        sources = sorted(
            p for p in config.interpolated_720p.glob("frame_*.*") if p.is_file()
        )
        if not sources:
            raise UpscaleError(
                f"No interpolated frames in {config.interpolated_720p}. "
                f"Run Step 3 first."
            )
        if config.interpolated_frame_count and len(sources) != config.interpolated_frame_count:
            self.log.warning(
                "Found %d interpolated frames but Step 3 reported %d -- "
                "proceeding with the %d on disk.",
                len(sources), config.interpolated_frame_count, len(sources),
            )
        self.log.info("Discovered %d interpolated frame(s).", len(sources))
        return sources

    def _clean_output_dir(self, out_dir: Path) -> None:
        stale = [p for p in out_dir.glob("frame_8k_*.*") if p.is_file()]
        for p in stale:
            try:
                p.unlink()
            except OSError:
                pass
        if stale:
            self.log.info("Removed %d stale 8K frame(s).", len(stale))

    # -- Validation & progress ----------------------------------------------------------------------
    def _validate(self, written: int, expected: int) -> bool:
        if written == expected:
            self.log.info("Validation OK: wrote %d/%d 8K frames.", written, expected)
            return True
        self.log.warning(
            "Upscaled count MISMATCH: wrote %d but expected %d. "
            "Step 5 will assemble whatever frames exist.",
            written, expected,
        )
        return False

    def _make_progress_bar(self, total: int, desc: str = "") -> Any:
        try:
            from tqdm import tqdm
        except ImportError:
            self.log.warning("tqdm not installed -- progress bar disabled.")
            return _NullProgress()
        return tqdm(total=total, desc=desc, unit="frame")


class _NullProgress:
    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
