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

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from pipeline.base import PipelineStep
from pipeline.checkpoint import (
    continuous_prefix_length,
    is_oom_error,
    scan_frame_indices,
)
from pipeline.config import (
    CONFIG_FILENAME,
    UPSCALED_FRAME_PATTERN,
    PipelineConfig,
)
from pipeline.exceptions import UpscaleError
from pipeline.logger import get_logger
from pipeline.resources import default_weights_root

if TYPE_CHECKING:
    from pipeline.checkpoint import CheckpointManager
    from pipeline.perf import Profiler

PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")
ACCEL_CHOICES = ("auto", "none", "onnx")


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
        backend: str = "esrgan",          # "esrgan" (AI) | "onnx" (AI) | "resize" (smoke)
        model: str = "x4plus",            # "x4plus" | "x4plus-anime"
        weights: Optional[str | Path] = None,
        weights_root: Optional[str | Path] = None,  # None -> default_weights_root()
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
        resume: bool = True,             # Step 7: skip on-disk 8K frames
        checkpoint: Optional["CheckpointManager"] = None,  # Step 7: progress recorder
        checkpoint_every: int = 10,      # record state every N frames
        oom_retry: bool = True,          # Step 7: halve tile + retry same frame on OOM
        min_tile: int = 64,              # floor for OOM-driven tile halving
        profiler: Optional["Profiler"] = None,  # Step 9: stage timing spans
        precision: str = "auto",         # Step 9: auto|fp32|fp16|bf16 (AI backend)
        async_transfers: bool = True,    # Step 9: CUDA stream + non-blocking H2D/D2H
        accel: str = "auto",             # Step 9: auto|none|onnx (ONNX Runtime)
    ) -> None:
        if output_format not in ("png", "jpg"):
            raise ValueError(f"output_format must be 'png' or 'jpg', got {output_format!r}")
        if tile is not None and tile < 0:
            raise ValueError(f"tile must be >= 0, got {tile}.")
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        if min_tile < 1:
            raise ValueError(f"min_tile must be >= 1, got {min_tile}.")
        if precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {precision!r}")
        if accel not in ACCEL_CHOICES:
            raise ValueError(f"accel must be one of {ACCEL_CHOICES}, got {accel!r}")
        self.backend_name = backend
        self.model_name = model
        self.weights = weights
        # str(): ckpt_params (JSON) must stay serialisable; frozen builds
        # resolve to <exe-dir>/models, dev runs keep "weights".
        self.weights_root = (
            weights_root if weights_root is not None
            else str(default_weights_root())
        )
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
        self.resume = resume
        self.checkpoint = checkpoint
        self.checkpoint_every = checkpoint_every
        self.oom_retry = oom_retry
        self.min_tile = min_tile
        self.profiler = profiler
        self.precision = precision
        self.async_transfers = async_transfers
        self.accel = accel

    def _span(self, name: str):  # noqa: ANN202 - contextmanager
        """Step 9 stage span (no-op without a profiler attached)."""
        if self.profiler is not None:
            return self.profiler.stage(name)
        return contextlib.nullcontext()

    def _torch_span(self, name: str):  # noqa: ANN202 - contextmanager
        if self.profiler is not None:
            return self.profiler.torch_span(name)
        return contextlib.nullcontext()

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
            self.model_name if self.backend_name in ("esrgan", "onnx") else "lanczos-resize",
            tile if tile > 0 else "off",
            self.backend_name,
        )

        frame_io = self._resolve_frame_io()
        backend = self._resolve_backend(tile, target)

        out_dir = config.upscaled_8k
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = UPSCALED_FRAME_PATTERN.replace(".png", f".{self.output_format}")
        ckpt_params = {
            "source_count": len(sources),
            "target_size": list(target),
            "pattern": pattern,
        }
        # Step 7: resume from on-disk frames BEFORE loading the model, so a
        # fully-complete output dir skips GPU init entirely.
        start_idx, complete, resumed = self._plan_resume(
            out_dir, len(sources), ckpt_params
        )
        backend_name = getattr(backend, "name", self.backend_name)
        if complete:
            return self._finish_already_complete(
                config, backend_name, out_dir, pattern, len(sources), tile, target
            )
        if resumed:
            self.log.info(
                "Resuming Step 4 from frame %d/%d — %d frame(s) already on "
                "disk, no re-processing.",
                start_idx, len(sources), start_idx - 1,
            )
        elif self.clean_output_dir:
            self._clean_output_dir(out_dir)

        backend.load()

        from pipeline.esrgan.writer import AsyncFrameWriter

        bar = self._make_progress_bar(
            len(sources), desc="Upscaling", initial=start_idx - 1
        )
        try:
            with AsyncFrameWriter(frame_io, self.writer_queue, self.log) as writer:
                with self._torch_span("step4/upscale-all"):
                    with bar:
                        for idx in range(start_idx, len(sources) + 1):
                            with self._span("step4/io_read"):
                                frame = frame_io.read(sources[idx - 1])
                            with self._span("step4/inference"):
                                upscaled = self._upscale_with_oom_retry(
                                    backend, frame, idx, ckpt_params, len(sources)
                                )
                            with self._span("step4/io_write"):
                                writer.submit(out_dir / (pattern % idx), upscaled)
                            del frame, upscaled
                            if idx % max(1, self.empty_cache_every) == 0:
                                backend.empty_cache()
                            if idx % max(1, self.checkpoint_every) == 0:
                                self._record(ckpt_params, idx, len(sources))
                            bar.update(1)
                            bar.set_postfix_str(f"{idx}/{len(sources)}")
            written = (start_idx - 1) + writer.count
            self._record(ckpt_params, written, len(sources))
        finally:
            backend.unload()  # release VRAM for Step 5 even on failure

        # OOM retries may have shrunk the tile mid-run: report what's real.
        tile = getattr(backend, "tile", tile)
        validation_ok = self._validate(written, len(sources))
        config.esrgan_model = self.model_name if self.backend_name in ("esrgan", "onnx") else None
        config.esrgan_backend = backend_name
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
            backend_name=backend_name,
            model_name=config.esrgan_model,
            validation_ok=validation_ok,
        )

    # -- Step 7: resume planning --------------------------------------------------------
    def _plan_resume(
        self, out_dir: Path, source_count: int, ckpt_params: dict
    ) -> tuple:
        """Returns ``(start_idx, complete, resumed)`` from on-disk 8K frames.

        Output file N always corresponds to source N, so resume is a pure
        index skip — except the last continuous file, which is reprocessed in
        case a crash tore it mid-write.
        """
        if not self.resume:
            return (1, False, False)
        indices = scan_frame_indices(out_dir, "frame_8k_", f".{self.output_format}")
        if not indices:
            return (1, False, False)
        on_disk = continuous_prefix_length(indices)
        if on_disk == 0:
            return (1, False, False)  # stray files only -> fresh+clean
        if self.checkpoint is not None:
            diffs = self.checkpoint.check_step_params(self.name, ckpt_params)
            if diffs:
                self.log.warning(
                    "On-disk Step 4 frames are from different settings "
                    "(%s) — discarding them and starting fresh.",
                    "; ".join(diffs),
                )
                return (1, False, False)
        if on_disk >= source_count:
            return (1, True, True)
        return (max(1, on_disk), False, True)

    def _finish_already_complete(
        self,
        config: PipelineConfig,
        backend_name: str,
        out_dir: Path,
        pattern: str,
        source_count: int,
        tile: int,
        target: tuple,
    ) -> Step04Result:
        self.log.info(
            "Step 4 outputs already complete: %d/%d frames on disk — "
            "skipping inference entirely (no model loaded).",
            source_count, source_count,
        )
        config.esrgan_model = self.model_name if self.backend_name in ("esrgan", "onnx") else None
        config.esrgan_backend = backend_name
        config.esrgan_tile = tile
        config.upscaled_frame_count = source_count
        config.upscaled_frame_pattern = pattern
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 5: %s", saved)
        self.log.info("\n%s", config.summary())
        return Step04Result(
            source_frames=source_count,
            written_count=source_count,
            target_size=target,
            tile_used=tile,
            out_dir=out_dir,
            frame_pattern=pattern,
            backend_name=backend_name,
            model_name=config.esrgan_model,
            validation_ok=True,
        )

    def _record(self, params: dict, frame_idx: int, total: int) -> None:
        if self.checkpoint is not None:
            self.checkpoint.record_progress(self.name, frame_idx, total, params)

    def _upscale_with_oom_retry(
        self, backend: Any, frame: Any, idx: int, params: dict, total: int
    ) -> Any:
        """Upscale one frame, halving the tile and retrying on VRAM OOM."""
        while True:
            try:
                return backend.upscale(frame)
            except Exception as exc:  # noqa: BLE001 - OOM detected by inspection
                if not self.oom_retry or not is_oom_error(exc):
                    raise
                current_tile = getattr(backend, "tile", None)
                if current_tile is None or current_tile <= self.min_tile:
                    raise
                new_tile = max(self.min_tile, current_tile // 2)
                # Persist the frontier BEFORE retrying; the failed frame is
                # NOT marked done, so the retry (or a later resume after a
                # real crash) restarts exactly from this frame.
                self._record(params, idx - 1, total)
                self.log.warning(
                    "CUDA out of memory at frame %d — halved tile size "
                    "%d → %d and retrying the same frame ...",
                    idx, current_tile, new_tile,
                )
                backend.tile = new_tile
                backend.empty_cache()

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
        weights_look_onnx = (
            self.weights is not None
            and str(self.weights).lower().endswith(".onnx")
        )
        if self.accel == "onnx" and not weights_look_onnx and self.backend_name != "onnx":
            raise UpscaleError(
                "--accel onnx needs ONNX weights: pass --esrgan-weights "
                "model.onnx (export one with: python -m "
                "pipeline.esrgan.onnx_backend --weights model.pth "
                "--model x4plus --out model.onnx)."
            )
        if self.accel == "none" and weights_look_onnx:
            raise UpscaleError(
                "--accel none forces the PyTorch backend, which cannot read "
                ".onnx weights. Use --accel auto (or backend 'onnx')."
            )
        if self.backend_name == "onnx":
            if self.weights is None:
                raise UpscaleError(
                    "Backend 'onnx' needs explicit weights: pass "
                    "--esrgan-weights model.onnx."
                )
            return create_upscaler(
                "onnx",
                logger=self.log,
                model=self.model_name,
                weights=self.weights,
                tile=tile,
                tile_pad=self.tile_pad,
                target_size=target,
            )
        return create_upscaler(
            "esrgan",
            logger=self.log,
            model=self.model_name,
            weights=self.weights,
            weights_root=self.weights_root,
            device=self.device,
            fp16=self.fp16,
            precision=self.precision,
            async_transfers=self.async_transfers,
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

    def _make_progress_bar(self, total: int, desc: str = "", initial: int = 0) -> Any:
        try:
            from tqdm import tqdm
        except ImportError:
            self.log.warning("tqdm not installed -- progress bar disabled.")
            return _NullProgress()
        return tqdm(total=total, desc=desc, unit="frame", initial=initial)


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
