"""STEP 3 — RIFE frame interpolation to the target FPS (1000).

Reads the 720p raw frames from Step 2 (``temp_raw_frames``), runs the official
RIFE v4 network (vendored in :mod:`pipeline.rife.vendor`) on CUDA/fp16, and
writes exactly ``target_fps``-paced frames to ``interpolated_720p`` for Step 4
(8K super-resolution).

## Algorithm (spec-compliant multi-pass + exact resample)

RIFE natively interpolates midpoints (``t=0.5``), so dense upsampling comes in
powers of two. For a required factor ``F = target_fps / source_fps``:

1. ``N = ceil(log2(F))`` recursive subdivision passes per frame pair
   (e.g. 30 -> 1000 FPS needs 33.33x, so ``N=6`` → 64x dense).
2. Each pair ``(F0, F1)`` is subdivided iteratively ``N`` times, batching all
   midpoints of a level into as few GPU forwards as ``batch_size`` allows.
3. The dense ``(P-1)*2^N + 1`` stream is **resampled on the fly** to exactly
   ``round(duration * target_fps)`` frames (nearest-neighbour index mapping),
   so output pacing is precisely 1000 FPS with bounded RAM (streaming, one
   pair resident at a time).

Upstream-inspired shortcuts (both optional, both logged):

- **static pairs** (mean abs diff below ``static_threshold``): intermediates
  are exact copies — no inference needed, saving ``2^N - 1`` forwards.
- **scene cuts** (diff above ``cut_threshold``): intermediates copy ``F0``,
  avoiding ghosting across cuts (mirrors upstream's ``ssim < 0.2`` branch).

Typical usage::

    from pipeline.config import PipelineConfig
    from pipeline.step03_interpolate import Step03Interpolate

    config = PipelineConfig.load("workspace/config.json")
    result = Step03Interpolate(backend="rife", batch_size=4).run(config)
"""

from __future__ import annotations

import bisect
import contextlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

from pipeline.base import PipelineStep
from pipeline.checkpoint import (
    continuous_prefix_length,
    is_oom_error,
    scan_frame_indices,
)
from pipeline.config import (
    CONFIG_FILENAME,
    INTERPOLATED_FRAME_PATTERN,
    PipelineConfig,
)
from pipeline.exceptions import InterpolationError
from pipeline.logger import get_logger
from pipeline.perf import RingBufferWriter
from pipeline.resources import default_weights_root

if TYPE_CHECKING:
    from pipeline.checkpoint import CheckpointManager
    from pipeline.perf import Profiler

PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


@dataclass
class Step03Result:
    source_frames: int          # P: raw frames discovered in temp_raw_frames
    exp: int                    # N: subdivision passes (dense factor 2^N)
    dense_count: int            # D: (P-1)*2^N + 1 (pre-resample stream)
    target_count: int           # T: exact output frames @ target_fps
    written_count: int          # frames actually written to interpolated_720p
    static_skips: int           # pairs skipped as static (copies, no inference)
    cut_skips: int              # pairs treated as scene cuts (copies of F0)
    forwards_run: int           # backend forward calls executed
    out_dir: Path
    frame_pattern: str
    backend_name: str
    validation_ok: bool


@dataclass
class _ResumePlan:
    """Where a crashed run restarts (Step 7 frame-level resume)."""

    start_pair: int      # first source pair to (re)process
    dense_start: int     # dense-stream index that pair starts at
    sel_ptr: int         # first output position to (re)emit
    written: int         # output counter value to continue from
    complete: bool       # all target frames already on disk
    on_disk: int         # continuous output files found (1..N, no gaps)


class Step03Interpolate(PipelineStep[Step03Result]):
    """Step 3 implementation. See module docstring for the full contract."""

    name = "step03_interpolate"

    def __init__(
        self,
        backend: str = "rife",           # "rife" (AI) | "blend" (smoke test)
        rife_version: str = "4",
        weights: Optional[str | Path] = None,
        weights_root: Optional[str | Path] = None,  # None -> default_weights_root()
        device: Optional[str] = None,    # "cuda" | "cpu" | None (auto)
        fp16: Optional[bool] = None,     # None -> auto (True on CUDA)
        batch_size: int = 4,             # pairs per GPU forward batch
        max_exp: int = 6,                # refuse dense factors above 2^6
        output_format: str = "png",      # "png" (lossless) | "jpg"
        clean_output_dir: bool = True,
        empty_cache_every: int = 10,     # torch.cuda.empty_cache() cadence (pairs)
        static_threshold: Optional[float] = 1.0,  # mean-abs-diff; None disables
        cut_threshold: Optional[float] = 60.0,    # mean-abs-diff; None disables
        backend_obj: Optional[Any] = None,  # injected backend (tests / power users)
        frame_io: Optional[Any] = None,     # injected FrameIO (tests)
        logger: Optional[logging.Logger] = None,
        resume: bool = True,             # Step 7: continue from on-disk frames
        checkpoint: Optional["CheckpointManager"] = None,  # Step 7: progress recorder
        checkpoint_every: int = 10,      # record state every N pairs
        oom_retry: bool = True,          # Step 7: halve batch + retry same pair on OOM
        writer_queue: int = 8,           # Step 9: async RAM ring depth (0 = sync)
        profiler: Optional["Profiler"] = None,  # Step 9: stage timing spans
        precision: str = "auto",         # Step 9: auto|fp32|fp16|bf16 (AI backend)
        async_transfers: bool = True,    # Step 9: CUDA stream + non-blocking H2D/D2H
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")
        if output_format not in ("png", "jpg"):
            raise ValueError(f"output_format must be 'png' or 'jpg', got {output_format!r}")
        if writer_queue < 0:
            raise ValueError(f"writer_queue must be >= 0, got {writer_queue}")
        if precision not in PRECISION_CHOICES:
            raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {precision!r}")
        self.backend_name = backend
        self.rife_version = rife_version
        self.weights = weights
        # str(): ckpt_params (JSON) must stay serialisable; frozen builds
        # resolve to <exe-dir>/models, dev runs keep "weights".
        self.weights_root = (
            weights_root if weights_root is not None
            else str(default_weights_root())
        )
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size
        self.max_exp = max_exp
        self.output_format = output_format
        self.clean_output_dir = clean_output_dir
        self.empty_cache_every = empty_cache_every
        self.static_threshold = static_threshold
        self.cut_threshold = cut_threshold
        self._backend = backend_obj
        self._frame_io = frame_io
        self.log = logger or get_logger(__name__)
        self._forwards = 0
        self.resume = resume
        self.checkpoint = checkpoint
        self.checkpoint_every = checkpoint_every
        self.oom_retry = oom_retry
        self.writer_queue = writer_queue
        self.profiler = profiler
        self.precision = precision
        self.async_transfers = async_transfers

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
    def run(self, config: PipelineConfig) -> Step03Result:
        self.log.info("=== Step 3: RIFE interpolation to %.1f FPS started ===", config.target_fps)
        frame_io = self._resolve_frame_io()
        backend = self._resolve_backend()

        sources = self._discover_sources(config)
        factor = self._interpolation_factor(config)
        exp = self.compute_exp(factor, self.max_exp)
        dense = self.dense_count(len(sources), exp)
        target = self.target_count(
            len(sources), config.duration_sec, factor, config.target_fps
        )
        self.log.info(
            "Source: %d frames @ %.2f FPS -> %.1f FPS needs %.2fx: "
            "exp=%d (%dx dense = %d frames), resampled to %d output frames.",
            len(sources), config.original_fps or 0.0, config.target_fps,
            factor, exp, 2**exp, dense, target,
        )

        out_dir = config.interpolated_720p
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = INTERPOLATED_FRAME_PATTERN.replace(".png", f".{self.output_format}")
        selected = self.selection_indices(dense, target)
        ckpt_params = {
            "source_count": len(sources),
            "exp": exp,
            "target_count": target,
            "pattern": pattern,
        }
        # Step 7: resume from on-disk frames BEFORE loading the model, so a
        # fully-complete output dir skips GPU init entirely.
        plan = self._plan_resume(
            out_dir, selected, exp, target, len(sources), ckpt_params
        )
        if plan is not None and plan.complete:
            return self._finish_already_complete(
                config, backend, out_dir, pattern, len(sources), exp, dense, target
            )
        if plan is None:
            if self.clean_output_dir:
                self._clean_output_dir(out_dir)
        else:
            self.log.info(
                "Resuming Step 3 from output frame %d/%d "
                "(source pair %d/%d) — %d frame(s) already on disk, "
                "no re-processing.",
                plan.written + 1, target,
                plan.start_pair + 1, len(sources) - 1, plan.on_disk,
            )

        backend.load()
        try:
            written, static_skips, cut_skips = self._interpolate_all(
                backend, frame_io, sources, exp, out_dir, pattern, selected,
                target, plan, ckpt_params,
            )
        finally:
            backend.unload()  # release VRAM for Step 4 even on failure

        validation_ok = self._validate(written, target)
        config.interpolation_exp = exp
        config.interpolated_frame_count = written
        config.interpolation_backend = backend.name
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 4: %s", saved)
        self.log.info(
            "Step 3 completed: %d frames @ %.1f FPS ready for 8K upscaling "
            "(%d static skips, %d cut skips, %d forwards).",
            written, config.target_fps, static_skips, cut_skips, self._forwards,
        )
        self.log.info("\n%s", config.summary())
        return Step03Result(
            source_frames=len(sources),
            exp=exp,
            dense_count=dense,
            target_count=target,
            written_count=written,
            static_skips=static_skips,
            cut_skips=cut_skips,
            forwards_run=self._forwards,
            out_dir=out_dir,
            frame_pattern=pattern,
            backend_name=backend.name,
            validation_ok=validation_ok,
        )

    # -- Static math (pure, fully unit-tested) -------------------------------------
    @staticmethod
    def compute_exp(factor: float, max_exp: int = 6) -> int:
        """Passes ``N`` with ``2^N >= factor`` (e.g. 33.33x -> 6)."""
        if factor <= 0:
            raise InterpolationError(f"Invalid interpolation factor: {factor}")
        if factor <= 1.0:
            return 0  # already at/above target fps: resample only, no inference
        exp = int(math.ceil(math.log2(factor)))
        if exp > max_exp:
            raise InterpolationError(
                f"Factor {factor:.2f}x needs exp={exp} (>{max_exp}, i.e. dense "
                f"{2**exp}x). Refusing to avoid VRAM/time explosion -- pass a "
                f"larger --max-exp explicitly if you really want it."
            )
        return exp

    @staticmethod
    def dense_count(num_sources: int, exp: int) -> int:
        """Dense stream length: ``(P-1) * 2^N + 1`` (shared endpoints)."""
        return (num_sources - 1) * (2**exp) + 1

    @staticmethod
    def target_count(
        num_sources: int,
        duration_sec: Optional[float],
        factor: float,
        target_fps: float,
    ) -> int:
        """Exact output frames for precise ``target_fps`` pacing."""
        if duration_sec and duration_sec > 0:
            return max(2, int(round(duration_sec * target_fps)))
        return max(2, int(round((num_sources - 1) * factor)) + 1)

    @staticmethod
    def selection_indices(dense: int, target: int) -> List[int]:
        """Nearest-neighbour dense->output index map (monotonic, exact ends)."""
        if dense < 1 or target < 1:
            raise InterpolationError(
                f"Invalid dense/target counts: {dense}/{target}."
            )
        if target == 1:
            return [0]
        return [round(j * (dense - 1) / (target - 1)) for j in range(target)]

    # -- Setup helpers ------------------------------------------------------------------
    def _resolve_frame_io(self) -> Any:
        if self._frame_io is not None:
            return self._frame_io
        from pipeline.rife.io import Cv2FrameIO

        return Cv2FrameIO()

    def _resolve_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        from pipeline.rife.backends import create_backend

        return create_backend(
            self.backend_name,
            logger=self.log,
            **(
                {
                    "version": self.rife_version,
                    "weights": self.weights,
                    "weights_root": self.weights_root,
                    "device": self.device,
                    "fp16": self.fp16,
                    "precision": self.precision,
                    "async_transfers": self.async_transfers,
                }
                if self.backend_name == "rife"
                else {}
            ),
        )

    def _discover_sources(self, config: PipelineConfig) -> List[Path]:
        sources = sorted(
            p for p in config.temp_raw_frames.glob("frame_*.*") if p.is_file()
        )
        if len(sources) < 2:
            raise InterpolationError(
                f"Need at least 2 raw frames in {config.temp_raw_frames} to "
                f"interpolate, found {len(sources)}. Re-run Step 2."
            )
        if config.total_frames and len(sources) != config.total_frames:
            self.log.warning(
                "Found %d raw frames but Step 2 reported %d -- proceeding "
                "with the %d on disk.",
                len(sources), config.total_frames, len(sources),
            )
        self.log.info("Discovered %d raw frames in %s.", len(sources), config.temp_raw_frames)
        return sources

    def _interpolation_factor(self, config: PipelineConfig) -> float:
        if config.interpolation_factor and config.interpolation_factor > 0:
            return config.interpolation_factor
        if config.original_fps and config.original_fps > 0:
            return config.target_fps / config.original_fps
        raise InterpolationError(
            "Cannot determine the interpolation factor: Step 2 metadata "
            "(original_fps) is missing. Re-run Step 2 first."
        )

    def _clean_output_dir(self, out_dir: Path) -> None:
        stale = [p for p in out_dir.glob("frame_*.*") if p.is_file()]
        for p in stale:
            try:
                p.unlink()
            except OSError:
                pass
        if stale:
            self.log.info("Removed %d stale interpolated frame(s).", len(stale))

    # -- Step 7: resume planning ----------------------------------------------------------
    def _plan_resume(
        self,
        out_dir: Path,
        selected: List[int],
        exp: int,
        target: int,
        source_count: int,
        ckpt_params: dict,
    ) -> Optional[_ResumePlan]:
        """Map on-disk output files back to (pair, dense, output) coordinates.

        Returns ``None`` for a fresh start (no usable files, resume disabled,
        or checkpoint params mismatch — the caller then cleans stale files).
        Inference is deterministic, so re-emitting the suspect frontier pair
        overwrites byte-identical content.
        """
        if not self.resume:
            return None
        suffix = f".{self.output_format}"
        indices = scan_frame_indices(out_dir, "frame_", suffix)
        if not indices:
            return None
        on_disk = continuous_prefix_length(indices)
        if on_disk == 0:
            return None  # stray files only (e.g. frame_99999999) -> fresh+clean
        if self.checkpoint is not None:
            diffs = self.checkpoint.check_step_params(self.name, ckpt_params)
            if diffs:
                self.log.warning(
                    "On-disk Step 3 frames are from different settings "
                    "(%s) — discarding them and starting fresh.",
                    "; ".join(diffs),
                )
                return None
        if on_disk >= target:
            return _ResumePlan(0, 0, 0, 0, True, on_disk)
        trusted = max(0, on_disk - 1)  # last prefix file may be torn: redo it
        if trusted == 0:
            return _ResumePlan(0, 0, 0, 0, False, on_disk)
        stride = 2**exp
        frontier_dense = selected[trusted - 1]
        start_pair = min((frontier_dense + 1) // stride, source_count - 2)
        dense_start = start_pair * stride
        # First output position at/after the restart pair; every position
        # before it provably has its file on disk (selected is monotonic).
        sel_ptr = bisect.bisect_left(selected, dense_start)
        return _ResumePlan(start_pair, dense_start, sel_ptr, sel_ptr, False, on_disk)

    def _finish_already_complete(
        self,
        config: PipelineConfig,
        backend: Any,
        out_dir: Path,
        pattern: str,
        source_count: int,
        exp: int,
        dense: int,
        target: int,
    ) -> Step03Result:
        self.log.info(
            "Step 3 outputs already complete: %d/%d frames on disk — "
            "skipping inference entirely (no model loaded).",
            target, target,
        )
        backend_name = getattr(backend, "name", self.backend_name)
        config.interpolation_exp = exp
        config.interpolated_frame_count = target
        config.interpolation_backend = backend_name
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 4: %s", saved)
        self.log.info("\n%s", config.summary())
        return Step03Result(
            source_frames=source_count,
            exp=exp,
            dense_count=dense,
            target_count=target,
            written_count=target,
            static_skips=0,
            cut_skips=0,
            forwards_run=0,
            out_dir=out_dir,
            frame_pattern=pattern,
            backend_name=backend_name,
            validation_ok=True,
        )

    # -- Core loop ------------------------------------------------------------------------------
    def _interpolate_all(
        self,
        backend: Any,
        frame_io: Any,
        sources: List[Path],
        exp: int,
        out_dir: Path,
        pattern: str,
        selected: List[int],
        target: int,
        plan: Optional[_ResumePlan],
        ckpt_params: dict,
    ) -> tuple:
        bar = self._make_progress_bar(
            target, desc="Interpolating",
            initial=plan.sel_ptr if plan else 0,
        )
        written = plan.written if plan else 0
        static_skips = 0
        cut_skips = 0
        sel_ptr = plan.sel_ptr if plan else 0
        dense_idx = plan.dense_start if plan else 0
        num_pairs = len(sources) - 1
        start_pair = plan.start_pair if plan else 0

        # Step 9: PNG encoding moves to a background thread — inference never
        # waits on the disk. Crash safety is unchanged: resume re-scans the
        # disk, so frames still sitting in the RAM ring are simply redone.
        writer = (
            RingBufferWriter(
                frame_io.write, capacity=self.writer_queue, logger=self.log,
                name="step3-writer",
            ).start()
            if self.writer_queue > 0
            else None
        )

        with self._span("step3/io_read"):
            prev_frame = frame_io.read(sources[start_pair])

        def _emit(frame: Any) -> None:
            nonlocal written, sel_ptr
            # `selected` is monotonic; duplicates (target > dense) re-emit.
            while sel_ptr < target and selected[sel_ptr] <= dense_idx:
                if selected[sel_ptr] == dense_idx:
                    written += 1
                    if writer is not None:
                        writer.submit(out_dir / (pattern % written), frame)
                    else:
                        frame_io.write(out_dir / (pattern % written), frame)
                    bar.update(1)
                sel_ptr += 1

        def _record() -> None:
            if self.checkpoint is not None:
                self.checkpoint.record_progress(
                    self.name, written, target, ckpt_params
                )

        def _on_oom() -> None:
            _record()  # persist the frontier BEFORE retrying, crash-proof

        try:
            with self._torch_span("step3/interpolate-all"):
                with bar:
                    for pair_idx in range(start_pair, num_pairs):
                        with self._span("step3/io_read"):
                            cur_frame = frame_io.read(sources[pair_idx + 1])
                        with self._span("step3/shortcut"):
                            shortcut = self._pair_shortcut(prev_frame, cur_frame)
                        if shortcut == "static":
                            static_skips += 1
                            dense_pair: Sequence[Any] = self._copies(
                                prev_frame, 2**exp + 1
                            )
                        elif shortcut == "cut":
                            cut_skips += 1
                            dense_pair = self._copies(prev_frame, 2**exp + 1)
                        else:
                            with self._span("step3/inference"):
                                dense_pair = self._subdivide_with_oom_retry(
                                    backend, prev_frame, cur_frame, exp,
                                    pair_idx, num_pairs, _on_oom,
                                )
                        # All pairs share endpoints: emit all but the last
                        # frame, except the final pair which also emits the
                        # stream end.
                        emit = dense_pair if pair_idx == num_pairs - 1 else dense_pair[:-1]
                        with self._span("step3/io_write"):
                            for frame in emit:
                                _emit(frame)
                                dense_idx += 1
                        del dense_pair
                        prev_frame = cur_frame
                        if (pair_idx + 1) % max(1, self.empty_cache_every) == 0:
                            backend.empty_cache()
                        if (pair_idx + 1) % max(1, self.checkpoint_every) == 0:
                            _record()
                        bar.set_postfix_str(f"pair {pair_idx + 1}/{num_pairs}")
                    _record()
        finally:
            if writer is not None:
                writer.close()  # flush + fail-loud before validation
        return written, static_skips, cut_skips

    def _subdivide_with_oom_retry(
        self,
        backend: Any,
        first: Any,
        last: Any,
        exp: int,
        pair_idx: int,
        num_pairs: int,
        on_oom: Callable[[], None],
    ) -> List[Any]:
        """Subdivide one pair, halving the batch and retrying on VRAM OOM."""
        while True:
            try:
                return self._subdivide_pair(backend, first, last, exp)
            except Exception as exc:  # noqa: BLE001 - OOM detected by inspection
                if (
                    not self.oom_retry
                    or not is_oom_error(exc)
                    or self.batch_size <= 1
                ):
                    raise
                on_oom()
                self.batch_size //= 2
                self.log.warning(
                    "CUDA out of memory at pair %d/%d — halved batch_size "
                    "to %d and retrying the same pair ...",
                    pair_idx + 1, num_pairs, self.batch_size,
                )
                backend.empty_cache()

    def _subdivide_pair(
        self, backend: Any, first: Any, last: Any, exp: int
    ) -> List[Any]:
        """Iterative midpoint subdivision -> ``2^N + 1`` dense frames."""
        level: List[Any] = [first, last]
        for _ in range(exp):
            mids: List[Any] = []
            num_mids = len(level) - 1
            for start in range(0, num_mids, self.batch_size):
                end = min(start + self.batch_size, num_mids)
                chunk0 = level[start:end]
                chunk1 = level[start + 1:end + 1]
                out = backend.interpolate_batch(list(chunk0), list(chunk1), 0.5)
                self._forwards += 1
                mids.extend(list(out))
            merged: List[Any] = [level[0]]
            for i, mid in enumerate(mids):
                merged.append(mid)
                merged.append(level[i + 1])
            level = merged
        return level

    @staticmethod
    def _copies(frame: Any, count: int) -> List[Any]:
        try:
            import numpy as np

            arr = np.asarray(frame)
            return [arr.copy() for _ in range(count)]
        except ImportError:
            return [frame for _ in range(count)]

    def _pair_shortcut(self, first: Any, last: Any) -> Optional[str]:
        """'static' | 'cut' | None via mean-abs-diff (None without numpy)."""
        if self.static_threshold is None and self.cut_threshold is None:
            return None
        try:
            import numpy as np

            diff = float(
                np.mean(
                    np.abs(
                        np.asarray(first, dtype=np.float32)
                        - np.asarray(last, dtype=np.float32)
                    )
                )
            )
        except (ImportError, TypeError, ValueError):
            return None
        if self.static_threshold is not None and diff < self.static_threshold:
            return "static"
        if self.cut_threshold is not None and diff > self.cut_threshold:
            return "cut"
        return None

    # -- Validation & progress ----------------------------------------------------------------------
    def _validate(self, written: int, target: int) -> bool:
        if written == target:
            self.log.info("Validation OK: wrote %d/%d interpolated frames.", written, target)
            return True
        self.log.warning(
            "Interpolated count MISMATCH: wrote %d but target is %d. "
            "Step 4 will upscale whatever frames exist.",
            written, target,
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
