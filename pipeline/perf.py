"""STEP 9 — profiling, async I/O and benchmark reporting (torch-optional).

Everything here works on a bare interpreter; GPU-only features degrade
gracefully when ``torch``/CUDA are absent:

- :class:`RingBufferWriter` — bounded in-RAM FIFO ring + one background
  writer thread (``threading.Thread``). ``cv2.imwrite`` latency for huge 8K
  files overlaps the next frame's inference, so the GPU never waits on the
  disk; the bound doubles as RAM backpressure. Order-preserving, fail-loud.
- :class:`Profiler` — wall-time stage spans (``io_read``/``inference``/
  ``io_write``/...) with CUDA synchronisation for honest GPU timings, plus
  optional :mod:`cProfile` function profiling and :mod:`torch.profiler`
  kernel-level spans (both degrade to no-ops without support).
- :class:`VramSampler` — background thread polling a VRAM probe (torch or
  ``nvidia-smi`` via :func:`pipeline.webui.probe_gpu`) for peak/average
  usage. The probe is injected, so this module never imports the WebUI.
- :class:`BenchmarkReport` — the final human + JSON report: per-step
  ms/frame, processing FPS vs target FPS, stage breakdowns, VRAM peak/avg.
- :func:`detect_accelerators` — capability scan (torch/CUDA/onnxruntime/
  TensorRT) without importing heavy frameworks (unless asked).
"""

from __future__ import annotations

import contextlib
import cProfile
import importlib.util
import io
import json
import logging
import pstats
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from pipeline.logger import get_logger

_SENTINEL: Any = object()


# ---------------------------------------------------------------------------
# Async RAM ring buffer for disk writes (Step 9 §3)
# ---------------------------------------------------------------------------
class RingBufferWriter:
    """Bounded FIFO ring + background thread around any ``write_fn``.

    ``submit()`` blocks only when the ring is full (backpressure: RAM stays
    bounded no matter how slow the SSD is). A single worker preserves order;
    the first worker error is re-raised from :meth:`submit`/:meth:`close`
    (fail-loud — a lost 8K frame must never pass silently).
    """

    def __init__(
        self,
        write_fn: Callable[[Path, Any], None],
        capacity: int = 8,
        logger: Optional[logging.Logger] = None,
        name: str = "ring-writer",
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.write_fn = write_fn
        self.capacity = capacity
        self.log = logger or get_logger(__name__)
        self.name = name
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=capacity)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._count = 0
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def queued(self) -> int:
        """Items currently buffered in RAM (0..capacity)."""
        return self._queue.qsize()

    def start(self) -> "RingBufferWriter":
        if self._started:
            return self
        self._thread = threading.Thread(
            target=self._worker, name=self.name, daemon=True
        )
        self._thread.start()
        self._started = True
        return self

    def submit(self, path: Path, frame: Any) -> None:
        if not self._started:
            raise RuntimeError(f"{self.name}.submit() before start().")
        if self._closed:
            raise RuntimeError(f"{self.name}.submit() after close().")
        self._raise_if_failed()
        self._queue.put((path, frame))

    def close(self) -> int:
        """Flush, stop the worker, re-raise worker errors; return count."""
        if not self._started or self._closed:
            return self.count
        self._closed = True
        self._queue.put(_SENTINEL)
        assert self._thread is not None
        self._thread.join()
        self._raise_if_failed()
        return self.count

    def __enter__(self) -> "RingBufferWriter":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                path, frame = item
                self.write_fn(path, frame)
                with self._lock:
                    self._count += 1
            except BaseException as exc:  # noqa: BLE001 - re-raised on close()
                with self._lock:
                    if self._error is None:
                        self._error = exc
                where = item[0] if item is not _SENTINEL else "?"
                self.log.error("Background writer failed on %s: %s", where, exc)
                return
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Background writer failed: {error}") from error


# ---------------------------------------------------------------------------
# Stage profiler (Step 9 §1)
# ---------------------------------------------------------------------------
_TORCH_MODULE: Any = None  # cached torch (None=unknown, False=absent)
_TORCH_KNOWN = False


def _cuda_sync() -> None:
    """Block until queued CUDA work finishes (no-op without torch/CUDA).

    Mandatory for honest GPU stage timings: CUDA launches are async, so a
    wall timer without synchronisation would only measure launch overhead.
    The torch lookup is cached — an import attempt per span would cost
    minutes on long runs.
    """
    global _TORCH_MODULE, _TORCH_KNOWN
    if not _TORCH_KNOWN:
        try:
            import torch

            _TORCH_MODULE = torch
        except ImportError:
            _TORCH_MODULE = False
        _TORCH_KNOWN = True
    if _TORCH_MODULE is False:
        return
    try:
        if _TORCH_MODULE.cuda.is_available():
            _TORCH_MODULE.cuda.synchronize()
    except Exception:  # noqa: BLE001 - timing must never break a run
        pass


@dataclass
class StageStat:
    calls: int = 0
    total_s: float = 0.0

    @property
    def avg_ms(self) -> float:
        return 1000.0 * self.total_s / self.calls if self.calls else 0.0


class Profiler:
    """Wall-time stage spans + optional cProfile/torch.profiler integration."""

    def __init__(self, enabled: bool = True, torch_profile: bool = False) -> None:
        self.enabled = enabled
        # torch sub-profiling only when explicitly asked AND torch exists.
        self.torch_profile = torch_profile and _has_torch()
        self.stages: Dict[str, StageStat] = {}
        self.torch_tables: List[tuple] = []  # (span name, key_averages table)
        self._cprof: Optional[cProfile.Profile] = None
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one span (CUDA-synchronised when possible)."""
        if not self.enabled:
            yield
            return
        _cuda_sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            elapsed = time.perf_counter() - start
            with self._lock:
                stat = self.stages.setdefault(name, StageStat())
                stat.calls += 1
                stat.total_s += elapsed

    def stats(self) -> Dict[str, Dict[str, float]]:
        total = sum(s.total_s for s in self.stages.values())
        out = {}
        for name, stat in sorted(self.stages.items()):
            out[name] = {
                "calls": stat.calls,
                "total_s": stat.total_s,
                "avg_ms": stat.avg_ms,
                "pct": (100.0 * stat.total_s / total) if total else 0.0,
            }
        return out

    def render_text(self, title: str = "Profile") -> str:
        lines = [f"--- {title} ---"]
        stats = self.stats()
        if not stats:
            return f"{lines[0]}\n(no stages recorded)"
        width = max(len(n) for n in stats)
        lines.append(
            f"{'stage':<{width}}  {'calls':>7}  {'total':>9}  {'avg':>9}  {'share':>6}"
        )
        for name, stat in sorted(stats.items(), key=lambda kv: -kv[1]["total_s"]):
            lines.append(
                f"{name:<{width}}  {stat['calls']:>7}  "
                f"{stat['total_s']:>8.2f}s  {stat['avg_ms']:>8.2f}ms  "
                f"{stat['pct']:>5.1f}%"
            )
        return "\n".join(lines)

    # -- cProfile ---------------------------------------------------------------
    def start_cprofile(self) -> None:
        self._cprof = cProfile.Profile()
        self._cprof.enable()

    def stop_cprofile(self, top_n: int = 30) -> str:
        """Stop and render the top functions by cumulative time."""
        if self._cprof is None:
            return "(cProfile was not started)"
        self._cprof.disable()
        stream = io.StringIO()
        stats = pstats.Stats(self._cprof, stream=stream).sort_stats("cumulative")
        stats.print_stats(top_n)
        self._cprof = None
        return stream.getvalue()

    # -- torch.profiler --------------------------------------------------------------
    @contextlib.contextmanager
    def torch_span(self, name: str) -> Iterator[Any]:
        """Kernel-level span; yields the profiler (or None when unavailable)."""
        if not (self.enabled and self.torch_profile):
            yield None
            return
        try:
            import torch
        except ImportError:
            yield None
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        use_cuda = bool(torch.cuda.is_available())
        if use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=True
        ) as prof:
            yield prof
        table = prof.key_averages().table(
            sort_by="cuda_time_total" if use_cuda else "cpu_time_total",
            row_limit=15,
        )
        self.torch_tables.append((name, table))

    def render_torch(self) -> str:
        if not self.torch_tables:
            return "(no torch.profiler spans recorded)"
        return "\n\n".join(
            f"--- torch.profiler: {name} ---\n{table}"
            for name, table in self.torch_tables
        )


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# VRAM sampler (Step 9 §4 — max/avg VRAM)
# ---------------------------------------------------------------------------
class VramSampler(threading.Thread):
    """Background thread polling ``probe_fn`` for (used_mb, total_mb)."""

    def __init__(
        self,
        probe_fn: Callable[[], Optional[tuple]],
        interval: float = 0.5,
    ) -> None:
        super().__init__(daemon=True, name="vram-sampler")
        self.probe_fn = probe_fn
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: List[tuple] = []  # (monotonic_t, used_mb, total_mb)
        self._lock = threading.Lock()

    def run(self) -> None:  # noqa: D102 - thread body
        while not self.stop_event.is_set():
            try:
                reading = self.probe_fn()
            except Exception:  # noqa: BLE001 - sampling must never break a run
                reading = None
            if reading is not None:
                used, total = reading
                with self._lock:
                    self.samples.append((time.monotonic(), float(used), float(total)))
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)

    def summary(self) -> Optional[Dict[str, float]]:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return None
        used = [s[1] for s in samples]
        return {
            "peak_mb": max(used),
            "avg_mb": sum(used) / len(used),
            "total_mb": samples[-1][2],
            "samples": len(samples),
        }


def nvidia_smi_probe() -> Optional[tuple]:
    """``(used_mb, total_mb)`` via the WebUI's nvidia-smi probe (lazy import)."""
    from pipeline.webui import probe_gpu

    stats = probe_gpu()
    if stats is None or stats.mem_used_mb is None:
        return None
    return (stats.mem_used_mb, stats.mem_total_mb or 0.0)


# ---------------------------------------------------------------------------
# Benchmark report (Step 9 §4)
# ---------------------------------------------------------------------------
@dataclass
class StepBench:
    wall_s: float = 0.0
    frames: int = 0

    @property
    def ms_per_frame(self) -> Optional[float]:
        return 1000.0 * self.wall_s / self.frames if self.frames > 0 else None

    @property
    def processing_fps(self) -> Optional[float]:
        if self.frames <= 0 or self.wall_s <= 0:
            return None
        return self.frames / self.wall_s


@dataclass
class BenchmarkReport:
    generated_at: str = ""
    target_fps: float = 1000.0
    target_size: List[int] = field(default_factory=lambda: [7680, 4320])
    steps: Dict[str, StepBench] = field(default_factory=dict)
    stages: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    vram: Optional[Dict[str, float]] = None
    accelerators: Dict[str, bool] = field(default_factory=dict)

    def render_text(self) -> str:
        lines = [
            "===== Benchmark report =====",
            f"target: {self.target_size[0]}x{self.target_size[1]} @ {self.target_fps:g} FPS",
            "",
            f"{'step':<8}  {'wall':>9}  {'frames':>7}  {'ms/frame':>9}  {'proc FPS':>9}",
        ]
        for key in sorted(self.steps, key=str):
            bench = self.steps[key]
            ms = f"{bench.ms_per_frame:.2f}" if bench.ms_per_frame is not None else "—"
            pf = (
                f"{bench.processing_fps:.2f}"
                if bench.processing_fps is not None else "—"
            )
            lines.append(
                f"{key:<8}  {bench.wall_s:>8.2f}s  {bench.frames:>7}  "
                f"{ms:>9}  {pf:>9}"
            )
        heavy = [
            (key, b) for key, b in self.steps.items()
            if b.processing_fps is not None and b.frames > 0
        ]
        if heavy:
            slowest = min(heavy, key=lambda kv: kv[1].processing_fps or 0)
            lines.append(
                f"\nslowest frame stage: {slowest[0]} "
                f"({slowest[1].processing_fps:.2f} FPS processing vs "
                f"{self.target_fps:g} FPS target)"
            )
        for step_key, stages in sorted(self.stages.items()):
            lines.append(f"\n--- {step_key} stage breakdown ---")
            for name, stat in sorted(stages.items(), key=lambda kv: -kv[1]["total_s"]):
                lines.append(
                    f"  {name:<12} {stat['total_s']:>8.2f}s total / "
                    f"{stat['avg_ms']:>8.2f}ms avg ({stat['pct']:.1f}%)"
                )
        if self.vram:
            lines.append(
                f"\nVRAM: peak {self.vram['peak_mb']:,.0f} MB / "
                f"avg {self.vram['avg_mb']:,.0f} MB "
                f"({self.vram['samples']} samples)"
            )
        else:
            lines.append("\nVRAM: n/a (no GPU samples collected)")
        if self.accelerators:
            caps = ", ".join(
                f"{k}={'yes' if v else 'no'}"
                for k, v in sorted(self.accelerators.items())
            )
            lines.append(f"accelerators: {caps}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, bench in self.steps.items():
            data["steps"][key] = {
                "wall_s": bench.wall_s,
                "frames": bench.frames,
                "ms_per_frame": bench.ms_per_frame,
                "processing_fps": bench.processing_fps,
            }
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = {
            key: StepBench(
                wall_s=value.get("wall_s", 0.0), frames=value.get("frames", 0)
            )
            for key, value in (data.get("steps") or {}).items()
        }
        return cls(
            generated_at=data.get("generated_at", ""),
            target_fps=data.get("target_fps", 1000.0),
            target_size=data.get("target_size", [7680, 4320]),
            steps=steps,
            stages=data.get("stages", {}),
            vram=data.get("vram"),
            accelerators=data.get("accelerators", {}),
        )


# ---------------------------------------------------------------------------
# Mixed precision resolution (Step 9 §2 — fp32/fp16/bf16)
# ---------------------------------------------------------------------------
PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def resolve_torch_precision(
    torch: Any,
    device: str,
    precision_request: Optional[str],
    fp16_request: Optional[bool],
    log: logging.Logger,
) -> str:
    """Resolve the compute dtype name: ``fp32`` | ``fp16`` | ``bf16``.

    Rules: explicit ``precision`` wins; legacy ``fp16`` bool is honoured when
    precision is ``auto``/``None``; ``auto`` means fp16 on CUDA, fp32 on CPU.
    Reduced precision on CPU always falls back to fp32; bf16 additionally
    requires Ampere+ (SM >= 8.0) and falls back to fp16 below that.
    """
    req = (precision_request or "auto").lower()
    if req not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {precision_request!r}")
    if req == "auto":
        if fp16_request is not None:
            req = "fp16" if fp16_request else "fp32"
        else:
            req = "fp16" if device == "cuda" else "fp32"
    if req in ("fp16", "bf16") and device == "cpu":
        log.warning("%s on CPU is unsupported -- falling back to fp32.", req)
        return "fp32"
    if req == "bf16" and device == "cuda":
        try:
            capability = torch.cuda.get_device_capability()
            capable = capability[0] >= 8
        except Exception:  # noqa: BLE001 - unknown GPU: assume incapable
            capable = False
        if not capable:
            log.warning(
                "bf16 needs Ampere+ (SM 8.0+) -- falling back to fp16. "
                "(Pass --precision fp32 to silence, or upgrade the GPU.)"
            )
            return "fp16"
    return req


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    """Map ``fp32``/``fp16``/``bf16`` to the live ``torch.dtype`` object."""
    return {
        "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
    }[dtype_name]


# ---------------------------------------------------------------------------
# Accelerator detection (Step 9 §2 — capability scan)
# ---------------------------------------------------------------------------
@dataclass
class AccelCaps:
    torch: bool = False
    cuda: bool = False          # True only when torch+CUDA verified live
    cuda_probed: bool = False   # False -> "unknown", not "no"
    onnxruntime: bool = False
    tensorrt: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "torch": self.torch,
            "cuda": self.cuda,
            "onnxruntime": self.onnxruntime,
            "tensorrt": self.tensorrt,
        }

    def describe(self) -> str:
        cuda = "yes" if self.cuda else ("unknown" if not self.cuda_probed else "no")
        return (
            f"torch={'yes' if self.torch else 'no'}, cuda={cuda}, "
            f"onnxruntime={'yes' if self.onnxruntime else 'no'}, "
            f"tensorrt={'yes' if self.tensorrt else 'no'}"
        )


def build_benchmark_report(
    *,
    target_fps: float,
    target_size: Sequence[int],
    profiler: Profiler,
    step_wall: Dict[int, float],
    step_frames: Dict[int, int],
    vram_summary: Optional[Dict[str, float]] = None,
    probe_torch: bool = True,
) -> BenchmarkReport:
    """Assemble a :class:`BenchmarkReport` from one finished run.

    ``step_wall`` maps step number -> wall seconds; ``step_frames`` maps
    step number -> frames processed (0 for non-frame steps). Profiler
    stages named ``"stepN/..."`` are grouped under that step.
    """
    stages: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, stat in profiler.stats().items():
        step_key, _, short = name.partition("/")
        stages.setdefault(step_key or "misc", {})[short or name] = stat
    steps = {
        f"step{n}": StepBench(wall_s=wall, frames=step_frames.get(n, 0))
        for n, wall in sorted(step_wall.items())
    }
    return BenchmarkReport(
        target_fps=target_fps,
        target_size=[int(target_size[0]), int(target_size[1])],
        steps=steps,
        stages=stages,
        vram=vram_summary,
        accelerators=detect_accelerators(import_torch=probe_torch).to_dict(),
    )


def detect_accelerators(import_torch: bool = False) -> AccelCaps:
    """Scan for acceleration frameworks (cheap ``find_spec``; torch import
    only when ``import_torch`` is set — importing torch costs seconds)."""
    caps = AccelCaps(
        torch=_has_torch(),
        onnxruntime=importlib.util.find_spec("onnxruntime") is not None,
        tensorrt=importlib.util.find_spec("tensorrt") is not None,
    )
    if import_torch and caps.torch:
        try:
            import torch

            caps.cuda = bool(torch.cuda.is_available())
            caps.cuda_probed = True
        except Exception:  # noqa: BLE001 - broken torch install: report, don't crash
            caps.cuda_probed = True
    return caps
