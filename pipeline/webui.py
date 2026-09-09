"""STEP 8 — WebUI/CLI backend: options, live progress bus, background runner.

This module holds everything the Gradio WebUI (``app.py``) and the new CLI
shortcuts (``--resolution``/``--model``/``--ui``) share — with **zero hard
dependency on Gradio**, so it stays unit-testable on a bare machine:

- :func:`parse_resolution` / :func:`parse_tile_size` — friendly CLI values
  (``8K``, ``4K``, ``auto``) parsed once, used by both frontends.
- :class:`RunOptions` — one validated option set driving a full run, plus
  :meth:`RunOptions.to_cli_args` so the UI can show the equivalent terminal
  command for reproducibility.
- :class:`ProgressBus` + :class:`QueueLogHandler` — thread-safe live feed of
  log lines, step banners and frame counts from the worker thread to the UI.
- :class:`FrameWatcher` — polls output folders and publishes frame counts,
  so progress/ETA work **without touching Steps 1-6** (their ``tqdm`` bars
  stay on the server console, the browser gets file counts + rates).
- :func:`probe_gpu` — ``nvidia-smi`` temperature/VRAM snapshot (``None``
  when no NVIDIA GPU exists — never fatal).
- :func:`run_pipeline` — the whole 1→5 (optionally 6) chain in *this*
  thread, honouring a ``threading.Event`` stop flag between steps; the UI
  runs it in a background thread so the interface never freezes. Stopping
  keeps the Step 7 checkpoint, so a stopped run resumes later.
"""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline.checkpoint import (
    STEP_KEYS,
    CheckpointManager,
    scan_frame_indices,
    steps_to_run,
)
from pipeline.config import PipelineConfig
from pipeline.exceptions import WebUIError
from pipeline.logger import get_logger

# ---------------------------------------------------------------------------
# Friendly value parsing (shared by CLI flags and WebUI widgets)
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS: Dict[str, Tuple[int, int]] = {
    "8k": (7680, 4320),
    "4k": (3840, 2160),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
}
RESOLUTION_LABELS: Dict[Tuple[int, int], str] = {
    size: label for label, size in RESOLUTION_PRESETS.items()
}
_CUSTOM_SIZE_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)")
_MAX_DIMENSION = 16384


def parse_resolution(value: str) -> Tuple[int, int]:
    """Parse ``8K``/``4K``/``1080p``/``720p``/``WIDTHxHEIGHT`` -> ``(w, h)``."""
    text = str(value).strip().lower()
    if text in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[text]
    match = _CUSTOM_SIZE_RE.fullmatch(text)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        if 0 < width <= _MAX_DIMENSION and 0 < height <= _MAX_DIMENSION:
            return (width, height)
    raise ValueError(
        f"Invalid resolution {value!r}: use 8K, 4K, 1080p, 720p or "
        f"WIDTHxHEIGHT (e.g. 7680x4320)."
    )


def parse_tile_size(value: Any) -> Optional[int]:
    """Parse ``auto``/``None`` -> ``None`` (Step 1 decides) else positive int."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "auto":
        return None
    try:
        size = int(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid tile size {value!r}: use 'auto' or a positive integer."
        ) from None
    if size <= 0:
        raise ValueError(
            f"Invalid tile size {value!r}: use 'auto' or a positive integer."
        )
    return size


# ---------------------------------------------------------------------------
# Run options
# ---------------------------------------------------------------------------
RESUME_CHOICES = ("auto", "resume", "fresh")
RESUME_TO_POLICY = {"auto": "ask", "resume": "yes", "fresh": "no"}
RESUME_TO_CLI = {"auto": "ask", "resume": "yes", "fresh": "no"}
INTERP_BACKENDS = ("rife", "blend")
UPSCALE_BACKENDS = ("esrgan", "resize")


@dataclass
class RunOptions:
    """One validated option set for a full pipeline run (UI or programmatic)."""

    input: str
    output: Optional[str] = None       # None -> derived next to the input
    workspace: str = "workspace"
    target_fps: float = 1000.0
    resolution: Tuple[int, int] = (7680, 4320)
    tile_size: Optional[int] = None     # None = auto (VRAM-derived)
    model: str = "x4plus"               # Real-ESRGAN weights (Real-CUGAN planned)
    interp_backend: str = "rife"        # "rife" (AI) | "blend" (smoke)
    upscale_backend: str = "esrgan"     # "esrgan" (AI) | "resize" (smoke)
    video_codec: str = "auto"
    crf: float = 19.0
    resume: str = "auto"                # auto | resume | fresh
    cleanup: bool = False               # also run Step 6 at the end
    checkpoint_every: int = 10
    auto_install: bool = False          # UI default: never pip-install mid-run

    def resolution_label(self) -> str:
        w, h = self.resolution
        return RESOLUTION_LABELS.get((w, h), f"{w}x{h}")

    def output_path(self) -> Path:
        if self.output:
            return Path(self.output)
        src = Path(self.input)
        fps = f"{self.target_fps:g}"
        return src.parent / f"{src.stem}_{self.resolution_label()}_{fps}fps.mp4"

    def validate(self) -> List[str]:
        """All problems at once (UI shows them together); [] means OK."""
        from pipeline.esrgan.weights import ESRGAN_MODELS
        from pipeline.step05_assemble import Step05Assemble

        errors = []
        if not self.input:
            errors.append("Giriş videosu seçilməyib (--input tələb olunur).")
        elif not Path(self.input).is_file():
            errors.append(f"Giriş videosu tapılmadı: {self.input}")
        if not isinstance(self.target_fps, (int, float)) or not self.target_fps > 0:
            errors.append(f"Hədəf FPS müsbət olmalıdır: {self.target_fps!r}.")
        try:
            w, h = self.resolution
            if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
                raise TypeError
        except (TypeError, ValueError):
            errors.append(f"Rezolusiya (en, hündürlük) olmalıdır: {self.resolution!r}.")
        if self.tile_size is not None and (
            not isinstance(self.tile_size, int) or self.tile_size <= 0
        ):
            errors.append(f"Tile ölçüsü 'auto' və ya müsbət tam olmalıdır: {self.tile_size!r}.")
        if self.model not in ESRGAN_MODELS:
            errors.append(
                f"Naməlum model: {self.model!r} (mövcud: {sorted(ESRGAN_MODELS)}; "
                f"Real-CUGAN planlaşdırılır)."
            )
        if self.interp_backend not in INTERP_BACKENDS:
            errors.append(f"Naməlum interpolasiya backend-i: {self.interp_backend!r}.")
        if self.upscale_backend not in UPSCALE_BACKENDS:
            errors.append(f"Naməlum upscale backend-i: {self.upscale_backend!r}.")
        if self.video_codec not in Step05Assemble.CODECS:
            errors.append(f"Naməlum kodek: {self.video_codec!r}.")
        if not 0 <= self.crf <= 63:
            errors.append(f"CRF 0..63 olmalıdır: {self.crf!r}.")
        if self.resume not in RESUME_CHOICES:
            errors.append(f"Resume rejimi auto/resume/fresh olmalıdır: {self.resume!r}.")
        if self.checkpoint_every < 1:
            errors.append("checkpoint_every ən azı 1 olmalıdır.")
        return errors

    def to_cli_args(self) -> List[str]:
        """Equivalent ``main.py`` command (reproducibility + power users)."""
        label = self.resolution_label()
        res_flag = {
            "8k": "8K", "4k": "4K", "1080p": "1080p", "720p": "720p",
        }.get(label, label)
        args = [
            "--input", self.input,
            "--output", str(self.output_path()),
            "--workspace", self.workspace,
            "--target-fps", f"{self.target_fps:g}",
            "--resolution", res_flag,
            "--tile-size", str(self.tile_size) if self.tile_size else "auto",
            "--model", self.model,
            "--backend", self.interp_backend,
            "--upscale-backend", self.upscale_backend,
            "--video-codec", self.video_codec,
            "--resume", RESUME_TO_CLI[self.resume],
            "--to-step", "6" if self.cleanup else "5",
            "--checkpoint-every", str(self.checkpoint_every),
        ]
        if self.crf != 19.0:
            args += ["--crf", f"{self.crf:g}"]
        if not self.auto_install:
            args.append("--no-auto-install")
        return args


# ---------------------------------------------------------------------------
# Live progress bus (worker thread -> UI thread)
# ---------------------------------------------------------------------------
@dataclass
class LogEvent:
    level: str
    message: str


@dataclass
class StepEvent:
    step: int
    name: str


@dataclass
class FrameProgress:
    stage: str  # "interpolate" | "upscale"
    done: int
    total: Optional[int] = None


@dataclass
class StatusEvent:
    status: str  # started | step-done | cancelled | done | error
    detail: str = ""


Event = Union[LogEvent, StepEvent, FrameProgress, StatusEvent]


class ProgressBus:
    """Thread-safe event queue decoupling the runner from any frontend."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Event]" = queue.Queue()

    def publish(self, event: Event) -> None:
        self._queue.put(event)

    def log(self, level: str, message: str) -> None:
        self.publish(LogEvent(level, message))

    def poll(self) -> List[Event]:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                return events

    def handler(self) -> "QueueLogHandler":
        return QueueLogHandler(self)


class QueueLogHandler(logging.Handler):
    """Mirror ``logging`` records into the bus; detect ``=== Step N`` banners."""

    _STEP_RE = re.compile(r"===\s*Step\s+(\d+)")

    def __init__(self, bus: ProgressBus) -> None:
        super().__init__()
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - logging must never crash a run
            return
        self.bus.publish(LogEvent(record.levelname, message))
        match = self._STEP_RE.search(message)
        if match:
            step = int(match.group(1))
            self.bus.publish(StepEvent(step, STEP_KEYS.get(step, f"step{step:02d}")))


# ---------------------------------------------------------------------------
# Formatting / rate / GPU helpers (pure, UI-agnostic)
# ---------------------------------------------------------------------------
def format_eta(seconds: Optional[float]) -> str:
    """``90`` -> ``1m 30s``; ``None``/negative -> ``—``."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class RateEstimator:
    """Exponentially-smoothed frames/sec + ETA from progress samples."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._rate: Optional[float] = None
        self._last_done: Optional[int] = None
        self._last_t: Optional[float] = None

    def update(self, done: int, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        if self._last_t is not None and now > self._last_t:
            delta = done - (self._last_done or 0)
            if delta >= 0:
                instant = delta / (now - self._last_t)
                self._rate = (
                    instant if self._rate is None
                    else self.alpha * instant + (1 - self.alpha) * self._rate
                )
        self._last_done = done
        self._last_t = now

    @property
    def rate(self) -> Optional[float]:
        return self._rate

    def eta(self, done: int, total: Optional[int]) -> Optional[float]:
        if total is None or total <= 0 or done >= total:
            return None
        if not self._rate or self._rate <= 0:
            return None
        return (total - done) / self._rate


@dataclass
class GpuStats:
    temp_c: Optional[float] = None
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None

    def describe(self) -> str:
        if self.mem_total_mb:
            pct = 100 * (self.mem_used_mb or 0) / self.mem_total_mb
            mem = (
                f"VRAM {self.mem_used_mb:,.0f} / {self.mem_total_mb:,.0f} MB "
                f"({pct:.0f}%)"
            )
        else:
            mem = "VRAM —"
        temp = f"{self.temp_c:.0f}°C" if self.temp_c is not None else "—"
        return f"🌡 {temp} · {mem}"


def probe_gpu(timeout: float = 5) -> Optional[GpuStats]:
    """One ``nvidia-smi`` snapshot; ``None`` when unavailable (never raises)."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired,
            subprocess.CalledProcessError, OSError):
        return None
    try:
        temp, used, total = [
            float(part) for part in completed.stdout.strip().split("\n")[0].split(",")
        ]
    except (ValueError, IndexError):
        return None
    return GpuStats(temp_c=temp, mem_used_mb=used, mem_total_mb=total)


class FrameWatcher(threading.Thread):
    """Daemon thread polling an output folder and publishing frame counts."""

    def __init__(
        self,
        directory: str | Path,
        prefix: str,
        suffix: str,
        stage: str,
        bus: ProgressBus,
        interval: float = 1.0,
        stop: Optional[threading.Event] = None,
        total: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.directory = Path(directory)
        self.prefix = prefix
        self.suffix = suffix
        self.stage = stage
        self.bus = bus
        self.interval = interval
        self.stop_event = stop or threading.Event()
        self._total = total

    def set_total(self, total: Optional[int]) -> None:
        self._total = total

    def run(self) -> None:  # noqa: D102 - thread body
        while not self.stop_event.is_set():
            try:
                done = len(scan_frame_indices(
                    self.directory, self.prefix, self.suffix
                ))
            except OSError:
                done = 0
            self.bus.publish(FrameProgress(self.stage, done, self._total))
            self.stop_event.wait(self.interval)


def expected_interpolated_total(config: PipelineConfig) -> Optional[int]:
    """Frames Step 3 will write (same math as the step itself)."""
    from pipeline.step03_interpolate import Step03Interpolate

    factor = config.interpolation_factor
    if not factor and config.original_fps:
        factor = config.target_fps / config.original_fps
    if not factor:
        return None
    sources = config.total_frames or 0
    if sources <= 0 and not config.duration_sec:
        return None
    return Step03Interpolate.target_count(
        sources, config.duration_sec, factor, config.target_fps
    )


# ---------------------------------------------------------------------------
# Background runner (this thread; the UI wraps it in a worker thread)
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    status: str  # "done" | "cancelled"
    output: Optional[str]
    message: str
    steps_completed: List[int] = field(default_factory=list)


def run_pipeline(
    opts: RunOptions,
    bus: Optional[ProgressBus] = None,
    stop: Optional[threading.Event] = None,
) -> RunResult:
    """Run steps 1→5 (optionally 6); stop flag is honoured between steps."""
    from pipeline.step01_environment import setup_environment
    from pipeline.step02_frames import Step02Frames
    from pipeline.step03_interpolate import Step03Interpolate
    from pipeline.step04_upscale import Step04Upscale
    from pipeline.step05_assemble import Step05Assemble
    from pipeline.step06_cleanup import Step06Cleanup

    bus = bus or ProgressBus()
    stop = stop or threading.Event()
    errors = opts.validate()
    if errors:
        raise WebUIError(" | ".join(errors))
    if stop.is_set():
        return RunResult("cancelled", None, "Başlamadan dayandırıldı.", [])

    log = get_logger("webui-run")
    handler = bus.handler()
    log.addHandler(handler)
    watchers: List[FrameWatcher] = []
    try:
        output = str(opts.output_path())
        bus.publish(StatusEvent("started", f"Giriş: {opts.input}"))
        config = setup_environment(
            input_video_path=opts.input,
            final_output_path=output,
            workspace_root=opts.workspace,
            target_fps=opts.target_fps,
            target_width=opts.resolution[0],
            target_height=opts.resolution[1],
            auto_install=opts.auto_install,
            tile_size_override=opts.tile_size,
            save_config=True,
            logger=log,
        )

        manager = CheckpointManager(config.workspace_root, logger=log)
        snapshot = manager.snapshot_from_config(config)
        decision = manager.decide(
            snapshot, policy=RESUME_TO_POLICY[opts.resume], interactive=False
        )
        bus.log("INFO", f"Checkpoint: {decision.reason}.")
        if decision.action == "overwrite":
            manager.discard_progress(from_step=1)
            manager.begin_run(snapshot, 1)
            skip = 0
        elif decision.action == "resume":
            manager.attach(decision.state)
            skip = decision.skip_through_step
        else:
            manager.begin_run(snapshot, 1)
            skip = 0
        manager.mark_step_complete(1)
        manager.refresh_snapshot(config)

        to_step = 6 if opts.cleanup else 5
        planned = steps_to_run(2, to_step, skip)
        done_steps: List[int] = [1] if skip >= 1 else [1]
        if skip:
            bus.log("INFO", f"Bitmiş addımlar keçilir: 1..{skip}.")

        for step_num in planned:
            if stop.is_set():
                return RunResult(
                    "cancelled", None,
                    f"Addım {step_num}-dən əvvəl dayandırıldı — checkpoint "
                    f"yadda saxlanıldı, təkrar başlatqda davam edəcək.",
                    done_steps,
                )
            if step_num == 2:
                Step02Frames(
                    image_format="png", audio_format="wav",
                    clean_frame_dir=True, logger=log,
                ).run(config)
                total = expected_interpolated_total(config)
                bus.publish(FrameProgress("interpolate", 0, total))
            elif step_num == 3:
                # NOTE: the watcher gets its OWN stop event — sharing the
                # run's `stop` flag would cancel the whole run when the
                # watcher shuts down after the step.
                watch_stop = threading.Event()
                watcher = FrameWatcher(
                    config.interpolated_720p, "frame_", ".png",
                    "interpolate", bus, stop=watch_stop,
                    total=expected_interpolated_total(config),
                )
                watchers.append(watcher)
                watcher.start()
                try:
                    Step03Interpolate(
                        backend=opts.interp_backend,
                        checkpoint=manager,
                        checkpoint_every=opts.checkpoint_every,
                        logger=log,
                    ).run(config)
                finally:
                    watch_stop.set()
                    watcher.join(timeout=5)
                bus.publish(FrameProgress(
                    "upscale", 0, config.interpolated_frame_count
                ))
            elif step_num == 4:
                watch_stop = threading.Event()
                watcher = FrameWatcher(
                    config.upscaled_8k, "frame_8k_", ".png",
                    "upscale", bus, stop=watch_stop,
                    total=config.interpolated_frame_count,
                )
                watchers.append(watcher)
                watcher.start()
                try:
                    Step04Upscale(
                        backend=opts.upscale_backend,
                        model=opts.model,
                        checkpoint=manager,
                        checkpoint_every=opts.checkpoint_every,
                        logger=log,
                    ).run(config)
                finally:
                    watch_stop.set()
                    watcher.join(timeout=5)
            elif step_num == 5:
                Step05Assemble(
                    video_codec=opts.video_codec, crf=opts.crf, logger=log
                ).run(config)
            elif step_num == 6:
                Step06Cleanup(logger=log).run(config)
            manager.mark_step_complete(step_num)
            manager.refresh_snapshot(config)
            done_steps.append(step_num)
            bus.publish(StatusEvent("step-done", f"Addım {step_num} bitdi."))

        final = output if Path(output).is_file() else None
        return RunResult(
            "done", final,
            f"Tamamlandı: {final}" if final else "Tamamlandı.",
            done_steps,
        )
    finally:
        for watcher in watchers:
            watcher.stop_event.set()
        log.removeHandler(handler)
