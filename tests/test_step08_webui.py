"""Unit tests for Step 8 — CLI shortcuts + WebUI backend (no Gradio needed)."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pipeline.exceptions import WebUIError
from pipeline.webui import (
    FrameProgress,
    FrameWatcher,
    LogEvent,
    ProgressBus,
    QueueLogHandler,
    RateEstimator,
    RunOptions,
    RunResult,
    StatusEvent,
    StepEvent,
    expected_interpolated_total,
    format_eta,
    parse_resolution,
    parse_tile_size,
    probe_gpu,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("8K", (7680, 4320)), ("8k", (7680, 4320)), (" 4K ", (3840, 2160)),
        ("1080p", (1920, 1080)), ("720p", (1280, 720)),
        ("7680x4320", (7680, 4320)), ("640X480", (640, 480)),
    ],
)
def test_parse_resolution_ok(value, expected):
    assert parse_resolution(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "8Kx", "0x720", "-1x720", "99999x99999x1"])
def test_parse_resolution_bad(value):
    with pytest.raises(ValueError, match="Invalid resolution"):
        parse_resolution(value)


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("auto", None), ("AUTO", None), ("256", 256), (512, 512)],
)
def test_parse_tile_size_ok(value, expected):
    assert parse_tile_size(value) == expected


@pytest.mark.parametrize("value", ["0", "-4", "big", ""])
def test_parse_tile_size_bad(value):
    with pytest.raises(ValueError, match="Invalid tile size"):
        parse_tile_size(value)


# ---------------------------------------------------------------------------
# RunOptions
# ---------------------------------------------------------------------------
def test_run_options_output_derivation_and_cli(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"x")
    opts = RunOptions(input=str(video))
    assert opts.output_path() == tmp_path / "movie_8k_1000fps.mp4"
    assert opts.validate() == []
    args = opts.to_cli_args()
    assert args[:4] == ["--input", str(video), "--output", str(tmp_path / "movie_8k_1000fps.mp4")]
    assert "--resolution" in args and "8K" in args
    assert "--to-step" in args and "5" in args
    assert "--no-auto-install" in args  # UI default: never pip mid-run
    assert opts.resolution_label() == "8k"

    custom = RunOptions(
        input=str(video), output=str(tmp_path / "o.mp4"), resolution=(640, 480),
        target_fps=60, cleanup=True, crf=23, auto_install=True,
    )
    assert custom.output_path() == tmp_path / "o.mp4"
    assert custom.resolution_label() == "640x480"
    cli = custom.to_cli_args()
    assert "640x480" in cli and "6" in cli and "--crf" in cli
    assert "--no-auto-install" not in cli


def test_run_options_validate_reports_all_problems(tmp_path: Path):
    opts = RunOptions(
        input=str(tmp_path / "missing.mp4"), target_fps=0, model="cugan",
        interp_backend="rife2", upscale_backend="x", video_codec="prores",
        resume="sometimes", checkpoint_every=0, crf=99,
    )
    errors = opts.validate()
    assert len(errors) >= 8
    assert any("tapılmadı" in e for e in errors)
    assert any("Real-CUGAN" in e for e in errors)
    assert RunOptions(input="").validate()  # empty input flagged, no crash


# ---------------------------------------------------------------------------
# Progress bus + log handler
# ---------------------------------------------------------------------------
def test_bus_poll_order_and_drain():
    bus = ProgressBus()
    assert bus.poll() == []
    bus.publish(LogEvent("INFO", "a"))
    bus.publish(StepEvent(3, "step03_interpolate"))
    bus.log("WARNING", "b")
    events = bus.poll()
    assert [type(e).__name__ for e in events] == ["LogEvent", "StepEvent", "LogEvent"]
    assert bus.poll() == []


def test_queue_log_handler_mirrors_and_detects_steps():
    bus = ProgressBus()
    handler = bus.handler()
    logger = logging.getLogger("test-step08-bus")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("=== Step 3: RIFE interpolation started ===")
        logger.warning("plain line")
    finally:
        logger.removeHandler(handler)
    events = bus.poll()
    assert isinstance(events[0], LogEvent) and "Step 3" in events[0].message
    assert isinstance(events[1], StepEvent) and events[1].step == 3
    assert events[1].name == "step03_interpolate"
    assert isinstance(events[2], LogEvent) and events[2].level == "WARNING"


def test_bus_thread_safe():
    bus = ProgressBus()
    threads = [
        threading.Thread(target=lambda: [bus.log("INFO", str(i)) for i in range(50)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(bus.poll()) == 200


# ---------------------------------------------------------------------------
# Formatting / rate / GPU
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seconds,expected",
    [(None, "—"), (-5, "—"), (0, "0s"), (7, "7s"), (90, "1m 30s"),
     (3599, "59m 59s"), (3600, "1h 00m"), (7384, "2h 03m")],
)
def test_format_eta(seconds, expected):
    assert format_eta(seconds) == expected


def test_rate_estimator():
    est = RateEstimator(alpha=1.0)  # no smoothing: exact rates
    assert est.rate is None and est.eta(0, 100) is None
    est.update(0, now=100.0)
    est.update(10, now=101.0)
    assert est.rate == pytest.approx(10.0)
    assert est.eta(10, 100) == pytest.approx(9.0)
    assert est.eta(100, 100) is None  # done
    assert est.eta(0, None) is None  # unknown total
    est.reset()
    assert est.rate is None


def test_probe_gpu_parses_and_tolerates(monkeypatch):
    class _Done:
        stdout = "45, 1200, 8192\n"

    monkeypatch.setattr(
        "pipeline.webui.subprocess.run", lambda *a, **k: _Done()
    )
    stats = probe_gpu()
    assert (stats.temp_c, stats.mem_used_mb, stats.mem_total_mb) == (45.0, 1200.0, 8192.0)
    assert "45°C" in stats.describe() and "VRAM" in stats.describe()

    def _boom(*a, **k):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr("pipeline.webui.subprocess.run", _boom)
    assert probe_gpu() is None

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr("pipeline.webui.subprocess.run", _timeout)
    assert probe_gpu() is None

    class _Garbage:
        stdout = "not-a-gpu-line\n"

    monkeypatch.setattr(
        "pipeline.webui.subprocess.run", lambda *a, **k: _Garbage()
    )
    assert probe_gpu() is None


# ---------------------------------------------------------------------------
# Frame watcher
# ---------------------------------------------------------------------------
def test_frame_watcher_publishes_counts(tmp_path: Path):
    bus = ProgressBus()
    stop = threading.Event()
    watcher = FrameWatcher(
        tmp_path, "frame_", ".png", "interpolate", bus,
        interval=0.05, stop=stop, total=3,
    )
    watcher.start()
    try:
        (tmp_path / "frame_00000001.png").write_bytes(b"a")
        (tmp_path / "frame_00000002.png").write_bytes(b"b")
        deadline = time.monotonic() + 5
        seen = 0
        while time.monotonic() < deadline and seen < 2:
            for event in bus.poll():
                if isinstance(event, FrameProgress):
                    seen = max(seen, event.done)
            time.sleep(0.05)
        assert seen == 2
    finally:
        stop.set()
        watcher.join(timeout=5)
        assert not watcher.is_alive()


# ---------------------------------------------------------------------------
# Totals helper
# ---------------------------------------------------------------------------
def test_expected_interpolated_total(tmp_path: Path):
    from pipeline.config import PipelineConfig

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    config = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "o.mp4",
        original_fps=30.0, total_frames=4, duration_sec=0.1,
        interpolation_factor=1000.0 / 30.0,
    )
    assert expected_interpolated_total(config) == 100
    config.interpolation_factor = None  # derived from original_fps
    assert expected_interpolated_total(config) == 100
    config.original_fps = None
    assert expected_interpolated_total(config) is None


# ---------------------------------------------------------------------------
# Runner guards (no heavy pipeline execution in unit tests)
# ---------------------------------------------------------------------------
def test_run_pipeline_rejects_bad_options_without_setup(tmp_path: Path, monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("setup must not run on invalid options")

    monkeypatch.setattr(
        "pipeline.step01_environment.setup_environment", _forbidden
    )
    with pytest.raises(WebUIError, match="tapılmadı"):
        run_pipeline(RunOptions(input=str(tmp_path / "missing.mp4")))


def test_run_pipeline_prestopped_returns_cancelled(tmp_path: Path, monkeypatch):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")

    def _forbidden(*a, **k):
        raise AssertionError("setup must not run when pre-stopped")

    monkeypatch.setattr(
        "pipeline.step01_environment.setup_environment", _forbidden
    )
    stop = threading.Event()
    stop.set()
    result = run_pipeline(RunOptions(input=str(video)), stop=stop)
    assert isinstance(result, RunResult)
    assert result.status == "cancelled" and result.output is None


# ---------------------------------------------------------------------------
# Runner with fake steps (fast end-to-end of the orchestration itself)
# ---------------------------------------------------------------------------
def test_run_pipeline_completes_with_fake_steps(tmp_path: Path, monkeypatch):
    import pipeline.step01_environment as step01
    import pipeline.step02_frames as step02
    import pipeline.step03_interpolate as step03
    import pipeline.step04_upscale as step04
    import pipeline.step05_assemble as step05
    from pipeline.config import PipelineConfig

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    made_config = {}

    def _fake_setup(**kwargs):
        config = PipelineConfig(
            input_video_path=kwargs["input_video_path"],
            final_output_path=kwargs["final_output_path"],
            workspace_root=kwargs["workspace_root"],
            target_fps=kwargs["target_fps"],
            target_width=kwargs["target_width"],
            target_height=kwargs["target_height"],
        )
        config.ensure_directories()
        made_config["config"] = config
        return config

    def _fake_step(tag, extra=None):
        class _Fake:
            CODECS = ("auto", "libx265")  # validate() reads CODECS off Step05

            def __init__(self, *a, **k):
                pass

            def run(self, config):
                if extra:
                    extra(config)
                return tag

        _Fake.__name__ = f"_Fake{tag}"
        return _Fake

    def _step2(config):
        config.original_fps = 30.0
        config.total_frames = 4
        config.duration_sec = 0.1
        config.interpolation_factor = 1000.0 / 30.0

    def _step3(config):
        for i in (1, 2):
            (config.interpolated_720p / f"frame_{i:08d}.png").write_bytes(b"f")
        config.interpolated_frame_count = 2

    def _step5(config):
        Path(config.final_output_path).write_bytes(b"video")

    _fake_s3 = _fake_step("s3", _step3)
    # expected_interpolated_total() reuses the step's own math:
    _fake_s3.target_count = staticmethod(lambda n, d, f, fps: 2)
    monkeypatch.setattr(step01, "setup_environment", _fake_setup)
    monkeypatch.setattr(step02, "Step02Frames", _fake_step("s2", _step2))
    monkeypatch.setattr(step03, "Step03Interpolate", _fake_s3)
    monkeypatch.setattr(step04, "Step04Upscale", _fake_step("s4"))
    monkeypatch.setattr(step05, "Step05Assemble", _fake_step("s5", _step5))

    bus = ProgressBus()
    opts = RunOptions(input=str(video), workspace=str(tmp_path / "ws"))
    result = run_pipeline(opts, bus)
    # Regression: the run must NOT cancel itself when a watcher shuts down.
    assert result.status == "done"
    assert result.steps_completed == [1, 2, 3, 4, 5]
    assert result.output == str(tmp_path / "in_8k_1000fps.mp4")
    assert Path(result.output).is_file()
    assert any(
        isinstance(e, StatusEvent) and e.status == "step-done" for e in bus.poll()
    )


def test_run_pipeline_stop_between_steps_keeps_checkpoint(tmp_path: Path, monkeypatch):
    import pipeline.step01_environment as step01
    import pipeline.step02_frames as step02
    import pipeline.step03_interpolate as step03
    import pipeline.step04_upscale as step04
    import pipeline.step05_assemble as step05
    from pipeline.config import PipelineConfig

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    stop = threading.Event()

    def _fake_setup(**kwargs):
        config = PipelineConfig(
            input_video_path=kwargs["input_video_path"],
            final_output_path=kwargs["final_output_path"],
            workspace_root=kwargs["workspace_root"],
        )
        config.ensure_directories()
        return config

    class _Step:
        CODECS = ("auto",)
        finish = None

        def __init__(self, *a, **k):
            pass

        def run(self, config):
            if self.finish is not None:
                self.finish(config)
            return "ok"

    class _S2(_Step):
        def run(self, config):
            config.original_fps = 30.0
            config.total_frames = 4
            config.duration_sec = 0.1
            config.interpolation_factor = 10.0
            return "s2"

    class _S3(_Step):
        @staticmethod
        def target_count(n, d, f, fps):
            return 4

        def run(self, config):
            stop.set()  # user hits ⏹ while step 3 runs
            config.interpolated_frame_count = 4
            return "s3"

    monkeypatch.setattr(step01, "setup_environment", _fake_setup)
    monkeypatch.setattr(step02, "Step02Frames", _S2)
    monkeypatch.setattr(step03, "Step03Interpolate", _S3)
    monkeypatch.setattr(step04, "Step04Upscale", _Step)
    monkeypatch.setattr(step05, "Step05Assemble", _Step)

    result = run_pipeline(
        RunOptions(input=str(video), workspace=str(tmp_path / "ws")), stop=stop
    )
    assert result.status == "cancelled"
    assert result.steps_completed == [1, 2, 3]  # stops BEFORE step 4
    assert "checkpoint" in result.message
    # ... and the checkpoint on disk really allows a later resume:
    from pipeline.checkpoint import PipelineState

    state = PipelineState.load(tmp_path / "ws" / "pipeline_state.json")
    assert state.last_completed_step == 3


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def test_cli_resolution_model_tile_ui_flags():
    from main import build_parser

    args = build_parser().parse_args(
        ["--resolution", "4K", "--model", "x4plus-anime",
         "--tile-size", "auto", "--ui-port", "1234"]
    )
    assert args.resolution == "4K"
    assert args.esrgan_model == "x4plus-anime"  # --model alias works
    assert args.tile_size == "auto"
    assert args.ui_port == 1234 and args.ui is False

    defaults = build_parser().parse_args([])
    assert defaults.resolution is None
    assert defaults.esrgan_model == "x4plus"  # alias does not clobber default
    assert defaults.tile_size is None


def test_cli_model_and_esrgan_model_last_wins():
    from main import build_parser

    args = build_parser().parse_args(
        ["--model", "x4plus-anime", "--esrgan-model", "x4plus"]
    )
    assert args.esrgan_model == "x4plus"


def test_cli_bad_resolution_and_tile_exit_2(capsys):
    from main import main

    assert main(["--input", "a.mp4", "--output", "b.mp4", "--resolution", "nope"]) == 2
    assert main(["--input", "a.mp4", "--output", "b.mp4", "--tile-size", "huge"]) == 2


def test_cli_ui_delegates_to_app(monkeypatch):
    import main as main_module

    calls = {}

    class _FakeApp:
        @staticmethod
        def launch_ui(port=7860, share=False):
            calls["port"] = port
            calls["share"] = share

    monkeypatch.setitem(sys.modules, "app", _FakeApp)
    assert main_module.main(["--ui", "--ui-port", "1234", "--ui-share"]) == 0
    assert calls == {"port": 1234, "share": True}


# ---------------------------------------------------------------------------
# app.py without Gradio installed
# ---------------------------------------------------------------------------
def test_create_app_without_gradio_helpful_error(monkeypatch):
    import app as app_module

    monkeypatch.setitem(sys.modules, "gradio", None)  # force ImportError
    with pytest.raises(RuntimeError, match="requirements-ui.txt"):
        app_module.create_app()
