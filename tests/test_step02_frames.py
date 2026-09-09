"""Unit tests for Step 2 — ffmpeg/ffprobe/cv2 are all mocked (no binaries needed)."""

from __future__ import annotations

import json
import logging
import subprocess as sp
import sys
import types
from pathlib import Path
from typing import Any, List, Optional

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import (
    FFmpegNotFoundError,
    FrameExtractionError,
    InputVideoNotFoundError,
    MetadataProbeError,
)
from pipeline.step02_frames import Step02Frames, VideoMetadata


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step02")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    video = tmp_path / "input_720p.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


@pytest.fixture
def config(dummy_video: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_video_path=dummy_video,
        final_output_path=tmp_path / "out" / "final.mp4",
        workspace_root=tmp_path / "workspace",
    )


def make_step(silent_logger, **kwargs) -> Step02Frames:
    kwargs.setdefault("ffmpeg_bin", "/fake/ffmpeg")
    kwargs.setdefault("ffprobe_bin", "/fake/ffprobe")
    kwargs.setdefault("logger", silent_logger)
    return Step02Frames(**kwargs)


class FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


FFPROBE_VIDEO_JSON = json.dumps({
    "streams": [{
        "width": 1280, "height": 720,
        "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
        "nb_frames": "60", "duration": "2.000000",
        "codec_name": "h264", "pix_fmt": "yuv420p",
    }],
    "format": {"duration": "2.000000"},
})

FFPROBE_AUDIO_JSON = json.dumps({"streams": [{"index": 1, "codec_name": "aac"}]})
FFPROBE_NO_AUDIO_JSON = json.dumps({"streams": []})


def fake_run_probe(audio: bool = True):
    """Dispatch ffprobe calls: video metadata vs. audio-stream check."""
    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in cmd[0] or "ffprobe" in str(cmd[0]):
            if "a" in cmd:  # -select_streams a
                return FakeCompleted(FFPROBE_AUDIO_JSON if audio else FFPROBE_NO_AUDIO_JSON)
            return FakeCompleted(FFPROBE_VIDEO_JSON)
        return FakeCompleted("", "", 0)
    return _run


# -- parse_frame_rate ------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("30/1", 30.0),
        ("30000/1001", pytest.approx(29.97, abs=0.01)),
        ("25", 25.0),
        ("60.0", 60.0),
        ("0/0", None),
        ("", None),
        ("garbage", None),
        ("30/0", None),
    ],
)
def test_parse_frame_rate(value, expected):
    result = Step02Frames.parse_frame_rate(value)
    assert result == expected if expected is None else result == pytest.approx(expected)


# -- Metadata probing ------------------------------------------------------------------
def test_probe_with_ffprobe(monkeypatch, dummy_video, silent_logger):
    monkeypatch.setattr(sp, "run", fake_run_probe())
    meta = make_step(silent_logger).probe_metadata(dummy_video)
    assert (meta.width, meta.height) == (1280, 720)
    assert meta.fps == 30.0
    assert meta.total_frames == 60
    assert meta.duration_sec == pytest.approx(2.0)
    assert meta.video_codec == "h264"
    assert meta.probe_source == "ffprobe"
    assert meta.total_frames_estimated is False


def test_probe_estimates_frames_when_nb_frames_missing(monkeypatch, dummy_video, silent_logger):
    no_count = json.dumps({
        "streams": [{"width": 1280, "height": 720, "avg_frame_rate": "30/1",
                     "r_frame_rate": "30/1", "duration": "2.0"}],
        "format": {"duration": "2.0"},
    })
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted(no_count))
    meta = make_step(silent_logger).probe_metadata(dummy_video)
    assert meta.total_frames == 60
    assert meta.total_frames_estimated is True


def _install_fake_cv2(monkeypatch, fps=30.0, count=60, w=1280, h=720, opened=True):
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT = 5, 7
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT = 3, 4

    class FakeCap:
        def isOpened(self):
            return opened

        def get(self, prop):
            return {5: fps, 7: count, 3: w, 4: h}.get(prop, 0)

        def release(self):
            pass

    cv2.VideoCapture = lambda *a, **k: FakeCap()
    monkeypatch.setitem(sys.modules, "cv2", cv2)


def test_probe_opencv_fallback(monkeypatch, dummy_video, silent_logger):
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no ffprobe")))
    _install_fake_cv2(monkeypatch)
    step = make_step(silent_logger, ffprobe_bin=None)
    monkeypatch.setattr("pipeline.step02_frames.shutil.which", lambda _: None)
    meta = step.probe_metadata(dummy_video)
    assert meta.probe_source == "opencv"
    assert meta.fps == 30.0
    assert meta.total_frames == 60


def test_probe_failure_raises(monkeypatch, dummy_video, silent_logger):
    import builtins

    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("mocked: no cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    step = make_step(silent_logger, ffprobe_bin=None)
    monkeypatch.setattr("pipeline.step02_frames.shutil.which", lambda _: None)
    with pytest.raises(MetadataProbeError):
        step.probe_metadata(dummy_video)


# -- Audio -------------------------------------------------------------------------------
def test_has_audio_stream_true_false(monkeypatch, dummy_video, silent_logger):
    monkeypatch.setattr(sp, "run", fake_run_probe(audio=True))
    assert make_step(silent_logger).has_audio_stream(dummy_video) is True
    monkeypatch.setattr(sp, "run", fake_run_probe(audio=False))
    assert make_step(silent_logger).has_audio_stream(dummy_video) is False


def test_has_audio_stream_unknown_without_ffprobe(dummy_video, silent_logger):
    assert make_step(silent_logger, ffprobe_bin=None).has_audio_stream(dummy_video) is None


def test_extract_audio_success(monkeypatch, config, silent_logger):
    config.ensure_directories()

    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in str(cmd[0]):
            return FakeCompleted(FFPROBE_AUDIO_JSON)
        Path(cmd[-1]).write_bytes(b"fake-audio" * 100)  # ffmpeg "writes" output
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)
    out = make_step(silent_logger).extract_audio(config)
    assert out is not None and out.name == "input_audio.wav" and out.is_file()


def test_extract_audio_aac_naming(monkeypatch, config, silent_logger):
    config.ensure_directories()

    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in str(cmd[0]):
            return FakeCompleted(FFPROBE_AUDIO_JSON)
        Path(cmd[-1]).write_bytes(b"x" * 10)
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)
    out = make_step(silent_logger, audio_format="aac").extract_audio(config)
    assert out is not None and out.name == "input_audio.aac"


def test_extract_audio_no_stream_skips(monkeypatch, config, silent_logger):
    config.ensure_directories()
    monkeypatch.setattr(sp, "run", fake_run_probe(audio=False))
    assert make_step(silent_logger).extract_audio(config) is None


def test_extract_audio_failure_does_not_raise(monkeypatch, config, silent_logger):
    """Audio is auxiliary: failures warn and return None (Step 8 goes silent)."""
    config.ensure_directories()

    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in str(cmd[0]):
            return FakeCompleted(FFPROBE_AUDIO_JSON)
        return FakeCompleted("", "encoder exploded", 1)

    monkeypatch.setattr(sp, "run", _run)
    assert make_step(silent_logger).extract_audio(config) is None


# -- Frames -------------------------------------------------------------------------------
class FakePopen:
    """Mimics the tiny Popen surface Step02Frames uses."""

    def __init__(self, lines: List[str], returncode: int = 0,
                 on_start=None) -> None:
        self._lines = lines
        self.returncode = returncode
        self._on_start = on_start
        if on_start:
            on_start()

    @property
    def stdout(self):
        return iter(self._lines)

    def wait(self):
        return self.returncode


def test_extract_frames_success(monkeypatch, config, silent_logger):
    config.ensure_directories()
    config.total_frames = 3

    def _create_files():
        for i in range(1, 4):
            (config.temp_raw_frames / f"frame_{i:06d}.png").write_bytes(b"img")

    monkeypatch.setattr(
        sp, "Popen",
        lambda *a, **k: FakePopen(
            ["frame=1", "frame=2", "frame=3", "progress=end"],
            on_start=_create_files,
        ),
    )
    count = make_step(silent_logger).extract_frames(config, "frame_%06d.png")
    assert count == 3


def test_extract_frames_failure_raises(monkeypatch, config, silent_logger):
    config.ensure_directories()
    (config.workspace_root / "ffmpeg_frames.log").write_text("boom")
    monkeypatch.setattr(
        sp, "Popen", lambda *a, **k: FakePopen(["progress=end"], returncode=1)
    )
    with pytest.raises(FrameExtractionError):
        make_step(silent_logger).extract_frames(config, "frame_%06d.png")


def test_extract_frames_cleans_stale_files(monkeypatch, config, silent_logger):
    config.ensure_directories()
    config.total_frames = 1
    stale = config.temp_raw_frames / "frame_999999.png"
    stale.write_bytes(b"old")
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: FakePopen(["progress=end"]))
    make_step(silent_logger).extract_frames(config, "frame_%06d.png")
    assert not stale.exists()


# -- Validation ------------------------------------------------------------------------------
def test_validate_frame_count(silent_logger):
    step = make_step(silent_logger)
    assert step.validate_frame_count(60, 60) is True
    assert step.validate_frame_count(60, 59) is False  # warns, doesn't raise
    assert step.validate_frame_count(None, 12) is True


# -- Constructor validation -----------------------------------------------------------------------
def test_bad_formats_raise(silent_logger):
    with pytest.raises(ValueError):
        Step02Frames(image_format="bmp", logger=silent_logger)
    with pytest.raises(ValueError):
        Step02Frames(audio_format="mp3", logger=silent_logger)


# -- Full run -----------------------------------------------------------------------------------------
def test_full_run(monkeypatch, config, silent_logger):
    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in str(cmd[0]):
            if cmd[cmd.index("-select_streams") + 1] == "a":
                return FakeCompleted(FFPROBE_AUDIO_JSON)
            return FakeCompleted(FFPROBE_VIDEO_JSON)
        if cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"audio" * 100)
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)

    def _popen(cmd, **kwargs):
        for i in range(1, 61):
            (config.temp_raw_frames / f"frame_{i:06d}.png").write_bytes(b"img")
        lines = [f"frame={i}" for i in range(1, 61)] + ["progress=end"]
        return FakePopen(lines)

    monkeypatch.setattr(sp, "Popen", _popen)

    result = make_step(silent_logger).run(config)

    assert result.metadata.fps == 30.0
    assert result.interpolation_factor == pytest.approx(1000.0 / 30.0)
    assert result.estimated_interpolated_frames == 2000
    assert result.has_audio is True
    assert result.audio_path is not None and result.audio_path.is_file()
    assert result.extracted_frame_count == 60
    assert result.validation_ok is True
    # Config object enriched for Step 3 + persisted.
    assert config.original_fps == 30.0
    assert config.total_frames == 60
    assert config.interpolation_factor == pytest.approx(33.33, abs=0.01)
    assert config.metadata["probe_source"] == "ffprobe"
    reloaded = PipelineConfig.load(config.workspace_root / "config.json")
    assert reloaded.total_frames == 60
    assert reloaded.audio_path == config.audio_path


def test_run_requires_ffmpeg(monkeypatch, config, silent_logger):
    monkeypatch.setattr("pipeline.step02_frames.shutil.which", lambda _: None)
    step = Step02Frames(ffmpeg_bin=None, ffprobe_bin=None, logger=silent_logger)
    with pytest.raises(FFmpegNotFoundError):
        step.run(config)


def test_run_missing_video_raises(config, silent_logger):
    config.input_video_path = config.input_video_path.parent / "ghost.mp4"
    with pytest.raises(InputVideoNotFoundError):
        make_step(silent_logger).run(config)


def test_old_config_json_without_step2_fields_still_loads(tmp_path):
    """Backward compat: configs saved by Step 1 v0.1.0 must load fine."""
    old = {
        "input_video_path": "in.mp4", "final_output_path": "out.mp4",
        "workspace_root": "ws", "target_fps": 1000.0,
        "target_width": 7680, "target_height": 4320, "tile_size": 256,
        "device": "cpu", "gpu_name": None, "vram_gb": None,
        "ffmpeg_path": None, "ffmpeg_available": False, "ffmpeg_version": None,
    }
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(old))
    loaded = PipelineConfig.load(path)
    assert loaded.original_fps is None
    assert loaded.metadata == {}
