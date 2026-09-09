"""Unit tests for Step 5 — ffmpeg/ffprobe are mocked (no binaries needed)."""

from __future__ import annotations

import json
import logging
import subprocess as sp
from pathlib import Path
from typing import List, Optional

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import (
    AssemblyError,
    FFmpegNotFoundError,
    OutputVerificationError,
)
from pipeline.step05_assemble import Step05Assemble


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step05")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    cfg = PipelineConfig(
        input_video_path=video,
        final_output_path=tmp_path / "final_8k_1000fps.mp4",
        workspace_root=tmp_path / "ws",
        device="cpu",
    )
    cfg.upscaled_8k.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        (cfg.upscaled_8k / f"frame_8k_{i:08d}.png").write_bytes(b"img")
    cfg.upscaled_frame_pattern = "frame_8k_%08d.png"
    cfg.upscaled_frame_count = 3
    return cfg


@pytest.fixture
def config_with_audio(config: PipelineConfig) -> PipelineConfig:
    audio = config.workspace_root / "input_audio.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF" + b"0" * 100)
    config.has_audio = True
    config.audio_path = audio
    return config


class FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakePopen:
    def __init__(self, lines: List[str], returncode: int = 0, on_start=None) -> None:
        self._lines = lines
        self.returncode = returncode
        if on_start:
            on_start()

    @property
    def stdout(self):
        return iter(self._lines)

    def wait(self):
        return self.returncode


def ffprobe_json(width=7680, height=4320, fps="1000/1", frames="2000",
                 duration="2.0", audio=True) -> str:
    streams = [{"codec_type": "video", "width": width, "height": height,
                "avg_frame_rate": fps, "nb_frames": frames}]
    if audio:
        streams.append({"codec_type": "audio"})
    return json.dumps({"streams": streams, "format": {"duration": duration}})


def make_step(silent_logger, **kwargs) -> Step05Assemble:
    kwargs.setdefault("ffmpeg_bin", "/fake/ffmpeg")
    kwargs.setdefault("ffprobe_bin", "/fake/ffprobe")
    kwargs.setdefault("logger", silent_logger)
    return Step05Assemble(**kwargs)


# -- Command construction ---------------------------------------------------------
def test_build_command_video_only(config, silent_logger):
    step = make_step(silent_logger)
    cmd = step.build_command(config, "libx265", None)
    assert cmd[:2] == ["/fake/ffmpeg", "-y"]
    assert ["-framerate", "1000", "-start_number", "1", "-i",
            str(config.upscaled_8k / "frame_8k_%08d.png")] == cmd[2:8]
    assert "-c:v" in cmd and "libx265" in cmd
    assert cmd[cmd.index("-crf") + 1] == "19"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-r") + 1] == "1000"
    assert "-c:a" not in cmd and "-shortest" not in cmd
    assert ["-movflags", "+faststart"] == cmd[cmd.index("-movflags"):cmd.index("-movflags") + 2]
    assert cmd[-3:] == ["-progress", "pipe:1", "-nostats"] or cmd[-4:-1] == ["-progress", "pipe:1", "-nostats"]
    assert cmd[-1] == str(config.final_output_path)


def test_build_command_with_audio_mkv_no_faststart(config_with_audio, silent_logger):
    config_with_audio.final_output_path = config_with_audio.final_output_path.with_suffix(".mkv")
    step = make_step(silent_logger)
    cmd = step.build_command(config_with_audio, "hevc_nvenc", config_with_audio.audio_path)
    assert cmd.count("-i") == 2
    assert str(config_with_audio.audio_path) in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-shortest" not in cmd  # -t trim is used instead (see code note)
    assert cmd[cmd.index("-t") + 1] == "0.003000"  # 3 frames @ 1000fps
    assert "-movflags" not in cmd  # mkv needs no faststart
    nvenc = cmd[cmd.index("-c:v"):cmd.index("-c:v") + 11]
    assert nvenc == ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq",
                     "-rc", "vbr", "-cq", "19", "-pix_fmt"]


def test_build_command_extra_args_overwrite_and_preset(config, silent_logger):
    step = make_step(silent_logger, overwrite=False, encoder_preset="ultrafast",
                     ffmpeg_args="-x265-params log-level=error -threads 4")
    cmd = step.build_command(config, "libx265", None)
    assert cmd[1] == "-n"
    assert cmd[cmd.index("-preset") + 1] == "ultrafast"
    assert "-x265-params" in cmd and "log-level=error" in cmd
    assert cmd[-1] == str(config.final_output_path)  # output stays last


def test_codec_args_variants(silent_logger):
    step = make_step(silent_logger)
    svt = step._codec_args("libsvtav1")
    assert svt[:4] == ["-c:v", "libsvtav1", "-preset", "6"]
    av1nv = step._codec_args("av1_nvenc")
    assert av1nv[2:4] == ["-preset", "p6"] and "-cq" in av1nv
    with pytest.raises(AssemblyError):
        step._codec_args("mpeg2video")


def test_bad_params_raise(silent_logger):
    with pytest.raises(ValueError):
        Step05Assemble(video_codec="prores", logger=silent_logger)
    with pytest.raises(ValueError):
        Step05Assemble(crf=99, logger=silent_logger)
    with pytest.raises(ValueError):
        Step05Assemble(crf=-1, logger=silent_logger)


# -- Encoder selection -----------------------------------------------------------------
def test_probe_encoder(monkeypatch, silent_logger):
    step = make_step(silent_logger)
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "", 0))
    assert step.probe_encoder("libx265") is True
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "Unknown encoder 'x'", 0))
    assert step.probe_encoder("nope") is False
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "", 1))
    assert step.probe_encoder("nope") is False


def test_auto_select_cuda_prefers_nvenc(monkeypatch, config, silent_logger):
    config.device = "cuda"
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "", 0))
    assert make_step(silent_logger)._select_encoder(config) == "hevc_nvenc"


def test_auto_select_falls_back_to_libx265(monkeypatch, config, silent_logger):
    config.device = "cuda"

    def _run(cmd, **kwargs):
        if "hevc_nvenc" in cmd[-1]:
            return FakeCompleted("", "Unknown encoder 'hevc_nvenc'", 0)
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)
    assert make_step(silent_logger)._select_encoder(config) == "libx265"


def test_auto_select_cpu_uses_libx265(monkeypatch, config, silent_logger):
    seen: List[str] = []

    def _run(cmd, **kwargs):
        seen.append(cmd[-1])
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)
    assert make_step(silent_logger)._select_encoder(config) == "libx265"
    assert seen == ["encoder=libx265"]  # nvenc never probed on CPU


def test_explicit_missing_encoder_raises(monkeypatch, config, silent_logger):
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "Unknown encoder", 0))
    with pytest.raises(AssemblyError, match="not available"):
        make_step(silent_logger, video_codec="hevc_nvenc")._select_encoder(config)


def test_auto_select_all_missing_raises(monkeypatch, config, silent_logger):
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted("", "Unknown encoder", 0))
    with pytest.raises(AssemblyError, match="None of"):
        make_step(silent_logger)._select_encoder(config)


# -- Low-RAM guardrail ---------------------------------------------------------------------
def test_total_ram_gb_sane():
    total = Step05Assemble._total_ram_gb()
    assert total is None or total > 0


def test_warn_if_low_memory_never_raises(monkeypatch, silent_logger):
    step = make_step(silent_logger)
    monkeypatch.setattr(Step05Assemble, "_total_ram_gb", staticmethod(lambda: 3.9))
    step._warn_if_low_memory("libx265")  # warns, returns
    step._warn_if_low_memory("hevc_nvenc")  # skipped for hardware encoders
    monkeypatch.setattr(Step05Assemble, "_total_ram_gb", staticmethod(lambda: None))
    step._warn_if_low_memory("libx265")  # unknown RAM: silent


# -- Encode run ------------------------------------------------------------------------------
def test_run_encode_failure_raises(monkeypatch, config, silent_logger):
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    (config.workspace_root / "ffmpeg_assemble.log").write_text("error: boom")
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: FakePopen(["progress=end"], returncode=1))
    step = make_step(silent_logger)
    with pytest.raises(AssemblyError, match="exit 1"):
        step._run_encode(["/fake/ffmpeg"], 3, config)


# -- Verification -------------------------------------------------------------------------------
def test_verify_ffprobe_pass(monkeypatch, config, silent_logger):
    config.final_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.final_output_path.write_bytes(b"video-bytes")
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted(ffprobe_json()))
    step = make_step(silent_logger)
    assert step._verify_output(config, config.final_output_path, 2000, False) is True


def test_verify_resolution_mismatch_raises(monkeypatch, config, silent_logger):
    config.final_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.final_output_path.write_bytes(b"video-bytes")
    bad = ffprobe_json(width=3840, height=2160)
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted(bad))
    with pytest.raises(OutputVerificationError, match="resolution"):
        make_step(silent_logger)._verify_output(config, config.final_output_path, 2000, False)


def test_verify_fps_mismatch_raises(monkeypatch, config, silent_logger):
    config.final_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.final_output_path.write_bytes(b"video-bytes")
    bad = ffprobe_json(fps="30/1")
    monkeypatch.setattr(sp, "run", lambda *a, **k: FakeCompleted(bad))
    with pytest.raises(OutputVerificationError, match="FPS"):
        make_step(silent_logger)._verify_output(config, config.final_output_path, 2000, False)


def test_verify_disabled_returns_false(config, silent_logger):
    assert make_step(silent_logger, verify=False)._verify_output(
        config, config.final_output_path, 3, False) is False


def test_verify_no_tools_unverified(monkeypatch, config, silent_logger):
    import builtins

    config.final_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.final_output_path.write_bytes(b"video-bytes")
    monkeypatch.setattr(sp, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no ffprobe")))
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("mocked: no cv2")
        return real_import(name, *args, **kwargs)

    import sys

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    step = make_step(silent_logger)
    assert step._verify_output(config, config.final_output_path, 3, False) is False


# -- Full run (mocked) ------------------------------------------------------------------------------
def test_full_run_with_audio(monkeypatch, config_with_audio: PipelineConfig, silent_logger):
    config = config_with_audio
    commands: List[List[str]] = []

    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in cmd[0]:
            return FakeCompleted(ffprobe_json(frames="3"))
        return FakeCompleted("", "", 0)  # encoder probe OK

    monkeypatch.setattr(sp, "run", _run)

    def _popen(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        config.final_output_path.write_bytes(b"final-video" * 100)
        return FakePopen(["frame=1", "frame=2", "frame=3", "progress=end"])

    monkeypatch.setattr(sp, "Popen", _popen)

    result = make_step(silent_logger).run(config)

    assert result.encoder == "libx265"  # cpu auto-select
    assert result.encoded_frames == 3
    assert result.resolution == (7680, 4320)
    assert result.has_audio is True
    assert result.verified is True
    assert config.assembly_codec == "libx265"
    assert config.assembled_frame_count == 3
    assert config.assembly_verified is True
    assert config.assembly_file_size_mb is not None and config.assembly_file_size_mb > 0
    reloaded = PipelineConfig.load(config.workspace_root / "config.json")
    assert reloaded.assembly_codec == "libx265"
    # Audio was on the command line.
    assert any("input_audio.wav" in c for c in commands[0])


def test_full_run_video_only(monkeypatch, config, silent_logger):
    def _run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        if "ffprobe" in cmd[0]:
            return FakeCompleted(ffprobe_json(frames="3", audio=False))
        return FakeCompleted("", "", 0)

    monkeypatch.setattr(sp, "run", _run)

    def _popen(cmd, **kwargs):
        config.final_output_path.write_bytes(b"v" * 100)
        return FakePopen(["frame=3", "progress=end"])

    monkeypatch.setattr(sp, "Popen", _popen)
    result = make_step(silent_logger).run(config)
    assert result.has_audio is False
    assert result.verified is True


def test_full_run_no_frames(tmp_path, silent_logger):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    cfg = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "o.mp4",
        workspace_root=tmp_path / "ws",
    )
    cfg.upscaled_8k.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssemblyError, match="No 8K frames"):
        make_step(silent_logger).run(cfg)


def test_full_run_requires_ffmpeg(monkeypatch, config, silent_logger):
    monkeypatch.setattr("pipeline.step05_assemble.shutil.which", lambda _: None)
    step = Step05Assemble(ffmpeg_bin=None, ffprobe_bin=None, logger=silent_logger)
    with pytest.raises(FFmpegNotFoundError):
        step.run(config)
