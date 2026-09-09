"""Unit tests for Step 1 — no GPU / FFmpeg / network required (all mocked)."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import InputVideoNotFoundError
from pipeline.step01_environment import Step01Environment


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step01")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    video = tmp_path / "input_720p.mp4"
    video.write_bytes(b"fake-video-bytes")
    return video


def make_config(dummy_video: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        input_video_path=dummy_video,
        final_output_path=tmp_path / "out" / "final.mp4",
        workspace_root=tmp_path / "workspace",
    )


# -- Tiling heuristic ---------------------------------------------------------
@pytest.mark.parametrize(
    "vram,has_cuda,expected",
    [
        (None, False, 256),   # CPU / unknown -> safe fallback
        (None, True, 256),
        (4.0, True, 256),     # VRAM < 8
        (7.99, True, 256),
        (8.0, True, 512),     # 8..16
        (10.0, True, 512),
        (16.0, True, 512),
        (16.01, True, 1024),  # > 16
        (24.0, True, 1024),
        (24.0, False, 256),   # GPU seen but unusable -> CPU-safe
    ],
)
def test_determine_tile_size(vram, has_cuda, expected):
    assert Step01Environment.determine_tile_size(vram, has_cuda) == expected


# -- Input validation ------------------------------------------------------------
def test_validate_input_video_ok(dummy_video, silent_logger):
    step = Step01Environment(auto_install=False, logger=silent_logger)
    assert step.validate_input_video(dummy_video).is_file()


def test_validate_input_video_missing(tmp_path, silent_logger):
    step = Step01Environment(auto_install=False, logger=silent_logger)
    with pytest.raises(InputVideoNotFoundError):
        step.validate_input_video(tmp_path / "nope.mp4")
    # Must also be catchable as plain FileNotFoundError (spec requirement).
    with pytest.raises(FileNotFoundError):
        step.validate_input_video(tmp_path / "nope.mp4")


# -- FFmpeg ----------------------------------------------------------------------------
def test_check_ffmpeg_missing(monkeypatch, silent_logger):
    monkeypatch.setattr("pipeline.step01_environment.shutil.which", lambda _: None)
    step = Step01Environment(auto_install=False, logger=silent_logger)
    info = step.check_ffmpeg()
    assert info.available is False
    assert info.path is None


def test_check_ffmpeg_found(monkeypatch, silent_logger):
    import subprocess as sp

    monkeypatch.setattr(
        "pipeline.step01_environment.shutil.which", lambda _: "/usr/bin/ffmpeg"
    )

    class Proc:
        returncode = 0
        stdout = "ffmpeg version 6.0 Copyright (c) 2000-2023\nbuilt with gcc"

    monkeypatch.setattr(sp, "run", lambda *a, **k: Proc())
    step = Step01Environment(auto_install=False, logger=silent_logger)
    info = step.check_ffmpeg()
    assert info.available is True
    assert info.path == "/usr/bin/ffmpeg"
    assert "ffmpeg version 6.0" in (info.version or "")


# -- GPU detection -------------------------------------------------------------------
def _fake_torch(vram_gb: float, name: str = "NVIDIA Test GPU"):
    mod = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda idx=0: name,
        get_device_properties=lambda idx=0: types.SimpleNamespace(
            total_memory=int(vram_gb * 1024**3)
        ),
    )
    mod.cuda = cuda
    return mod


def test_detect_gpu_via_torch(monkeypatch, silent_logger):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(10.0, "NVIDIA RTX 3080"))
    step = Step01Environment(auto_install=False, logger=silent_logger)
    info = step.detect_gpu()
    assert info.has_cuda is True
    assert info.device == "cuda"
    assert info.gpu_name == "NVIDIA RTX 3080"
    assert info.vram_gb == pytest.approx(10.0)
    assert info.source == "torch.cuda"


def test_detect_gpu_cpu_fallback(monkeypatch, silent_logger):
    # No torch, no nvidia-smi -> CPU mode, must not crash.
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mocked: no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("pipeline.step01_environment.shutil.which", lambda _: None)
    step = Step01Environment(auto_install=False, logger=silent_logger)
    info = step.detect_gpu()
    assert info.has_cuda is False
    assert info.device == "cpu"


# -- Full run ------------------------------------------------------------------------
def test_full_run_creates_everything(monkeypatch, dummy_video, tmp_path, silent_logger):
    # Force deterministic env: no ffmpeg, CPU, all deps "installed".
    monkeypatch.setattr("pipeline.step01_environment.shutil.which", lambda _: None)
    monkeypatch.setattr(
        Step01Environment, "_is_importable", staticmethod(lambda name: True)
    )
    monkeypatch.setattr(
        Step01Environment, "_installed_version", staticmethod(lambda name: "0.0-test")
    )

    config = make_config(dummy_video, tmp_path)
    step = Step01Environment(auto_install=False, logger=silent_logger)
    result = step.run(config)

    assert result.device == "cpu"
    assert result.tile_size == 256
    assert result.temp_raw_frames.is_dir()
    assert result.interpolated_720p.is_dir()
    assert result.upscaled_8k.is_dir()
    assert result.target_fps == 1000.0
    assert result.target_resolution_str == "7680x4320"
    assert (tmp_path / "out").is_dir()  # output parent auto-created


def test_full_run_missing_video_raises(tmp_path, silent_logger):
    config = PipelineConfig(
        input_video_path=tmp_path / "ghost.mp4",
        final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
    )
    step = Step01Environment(auto_install=False, logger=silent_logger)
    with pytest.raises(InputVideoNotFoundError):
        step.run(config)


def test_tile_size_override(dummy_video, tmp_path, silent_logger, monkeypatch):
    monkeypatch.setattr("pipeline.step01_environment.shutil.which", lambda _: None)
    monkeypatch.setattr(
        Step01Environment, "_is_importable", staticmethod(lambda name: True)
    )
    config = make_config(dummy_video, tmp_path)
    step = Step01Environment(
        auto_install=False, tile_size_override=1024, logger=silent_logger
    )
    result = step.run(config)
    assert result.tile_size == 1024


# -- Config round-trip ------------------------------------------------------------------
def test_config_save_load_roundtrip(dummy_video, tmp_path):
    config = make_config(dummy_video, tmp_path)
    config.ensure_directories()
    saved = config.save(tmp_path / "workspace" / "config.json")
    loaded = PipelineConfig.load(saved)
    assert loaded.input_video_path == config.input_video_path
    assert loaded.target_fps == 1000.0
    assert loaded.tile_size == config.tile_size
