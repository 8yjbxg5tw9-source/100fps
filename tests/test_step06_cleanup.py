"""Unit tests for Step 6 — real temp files in tmp_path (no mocks needed mostly)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import CleanupSafetyError
from pipeline.step06_cleanup import Step06Cleanup


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step06")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def populated(tmp_path: Path) -> PipelineConfig:
    """Workspace with frame dirs + audio + a valid final video."""
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    final = tmp_path / "final_8k_1000fps.mp4"
    final.write_bytes(b"final-video-bytes" * 100)
    cfg = PipelineConfig(
        input_video_path=video,
        final_output_path=final,
        workspace_root=tmp_path / "ws",
        assembly_verified=True,
    )
    cfg.temp_raw_frames.mkdir(parents=True, exist_ok=True)
    cfg.interpolated_720p.mkdir(parents=True, exist_ok=True)
    cfg.upscaled_8k.mkdir(parents=True, exist_ok=True)
    (cfg.temp_raw_frames / "frame_000001.png").write_bytes(b"r" * 1000)
    (cfg.temp_raw_frames / "frame_000002.png").write_bytes(b"r" * 2000)
    (cfg.interpolated_720p / "frame_00000001.png").write_bytes(b"i" * 4000)
    (cfg.upscaled_8k / "frame_8k_00000001.png").write_bytes(b"u" * 8000)
    (cfg.upscaled_8k / "nested").mkdir(exist_ok=True)
    (cfg.upscaled_8k / "nested" / "x.png").write_bytes(b"u" * 500)
    audio = cfg.workspace_root / "input_audio.wav"
    audio.write_bytes(b"a" * 1500)
    cfg.has_audio = True
    cfg.audio_path = audio
    return cfg


# -- Safety gate --------------------------------------------------------------------
def test_missing_final_refuses_and_keeps_everything(populated, silent_logger):
    populated.final_output_path.unlink()
    with pytest.raises(CleanupSafetyError, match="missing"):
        Step06Cleanup(logger=silent_logger).run(populated)
    assert populated.temp_raw_frames.is_dir()  # nothing deleted
    assert populated.upscaled_8k.is_dir()
    assert populated.audio_path.is_file()


def test_empty_final_refuses(populated, silent_logger):
    populated.final_output_path.write_bytes(b"")
    with pytest.raises(CleanupSafetyError, match="empty"):
        Step06Cleanup(logger=silent_logger).run(populated)
    assert populated.temp_raw_frames.is_dir()


def test_unverified_final_warns_but_proceeds(populated, silent_logger):
    populated.assembly_verified = False
    result = Step06Cleanup(logger=silent_logger).run(populated)
    assert result.freed_bytes > 0  # file check is authoritative


# -- Happy path --------------------------------------------------------------------------
def test_full_purge(populated, silent_logger):
    expected = 1000 + 2000 + 4000 + 8000 + 500 + 1500
    result = Step06Cleanup(logger=silent_logger).run(populated)

    assert result.freed_bytes == expected
    assert result.freed_gb == pytest.approx(expected / (1024**3))
    assert result.deleted_files == 6
    assert result.failed == []
    assert result.dry_run is False
    assert not populated.temp_raw_frames.exists()
    assert not populated.interpolated_720p.exists()
    assert not populated.upscaled_8k.exists()
    assert not populated.audio_path.exists()
    # Provenance kept: final video + config + workspace root survive.
    assert populated.final_output_path.is_file()
    assert populated.workspace_root.is_dir()
    assert populated.cleanup_completed is True
    assert populated.cleanup_freed_bytes == expected
    assert populated.cleanup_failed_count == 0
    reloaded = PipelineConfig.load(populated.workspace_root / "config.json")
    assert reloaded.cleanup_completed is True
    assert "freed" in reloaded.summary()


def test_dry_run_deletes_nothing(populated, silent_logger):
    result = Step06Cleanup(dry_run=True, logger=silent_logger).run(populated)
    assert result.dry_run is True
    assert result.freed_bytes == 1000 + 2000 + 4000 + 8000 + 500 + 1500
    assert populated.temp_raw_frames.is_dir()
    assert populated.audio_path.is_file()
    assert populated.cleanup_completed is None  # not "done"


def test_keep_flags(populated, silent_logger):
    result = Step06Cleanup(keep_raw=True, keep_audio=True, logger=silent_logger).run(populated)
    assert populated.temp_raw_frames.is_dir()
    assert populated.audio_path.is_file()
    assert not populated.interpolated_720p.exists()
    assert not populated.upscaled_8k.exists()
    assert result.freed_bytes == 4000 + 8000 + 500


def test_already_clean_no_crash(tmp_path, silent_logger):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    final = tmp_path / "final.mp4"
    final.write_bytes(b"xyz")
    cfg = PipelineConfig(
        input_video_path=video, final_output_path=final,
        workspace_root=tmp_path / "ws",
    )
    cfg.workspace_root.mkdir(parents=True, exist_ok=True)
    result = Step06Cleanup(logger=silent_logger).run(cfg)
    assert result.freed_bytes == 0
    assert result.failed == []


# -- Containment ------------------------------------------------------------------------------
def test_audio_outside_workspace_is_never_deleted(populated, tmp_path, silent_logger):
    outsider = tmp_path / "precious.wav"
    outsider.write_bytes(b"do-not-delete" * 100)
    populated.audio_path = outsider
    result = Step06Cleanup(logger=silent_logger).run(populated)
    assert outsider.is_file()  # survived!
    assert outsider.read_bytes() == b"do-not-delete" * 100
    assert result.freed_bytes == 1000 + 2000 + 4000 + 8000 + 500  # audio excluded


def test_workspace_root_itself_never_targeted(populated, silent_logger):
    assert Step06Cleanup._is_within_workspace(
        populated.workspace_root, populated.workspace_root.resolve()) is False
    assert Step06Cleanup._is_within_workspace(
        populated.temp_raw_frames, populated.workspace_root.resolve()) is True


# -- Failure tolerance -------------------------------------------------------------------------------
def test_locked_file_reported_not_raised(populated, monkeypatch, silent_logger):
    real_rmtree = shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if Path(path) == populated.upscaled_8k:
            raise PermissionError("file is locked by another process")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("pipeline.step06_cleanup.shutil.rmtree", flaky_rmtree)
    result = Step06Cleanup(logger=silent_logger).run(populated)

    assert len(result.failed) == 1
    assert "locked" in result.failed[0][1]
    assert populated.upscaled_8k.is_dir()  # survived, correctly
    assert not populated.temp_raw_frames.exists()  # others still purged
    assert populated.cleanup_completed is False  # partial: not "done"


def test_rmtree_onexc_fallback_for_old_python(populated, monkeypatch, silent_logger):
    """On Python < 3.12 rmtree() has no `onexc` kwarg — must use `onerror`."""
    real_rmtree = shutil.rmtree
    seen = {}

    def fake_rmtree(path, *args, **kwargs):
        seen.update(kwargs)
        if "onexc" in kwargs:
            raise TypeError("unexpected keyword 'onexc' (old python)")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("pipeline.step06_cleanup.shutil.rmtree", fake_rmtree)
    Step06Cleanup(logger=silent_logger).run(populated)
    assert "onerror" in seen
    assert not populated.temp_raw_frames.exists()
