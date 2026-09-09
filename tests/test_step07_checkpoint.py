"""Unit tests for Step 7 — checkpoint & resume (no torch needed).

Resume is scan-based ("disk is the truth"), so these tests use real files in
``tmp_path`` with tiny fake backends / disk-writing fake IO.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pipeline.checkpoint import (
    CheckpointManager,
    PipelineState,
    ResumeDecision,
    StepProgress,
    continuous_prefix_length,
    is_oom_error,
    scan_frame_indices,
    steps_to_run,
)
from pipeline.config import PipelineConfig
from pipeline.esrgan.backends import EsrganBackend
from pipeline.exceptions import CheckpointError
from pipeline.frame_io import FrameIO
from pipeline.rife.backends import RifeBackend
from pipeline.step03_interpolate import Step03Interpolate
from pipeline.step04_upscale import Step04Upscale


OOM_MSG = "CUDA out of memory. Tried to allocate 2.5 GiB."


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step07")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


def _snap(manager: CheckpointManager) -> Dict[str, Any]:
    return manager.snapshot_from_values("in.mp4", "out.mp4", 1000.0, 7680, 4320)


# ---------------------------------------------------------------------------
# Scan helpers + OOM detection (pure functions)
# ---------------------------------------------------------------------------
def test_scan_frame_indices(tmp_path: Path):
    (tmp_path / "frame_00000001.png").write_bytes(b"a")
    (tmp_path / "frame_00000003.png").write_bytes(b"c")
    (tmp_path / "frame_00000002.png").write_bytes(b"b")
    (tmp_path / "frame_00000004.png").write_bytes(b"")  # torn: ignored
    (tmp_path / "frame_00000005.jpg").write_bytes(b"x")  # wrong suffix
    (tmp_path / "frame_preview.png").write_bytes(b"x")  # not numbered
    (tmp_path / "other_00000006.png").write_bytes(b"x")  # wrong prefix
    assert scan_frame_indices(tmp_path, "frame_", ".png") == [1, 2, 3]
    assert scan_frame_indices(tmp_path / "missing", "frame_", ".png") == []


@pytest.mark.parametrize(
    "indices,expected",
    [([], 0), ([2, 3], 0), ([1], 1), ([1, 2, 3, 5], 3), ([1, 2, 3, 4], 4)],
)
def test_continuous_prefix_length(indices, expected):
    assert continuous_prefix_length(indices) == expected


def test_is_oom_error():
    assert is_oom_error(RuntimeError(OOM_MSG)) is True
    try:
        try:
            raise RuntimeError(OOM_MSG)
        except RuntimeError as exc:
            raise ValueError("RIFE inference failed: boom") from exc
    except ValueError as wrapped:
        assert is_oom_error(wrapped) is True  # cause chain inspected
    assert is_oom_error(RuntimeError("plain failure")) is False
    assert is_oom_error(ValueError("nope")) is False

    class OutOfMemoryError(Exception):
        pass

    assert is_oom_error(OutOfMemoryError("x")) is True  # torch-like name


def test_steps_to_run():
    assert steps_to_run(1, 6, 0) == [1, 2, 3, 4, 5, 6]
    assert steps_to_run(1, 6, 2) == [3, 4, 5, 6]
    assert steps_to_run(3, 4, 9) == []
    assert steps_to_run(2, 2, 1) == [2]


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------
def test_state_roundtrip_atomic(tmp_path: Path):
    state = PipelineState(
        workspace=str(tmp_path),
        snapshot={"input_video": "a"},
        last_completed_step=2,
        steps={"step03_interpolate": StepProgress(
            status="in_progress", last_processed_frame_index=812,
            target_count=2000, params={"exp": 6},
        )},
    )
    saved = state.save(tmp_path / "pipeline_state.json")
    assert list(tmp_path.glob("*.tmp-*")) == []  # no temp leftovers
    reloaded = PipelineState.load(saved)
    assert reloaded.last_completed_step == 2
    assert reloaded.steps["step03_interpolate"].last_processed_frame_index == 812
    assert reloaded.steps["step03_interpolate"].params == {"exp": 6}
    assert "step: 2" in reloaded.describe_progress()
    assert "812/2000" in reloaded.describe_progress()


def test_state_load_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        PipelineState.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        PipelineState.load(bad)
    bad.write_text("[1,2]", encoding="utf-8")
    with pytest.raises(CheckpointError, match="not a JSON object"):
        PipelineState.load(bad)


# ---------------------------------------------------------------------------
# Manager lifecycle
# ---------------------------------------------------------------------------
def test_begin_record_complete_persist(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    state = manager.begin_run(_snap(manager), from_step=3)
    assert state.last_completed_step == 2  # explicit range trusted
    manager.record_progress("step03_interpolate", 50, 100, {"exp": 6})
    manager.mark_step_complete(3, {"output_count": 100})
    assert manager.state.last_completed_step == 3
    manager.mark_step_complete(2)  # monotonic: never moves back
    assert manager.state.last_completed_step == 3

    fresh = CheckpointManager(tmp_path, logger=silent_logger)
    reloaded = fresh.load_existing()
    assert reloaded.last_completed_step == 3
    assert reloaded.steps["step03_interpolate"].status == "complete"
    assert reloaded.steps["step03_interpolate"].params["output_count"] == 100


def test_record_without_run_raises(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    with pytest.raises(CheckpointError, match="begin_run"):
        manager.record_progress("step03_interpolate", 1)
    with pytest.raises(CheckpointError, match="begin_run"):
        manager.mark_step_complete(1)


def test_corrupt_state_quarantined_not_fatal(tmp_path: Path, silent_logger):
    (tmp_path / "pipeline_state.json").write_text("{torn", encoding="utf-8")
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    assert manager.load_existing() is None
    assert len(list(tmp_path.glob("pipeline_state.json.corrupt-*"))) == 1
    assert not (tmp_path / "pipeline_state.json").exists()


def test_snapshot_validate_and_norm(tmp_path: Path, silent_logger):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    config = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws", original_fps=30.0, duration_sec=4.0,
    )
    snap = CheckpointManager.snapshot_from_config(config)
    assert snap["original_fps"] == 30.0 and snap["duration_sec"] == 4.0
    assert CheckpointManager.validate_against(snap, dict(snap)) == []
    # Same file via different spelling still matches (path normalisation).
    twin = dict(snap)
    twin["input_video"] = str(tmp_path / "sub" / ".." / "in.mp4")
    assert CheckpointManager.validate_against(snap, twin) == []
    changed = dict(snap)
    changed["target_fps"] = 60.0
    diffs = CheckpointManager.validate_against(snap, changed)
    assert len(diffs) == 1 and "target_fps" in diffs[0]


def test_check_step_params(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    manager.begin_run(_snap(manager))
    assert manager.check_step_params("step04_upscale", {"a": 1}) == []  # no entry
    manager.record_progress("step04_upscale", 3, 5, {"source_count": 5})
    assert manager.check_step_params("step04_upscale", {"source_count": 5}) == []
    diffs = manager.check_step_params("step04_upscale", {"source_count": 4})
    assert len(diffs) == 1 and "source_count" in diffs[0]


def test_refresh_and_restore(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    manager.begin_run(_snap(manager))
    config = PipelineConfig("in.mp4", "out.mp4", workspace_root=tmp_path / "ws")
    config.original_fps = 30.0
    manager.refresh_snapshot(config)
    assert manager.state.snapshot["original_fps"] == 30.0

    fresh_config = PipelineConfig("in.mp4", "out.mp4", workspace_root=tmp_path / "ws")
    restored = manager.restore_metadata(fresh_config, manager.state)
    assert "original_fps" in restored
    assert fresh_config.original_fps == 30.0
    fresh_config.original_fps = 60.0  # present values are never overwritten
    manager.restore_metadata(fresh_config, manager.state)
    assert fresh_config.original_fps == 60.0


# ---------------------------------------------------------------------------
# Startup decisions
# ---------------------------------------------------------------------------
def _manager_with_progress(tmp_path: Path, silent_logger, last_completed=2):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    manager.begin_run(_snap(manager))
    manager.mark_step_complete(last_completed)
    return manager


def test_decide_fresh_without_state(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    decision = manager.decide(_snap(manager), policy="yes")
    assert (decision.action, decision.skip_through_step) == ("fresh", 0)


def test_decide_fresh_without_progress(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    manager.begin_run(_snap(manager))  # no completed steps, no frames
    decision = manager.decide(_snap(manager), policy="yes")
    assert decision.action == "fresh"


def test_decide_resume_and_overwrite(tmp_path: Path, silent_logger):
    manager = _manager_with_progress(tmp_path, silent_logger, last_completed=2)
    snap = _snap(manager)
    resumed = manager.decide(snap, policy="yes")
    assert isinstance(resumed, ResumeDecision)
    assert resumed.action == "resume" and resumed.skip_through_step == 2
    assert resumed.state is not None
    wiped = manager.decide(snap, policy="no")
    assert wiped.action == "overwrite"


def test_decide_mismatch_goes_fresh(tmp_path: Path, silent_logger):
    manager = _manager_with_progress(tmp_path, silent_logger)
    other = manager.snapshot_from_values("OTHER.mp4", "out.mp4", 1000.0, 7680, 4320)
    decision = manager.decide(other, policy="yes")
    assert decision.action == "fresh"
    assert "input_video" in decision.reason


def test_decide_bad_policy(tmp_path: Path, silent_logger):
    manager = CheckpointManager(tmp_path, logger=silent_logger)
    with pytest.raises(ValueError, match="policy"):
        manager.decide({}, policy="maybe")


@pytest.mark.parametrize("answer,expected", [("1", "resume"), ("2", "overwrite"), ("", "resume")])
def test_decide_ask_interactive(tmp_path, silent_logger, monkeypatch, answer, expected):
    manager = _manager_with_progress(tmp_path, silent_logger)
    monkeypatch.setattr("builtins.input", lambda *a: answer)
    decision = manager.decide(_snap(manager), policy="ask", interactive=True)
    assert decision.action == expected


def test_decide_ask_eof_and_garbage_resume(tmp_path, silent_logger, monkeypatch):
    manager = _manager_with_progress(tmp_path, silent_logger)

    def _eof(*a):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert manager.decide(_snap(manager), policy="ask", interactive=True).action == "resume"
    monkeypatch.setattr("builtins.input", lambda *a: "banana")
    assert manager.decide(_snap(manager), policy="ask", interactive=True).action == "resume"


def test_decide_ask_non_interactive_auto_resumes(tmp_path: Path, silent_logger):
    manager = _manager_with_progress(tmp_path, silent_logger)
    decision = manager.decide(_snap(manager), policy="ask", interactive=False)
    assert decision.action == "resume" and decision.skip_through_step == 2


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------
def test_discard_progress_from_step_1(tmp_path: Path, silent_logger):
    ws = tmp_path / "ws"
    (ws / "temp_raw_frames").mkdir(parents=True)
    (ws / "interpolated_720p").mkdir(parents=True)
    (ws / "upscaled_8k").mkdir(parents=True)
    (ws / "temp_raw_frames" / "frame_000001.png").write_bytes(b"r" * 100)
    (ws / "interpolated_720p" / "frame_00000001.png").write_bytes(b"i" * 100)
    (ws / "upscaled_8k" / "frame_8k_00000001.png").write_bytes(b"u" * 100)
    (ws / "input_audio.wav").write_bytes(b"a" * 100)
    (ws / "pipeline_state.json").write_text("{}", encoding="utf-8")
    (ws / "config.json").write_text("{}", encoding="utf-8")
    (ws / "final_8k_1000fps.mp4").write_bytes(b"FINAL")  # sacred, even in ws
    manager = CheckpointManager(ws, logger=silent_logger)

    files, freed = manager.discard_progress(from_step=1)
    assert files == 5 and freed == 402  # 4x100 B frames/audio + "{}" state
    assert (ws / "config.json").is_file()
    assert (ws / "final_8k_1000fps.mp4").read_bytes() == b"FINAL"
    assert list((ws / "upscaled_8k").glob("*")) == []


def test_discard_progress_from_step_4_keeps_earlier(tmp_path: Path, silent_logger):
    ws = tmp_path / "ws"
    (ws / "temp_raw_frames").mkdir(parents=True)
    (ws / "upscaled_8k").mkdir(parents=True)
    raw = ws / "temp_raw_frames" / "frame_000001.png"
    raw.write_bytes(b"r")
    (ws / "upscaled_8k" / "frame_8k_00000001.png").write_bytes(b"u")
    manager = CheckpointManager(ws, logger=silent_logger)
    manager.discard_progress(from_step=4)
    assert raw.is_file()  # step 2 outputs untouched
    assert list((ws / "upscaled_8k").glob("*")) == []


# ---------------------------------------------------------------------------
# Step 4 frame-level resume + OOM retry (real frame files on disk)
# ---------------------------------------------------------------------------
class DiskIO(FrameIO):
    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = dict(values)

    def read(self, path: Path) -> Any:
        return self.values[Path(path).name]

    def write(self, path: Path, frame: Any) -> None:
        Path(path).write_bytes(f"frame={frame}".encode())


class FakeEsrgan(EsrganBackend):
    name = "fake-esrgan"

    def __init__(self, fail_times: int = 0, tile: int = 512) -> None:
        self.calls: List[Any] = []
        self.fail_times = fail_times
        self.tile = tile
        self.loaded = False
        self.unloaded = False
        self.cache_clears = 0

    def load(self) -> None:
        self.loaded = True

    def upscale(self, frame: Any) -> Any:
        self.calls.append(frame)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(OOM_MSG)
        return frame

    def empty_cache(self) -> None:
        self.cache_clears += 1

    def unload(self) -> None:
        self.unloaded = True


def _upscale_config(tmp_path: Path, count: int = 5) -> tuple:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    config = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
    )
    config.interpolated_720p.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i in range(1, count + 1):
        name = f"frame_{i:08d}.png"
        (config.interpolated_720p / name).write_bytes(b"s")
        mapping[name] = i
    return config, mapping


def test_step04_resumes_partial_outputs(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):  # crash after 3 of 5: file 3 is suspect, redone
        (config.upscaled_8k / f"frame_8k_{i:08d}.png").write_bytes(b"old")
    backend = FakeEsrgan()
    result = Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
    ).run(config)
    assert [c for c in backend.calls] == [3, 4, 5]  # no re-processing of 1-2
    assert result.written_count == 5 and result.validation_ok is True
    assert len(list(config.upscaled_8k.glob("frame_8k_*.png"))) == 5
    assert config.upscaled_frame_count == 5


def test_step04_complete_outputs_skip_model_load(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        (config.upscaled_8k / f"frame_8k_{i:08d}.png").write_bytes(b"old")
    backend = FakeEsrgan()
    result = Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
    ).run(config)
    assert backend.loaded is False and backend.calls == []
    assert result.written_count == 5 and result.validation_ok is True


def test_step04_gap_restarts_before_it(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 4, 5):  # file 3 missing -> trust only 1, redo from 2
        (config.upscaled_8k / f"frame_8k_{i:08d}.png").write_bytes(b"old")
    backend = FakeEsrgan()
    Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
    ).run(config)
    assert backend.calls == [2, 3, 4, 5]
    assert len(list(config.upscaled_8k.glob("frame_8k_*.png"))) == 5


def test_step04_torn_last_file_redone(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    (config.upscaled_8k / "frame_8k_00000001.png").write_bytes(b"ok")
    (config.upscaled_8k / "frame_8k_00000002.png").write_bytes(b"ok")
    (config.upscaled_8k / "frame_8k_00000003.png").write_bytes(b"")  # torn
    backend = FakeEsrgan()
    Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
    ).run(config)
    assert backend.calls == [2, 3, 4, 5]


def test_step04_checkpoint_params_mismatch_starts_fresh(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    (config.upscaled_8k / "frame_8k_00000001.png").write_bytes(b"old")
    (config.upscaled_8k / "frame_8k_00000002.png").write_bytes(b"old")
    manager = CheckpointManager(config.workspace_root, logger=silent_logger)
    manager.begin_run(_snap(manager))
    manager.record_progress("step04_upscale", 2, 5, {"source_count": 999})
    backend = FakeEsrgan()
    Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger,
        checkpoint=manager,
    ).run(config)
    assert backend.calls == [1, 2, 3, 4, 5]  # incompatible partials discarded


def test_step04_records_progress(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path)
    manager = CheckpointManager(config.workspace_root, logger=silent_logger)
    manager.begin_run(_snap(manager))
    Step04Upscale(
        backend_obj=FakeEsrgan(), frame_io=DiskIO(mapping), logger=silent_logger,
        checkpoint=manager, checkpoint_every=2,
    ).run(config)
    entry = manager.state.steps["step04_upscale"]
    assert entry.last_processed_frame_index == 5 and entry.target_count == 5
    assert entry.params["source_count"] == 5
    assert json.loads((config.workspace_root / "pipeline_state.json").read_text())


def test_step04_oom_halves_tile_and_retries_same_frame(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path, count=3)
    backend = FakeEsrgan(fail_times=1, tile=512)
    result = Step04Upscale(
        backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
    ).run(config)
    assert backend.tile == 256 and result.tile_used == 256
    assert backend.cache_clears >= 1
    assert backend.calls == [1, 1, 2, 3]  # frame 1 retried, none skipped
    assert result.written_count == 3 and config.esrgan_tile == 256


def test_step04_oom_gives_up_at_floor(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path, count=2)
    backend = FakeEsrgan(fail_times=999, tile=512)
    with pytest.raises(RuntimeError, match="out of memory"):
        Step04Upscale(
            backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
        ).run(config)
    assert backend.tile == 64  # 512 -> 256 -> 128 -> 64, then raise


def test_step04_oom_without_tile_or_retry_opt_out(tmp_path: Path, silent_logger):
    config, mapping = _upscale_config(tmp_path, count=2)

    backend = FakeEsrgan(fail_times=1, tile=512)
    del backend.tile  # e.g. whole-image / non-tiled backends
    with pytest.raises(RuntimeError, match="out of memory"):
        Step04Upscale(
            backend_obj=backend, frame_io=DiskIO(mapping), logger=silent_logger
        ).run(config)
    assert backend.calls == [1]

    backend2 = FakeEsrgan(fail_times=1, tile=512)
    with pytest.raises(RuntimeError, match="out of memory"):
        Step04Upscale(
            backend_obj=backend2, frame_io=DiskIO(mapping), logger=silent_logger,
            oom_retry=False,
        ).run(config)
    assert backend2.tile == 512  # untouched


# ---------------------------------------------------------------------------
# Step 3 pair-level resume + OOM retry (real frame files on disk)
# ---------------------------------------------------------------------------
class FakeRife(RifeBackend):
    name = "fake-rife"

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: List[tuple] = []
        self.fail_times = fail_times
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def interpolate_batch(self, batch0: Any, batch1: Any, timestep: float = 0.5) -> Any:
        self.calls.append((list(batch0), list(batch1)))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError(OOM_MSG)
        return [(a + b) / 2 for a, b in zip(batch0, batch1)]

    def empty_cache(self) -> None:
        pass

    def unload(self) -> None:
        self.unloaded = True


def _interp_config(tmp_path: Path, values: List[float]) -> tuple:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    config = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
        original_fps=30.0, total_frames=len(values), duration_sec=0.01,
        interpolation_factor=1000.0 / 30.0,
    )
    config.temp_raw_frames.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, value in enumerate(values, start=1):
        name = f"frame_{i:06d}.png"
        (config.temp_raw_frames / name).write_bytes(b"s")
        mapping[name] = value
    return config, mapping


def _run_step3(config, mapping, silent_logger, **kwargs):
    kwargs.setdefault("static_threshold", None)
    kwargs.setdefault("cut_threshold", None)
    kwargs.setdefault("logger", silent_logger)
    return Step03Interpolate(**kwargs).run(config)


def test_step03_resumes_from_crash_without_redo(tmp_path: Path, silent_logger):
    config, mapping = _interp_config(tmp_path, [0.0, 100.0, 200.0])
    backend = FakeRife()
    full = _run_step3(config, mapping, silent_logger, backend_obj=backend,
                       frame_io=DiskIO(mapping))
    assert full.target_count == 10 and full.written_count == 10
    assert len(backend.calls) == 34  # 17 forwards x 2 pairs

    # Simulate a crash: outputs 8..10 never hit the disk.
    for i in (8, 9, 10):
        (config.interpolated_720p / f"frame_{i:08d}.png").unlink()
    manager = CheckpointManager(config.workspace_root, logger=silent_logger)
    manager.begin_run(_snap(manager))
    manager.record_progress("step03_interpolate", 7, 10, {
        "source_count": 3, "exp": 6, "target_count": 10,
        "pattern": "frame_%08d.png",
    })
    backend2 = FakeRife()
    resumed = _run_step3(config, mapping, silent_logger, backend_obj=backend2,
                          frame_io=DiskIO(mapping), checkpoint=manager)
    assert len(backend2.calls) == 17  # pair 2 only — pair 1 not recomputed
    assert resumed.written_count == 10 and resumed.validation_ok is True
    assert len(list(config.interpolated_720p.glob("frame_*.png"))) == 10


def test_step03_complete_outputs_skip_model_load(tmp_path: Path, silent_logger):
    config, mapping = _interp_config(tmp_path, [0.0, 50.0])
    _run_step3(config, mapping, silent_logger, backend_obj=FakeRife(),
               frame_io=DiskIO(mapping))
    backend = FakeRife()
    result = _run_step3(config, mapping, silent_logger, backend_obj=backend,
                         frame_io=DiskIO(mapping))
    assert backend.loaded is False and backend.calls == []
    assert result.written_count == result.target_count == 10
    assert result.validation_ok is True


def test_step03_oom_halves_batch_and_retries_same_pair(tmp_path: Path, silent_logger):
    config, mapping = _interp_config(tmp_path, [0.0, 50.0])
    step = Step03Interpolate(
        backend_obj=FakeRife(fail_times=1), frame_io=DiskIO(mapping),
        static_threshold=None, cut_threshold=None, logger=silent_logger,
    )
    result = step.run(config)
    assert step.batch_size == 2  # 4 -> 2 after the OOM
    assert result.written_count == 10 and result.validation_ok is True


def test_step03_oom_at_batch_one_reraises(tmp_path: Path, silent_logger):
    config, mapping = _interp_config(tmp_path, [0.0, 50.0])
    step = Step03Interpolate(
        backend_obj=FakeRife(fail_times=1), frame_io=DiskIO(mapping),
        static_threshold=None, cut_threshold=None, logger=silent_logger,
        batch_size=1,
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        step.run(config)


def test_step03_records_progress(tmp_path: Path, silent_logger):
    config, mapping = _interp_config(tmp_path, [0.0, 100.0, 200.0])
    manager = CheckpointManager(config.workspace_root, logger=silent_logger)
    manager.begin_run(_snap(manager))
    _run_step3(config, mapping, silent_logger, backend_obj=FakeRife(),
               frame_io=DiskIO(mapping), checkpoint=manager, checkpoint_every=1)
    entry = manager.state.steps["step03_interpolate"]
    assert entry.last_processed_frame_index == 10 and entry.target_count == 10
    assert entry.params["exp"] == 6


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def test_cli_resume_flags_parse():
    from main import build_parser

    args = build_parser().parse_args(
        ["--resume", "no", "--checkpoint-every", "5", "--no-checkpoint"]
    )
    assert args.resume == "no"
    assert args.checkpoint_every == 5
    assert args.no_checkpoint is True
    defaults = build_parser().parse_args([])
    assert defaults.resume == "ask" and defaults.no_checkpoint is False
