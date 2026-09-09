"""Step 9 — profiling, GPU optimisation, ring writer, benchmark report.

Torch-free by design: CUDA/stream/precision plumbing is exercised with fake
``torch`` modules, and the ONNX path runs for real on onnxruntime-CPU with
tiny synthetic graphs.
"""

from __future__ import annotations

import ast
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from pipeline.esrgan.backends import TorchESRGANBackend, create_upscaler
from pipeline.esrgan.onnx_backend import (
    OnnxESRGANBackend,
    export_esrgan_onnx,
    main as onnx_main,
    pick_providers,
)
from pipeline.exceptions import EsrganInferenceError, UpscaleError
from pipeline.perf import (
    BenchmarkReport,
    Profiler,
    RingBufferWriter,
    StepBench,
    VramSampler,
    build_benchmark_report,
    detect_accelerators,
    resolve_torch_precision,
    torch_dtype,
)
from pipeline.rife.backends import TorchRifeBackend
from pipeline.step03_interpolate import Step03Interpolate
from pipeline.step04_upscale import Step04Upscale


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------
@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step09")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


def _fake_torch(capability=(8, 0), cuda_available: bool = True) -> SimpleNamespace:
    def _cap():
        if isinstance(capability, Exception):
            raise capability
        return capability

    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=_cap,
        Stream=lambda: "STREAM",
        stream=lambda s: _nullcontext(),
        synchronize=lambda: None,
    )
    return SimpleNamespace(
        cuda=cuda, float32="f32", float16="f16", bfloat16="bf16",
    )


class _nullcontext:
    def __enter__(self):  # noqa: ANN204, D102
        return None

    def __exit__(self, *a):  # noqa: ANN001, D102
        return False


def _make_upsample_onnx(path: Path, scale: int, size: int = 8) -> Path:
    """Write a minimal nearest-neighbour x{scale} ONNX graph (Resize-11)."""
    import onnx
    from onnx import TensorProto, helper

    # Dynamic H/W like a real ESRGAN export (tiling feeds odd-size crops).
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h4", "w4"])
    roi = helper.make_tensor("roi", TensorProto.FLOAT, [8], [0.0] * 8)
    scales = helper.make_tensor(
        "scales", TensorProto.FLOAT, [4], [1.0, 1.0, float(scale), float(scale)]
    )
    # Resize-11 takes exactly one of scales/sizes: give scales, omit sizes.
    node = helper.make_node(
        "Resize", ["input", "roi", "scales", ""], ["output"], mode="nearest"
    )
    graph = helper.make_graph([node], "nn-up", [inp], [out], [roi, scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7  # loadable by old and new onnxruntime alike
    onnx.save(model, str(path))
    return path


# ---------------------------------------------------------------------------
# RingBufferWriter (§3)
# ---------------------------------------------------------------------------
def test_ring_preserves_fifo_order_and_count(tmp_path: Path, silent_logger):
    seen: List[str] = []

    def _write(path: Path, frame: Any) -> None:
        seen.append(path.name)

    writer = RingBufferWriter(_write, capacity=8, logger=silent_logger).start()
    for i in range(5):
        writer.submit(tmp_path / f"f{i}.png", i)
    assert writer.close() == 5
    assert seen == [f"f{i}.png" for i in range(5)]


def test_ring_submit_before_start_and_after_close_raise(tmp_path, silent_logger):
    writer = RingBufferWriter(lambda p, f: None, logger=silent_logger)
    with pytest.raises(RuntimeError, match="before start"):
        writer.submit(tmp_path / "x.png", 0)
    writer.start()
    writer.close()
    with pytest.raises(RuntimeError, match="after close"):
        writer.submit(tmp_path / "x.png", 0)
    assert writer.close() == 0  # idempotent


def test_ring_rejects_bad_capacity(silent_logger):
    with pytest.raises(ValueError, match="capacity"):
        RingBufferWriter(lambda p, f: None, capacity=0, logger=silent_logger)


def test_ring_worker_error_is_fail_loud(tmp_path, silent_logger):
    def _write(path: Path, frame: Any) -> None:
        if frame == 1:
            raise IOError("disk on fire")

    writer = RingBufferWriter(_write, capacity=8, logger=silent_logger).start()
    writer.submit(tmp_path / "f0.png", 0)
    writer.submit(tmp_path / "f1.png", 1)
    writer.submit(tmp_path / "f2.png", 2)
    with pytest.raises(RuntimeError, match="Background writer failed: disk on fire"):
        writer.close()


def test_ring_submit_surfaces_worker_error_fast(tmp_path, silent_logger):
    entered = threading.Event()
    release = threading.Event()

    def _write(path: Path, frame: Any) -> None:
        if frame == "boom":
            raise IOError("boom")
        entered.set()
        release.wait(timeout=5)

    writer = RingBufferWriter(_write, capacity=8, logger=silent_logger).start()
    writer.submit(tmp_path / "slow.png", "slow")  # occupies the worker
    assert entered.wait(timeout=5)
    writer.submit(tmp_path / "bad.png", "boom")  # queued behind it
    release.set()
    deadline = time.monotonic() + 5
    while writer._error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer._error is not None
    with pytest.raises(RuntimeError, match="Background writer failed"):
        writer.submit(tmp_path / "late.png", "late")
    with pytest.raises(RuntimeError, match="Background writer failed"):
        writer.close()


# ---------------------------------------------------------------------------
# Profiler (§1)
# ---------------------------------------------------------------------------
def test_profiler_stage_records_calls_totals_and_pct():
    profiler = Profiler()
    with profiler.stage("io_read"):
        pass
    with profiler.stage("io_read"):
        pass
    with profiler.stage("inference"):
        pass
    stats = profiler.stats()
    assert stats["io_read"]["calls"] == 2
    assert stats["io_read"]["total_s"] >= 0
    assert stats["io_read"]["avg_ms"] >= 0
    assert stats["inference"]["calls"] == 1
    assert stats["io_read"]["pct"] + stats["inference"]["pct"] == pytest.approx(100.0)


def test_profiler_disabled_records_nothing():
    profiler = Profiler(enabled=False)
    with profiler.stage("x"):
        pass
    assert profiler.stats() == {}
    assert "(no stages recorded)" in profiler.render_text()


def test_profiler_cprofile_roundtrip():
    def _hot() -> int:
        return sum(range(200))

    profiler = Profiler()
    profiler.start_cprofile()
    for _ in range(5):
        _hot()
    text = profiler.stop_cprofile(top_n=10)
    assert "_hot" in text
    assert profiler.stop_cprofile() == "(cProfile was not started)"


def test_cuda_sync_caches_torch_lookup():
    import pipeline.perf as perf

    perf._cuda_sync()
    assert perf._TORCH_KNOWN is True  # no import attempt per span


def test_torch_span_yields_none_without_torch():
    profiler = Profiler(torch_profile=True)
    with profiler.torch_span("step3/interpolate-all") as span:
        assert span is None
    assert profiler.torch_tables == []


# ---------------------------------------------------------------------------
# Precision resolution (§2)
# ---------------------------------------------------------------------------
def test_precision_auto_cuda_cpu_defaults(silent_logger):
    assert resolve_torch_precision(_fake_torch(), "cuda", None, None, silent_logger) == "fp16"
    assert resolve_torch_precision(_fake_torch(), "cpu", None, None, silent_logger) == "fp32"
    assert resolve_torch_precision(_fake_torch(), "cuda", "auto", None, silent_logger) == "fp16"


def test_precision_explicit_beats_fp16_flag(silent_logger):
    torch = _fake_torch()
    assert resolve_torch_precision(torch, "cuda", "fp32", True, silent_logger) == "fp32"
    assert resolve_torch_precision(torch, "cpu", "fp16", None, silent_logger) == "fp32"
    assert resolve_torch_precision(torch, "cpu", None, True, silent_logger) == "fp32"


def test_precision_bf16_needs_ampere(silent_logger):
    assert resolve_torch_precision(_fake_torch((8, 6)), "cuda", "bf16", None, silent_logger) == "bf16"
    assert resolve_torch_precision(_fake_torch((8, 0)), "cuda", "bf16", None, silent_logger) == "bf16"
    assert resolve_torch_precision(_fake_torch((7, 5)), "cuda", "bf16", None, silent_logger) == "fp16"
    assert (
        resolve_torch_precision(_fake_torch(RuntimeError("nope")), "cuda", "bf16", None, silent_logger)
        == "fp16"
    )


def test_precision_invalid_raises(silent_logger):
    with pytest.raises(ValueError, match="precision"):
        resolve_torch_precision(_fake_torch(), "cuda", "fp8", None, silent_logger)


def test_torch_dtype_maps_names():
    torch = _fake_torch()
    assert torch_dtype(torch, "fp32") == "f32"
    assert torch_dtype(torch, "fp16") == "f16"
    assert torch_dtype(torch, "bf16") == "bf16"


# ---------------------------------------------------------------------------
# CUDA stream plumbing (fake torch — no GPU needed)
# ---------------------------------------------------------------------------
def test_rife_stream_plumbing(silent_logger):
    torch = _fake_torch()
    backend = TorchRifeBackend(logger=silent_logger)
    assert backend._resolve_device(torch) == "cuda"
    backend.device = "cuda"
    assert backend._make_stream(torch) == "STREAM"
    backend._torch = torch
    backend._stream = "STREAM"
    with backend._compute_ctx():
        pass
    backend._sync_if_stream()  # must not raise

    cpu = TorchRifeBackend(logger=silent_logger)
    cpu.device = cpu._resolve_device(_fake_torch(cuda_available=False))
    assert cpu.device == "cpu"
    assert cpu._make_stream(torch) is None

    off = TorchRifeBackend(async_transfers=False, logger=silent_logger)
    off.device = "cuda"
    assert off._make_stream(torch) is None


def test_esrgan_stream_plumbing(silent_logger):
    torch = _fake_torch()
    backend = TorchESRGANBackend(logger=silent_logger)
    backend.device = "cuda"
    assert backend._make_stream(torch) == "STREAM"
    backend._torch = torch
    backend._stream = "STREAM"
    with backend._compute_ctx():
        pass
    backend._sync_if_stream()

    off = TorchESRGANBackend(async_transfers=False, logger=silent_logger)
    off.device = "cuda"
    assert off._make_stream(torch) is None


# ---------------------------------------------------------------------------
# VRAM sampler (§4)
# ---------------------------------------------------------------------------
def test_vram_sampler_collects_and_summarizes():
    sampler = VramSampler(lambda: (100.0, 1000.0), interval=0.01)
    sampler.start()
    deadline = time.monotonic() + 5
    while not sampler.samples and time.monotonic() < deadline:
        time.sleep(0.01)
    sampler.stop()
    summary = sampler.summary()
    assert summary is not None
    assert summary["peak_mb"] == 100.0
    assert summary["avg_mb"] == 100.0
    assert summary["total_mb"] == 1000.0
    assert summary["samples"] >= 1


def test_vram_sampler_empty_and_broken_probe():
    assert VramSampler(lambda: None).summary() is None

    def _bad():
        raise RuntimeError("no gpu")

    sampler = VramSampler(_bad, interval=0.01)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    assert sampler.summary() is None


# ---------------------------------------------------------------------------
# Benchmark report (§4)
# ---------------------------------------------------------------------------
def test_stepbench_derived_metrics():
    bench = StepBench(wall_s=2.0, frames=100)
    assert bench.ms_per_frame == pytest.approx(20.0)
    assert bench.processing_fps == pytest.approx(50.0)
    assert StepBench(wall_s=1.0, frames=0).ms_per_frame is None
    assert StepBench(wall_s=1.0, frames=0).processing_fps is None
    assert StepBench(wall_s=0.0, frames=10).processing_fps is None


def test_report_render_save_roundtrip(tmp_path: Path):
    report = BenchmarkReport(
        target_fps=1000.0,
        target_size=[1280, 720],
        steps={"step3": StepBench(wall_s=2.0, frames=100)},
        stages={"step3": {"inference": {"total_s": 1.5, "avg_ms": 15.0, "pct": 75.0}}},
        vram={"peak_mb": 500.0, "avg_mb": 400.0, "samples": 10},
        accelerators={"torch": False, "cuda": False},
    )
    text = report.render_text()
    assert "Benchmark report" in text
    assert "step3" in text
    assert "processing vs 1000 FPS target" in text
    assert "peak 500 MB" in text

    saved = report.save(tmp_path / "benchmark.json")
    loaded = BenchmarkReport.load(saved)
    assert loaded.steps["step3"].frames == 100
    assert loaded.vram["peak_mb"] == 500.0
    assert loaded.generated_at != ""


def test_build_benchmark_report_groups_stages():
    profiler = Profiler()
    with profiler.stage("step3/io_read"):
        pass
    with profiler.stage("step4/inference"):
        pass
    with profiler.stage("unscoped"):
        pass
    report = build_benchmark_report(
        target_fps=1000.0,
        target_size=(7680, 4320),
        profiler=profiler,
        step_wall={3: 1.0, 4: 2.0, 5: 0.5},
        step_frames={3: 10, 4: 20},
    )
    assert set(report.stages) == {"step3", "step4", "unscoped"}
    assert report.steps["step3"].frames == 10
    assert report.steps["step4"].ms_per_frame == pytest.approx(100.0)
    assert report.steps["step5"].frames == 0
    assert set(report.accelerators) == {"torch", "cuda", "onnxruntime", "tensorrt"}


def test_detect_accelerators_cheap_scan():
    caps = detect_accelerators()
    assert caps.onnxruntime is True  # installed in this environment
    assert caps.torch is False
    assert caps.cuda_probed is False
    caps2 = detect_accelerators(import_torch=True)
    # CUDA is only probed when torch exists (cheap scan otherwise).
    assert caps2.cuda_probed == caps2.torch
    if not caps2.torch:
        assert caps2.cuda is False


# ---------------------------------------------------------------------------
# Step wiring (§1 + §3)
# ---------------------------------------------------------------------------
def test_step3_rejects_bad_step9_options(silent_logger):
    with pytest.raises(ValueError, match="precision"):
        Step03Interpolate(precision="fp8", logger=silent_logger)
    with pytest.raises(ValueError, match="writer_queue"):
        Step03Interpolate(writer_queue=-1, logger=silent_logger)


def test_step4_rejects_bad_step9_options(silent_logger):
    with pytest.raises(ValueError, match="precision"):
        Step04Upscale(precision="fp8", logger=silent_logger)
    with pytest.raises(ValueError, match="accel"):
        Step04Upscale(accel="tensorrt", logger=silent_logger)


def test_step3_passes_precision_to_rife_backend(monkeypatch, silent_logger):
    import pipeline.rife.backends as rife_backends

    seen: Dict[str, Any] = {}

    def _fake_create(name: str, logger=None, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(rife_backends, "create_backend", _fake_create)
    step = Step03Interpolate(
        backend="rife", precision="bf16", async_transfers=False,
        logger=silent_logger,
    )
    assert step._resolve_backend().name == "rife"
    assert seen["precision"] == "bf16"
    assert seen["async_transfers"] is False


def test_step3_profiler_and_sync_writer_full_run(tmp_path: Path, silent_logger):
    """Blend run with profiler attached + writer_queue=0 (sync path)."""
    from pipeline.config import PipelineConfig
    from pipeline.rife.backends import BlendBackend
    from tests.test_step03_interpolate import FakeIO, make_sources

    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    config = PipelineConfig(
        input_video_path=video,
        final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
        original_fps=30.0,
        total_frames=2,
        duration_sec=2 / 30,
        interpolation_factor=1000.0 / 30.0,
    )
    mapping = make_sources(config, [0.0, 100.0])
    profiler = Profiler()
    step = Step03Interpolate(
        backend_obj=BlendBackend(logger=silent_logger),
        frame_io=FakeIO(mapping),
        writer_queue=0,
        profiler=profiler,
        static_threshold=None,
        cut_threshold=None,
        logger=silent_logger,
    )
    result = step.run(config)
    assert result.validation_ok is True
    assert set(profiler.stages) >= {
        "step3/io_read", "step3/shortcut", "step3/inference", "step3/io_write",
    }


def test_step4_accel_guardrails(silent_logger):
    with pytest.raises(UpscaleError, match="--accel onnx needs ONNX weights"):
        Step04Upscale(accel="onnx", weights="m.pth", logger=silent_logger)._resolve_backend(
            0, (64, 64)
        )
    with pytest.raises(UpscaleError, match="--accel none forces the PyTorch backend"):
        Step04Upscale(accel="none", weights="m.onnx", logger=silent_logger)._resolve_backend(
            0, (64, 64)
        )
    with pytest.raises(UpscaleError, match="needs explicit weights"):
        Step04Upscale(backend="onnx", weights=None, logger=silent_logger)._resolve_backend(
            0, (64, 64)
        )


def test_step4_explicit_onnx_backend(tmp_path: Path, silent_logger):
    weights = _make_upsample_onnx(tmp_path / "m.onnx", scale=4)
    backend = Step04Upscale(
        backend="onnx", weights=weights, logger=silent_logger
    )._resolve_backend(0, (32, 32))
    assert isinstance(backend, OnnxESRGANBackend)


# ---------------------------------------------------------------------------
# ONNX backend (§2 — real onnxruntime, tiny synthetic graphs)
# ---------------------------------------------------------------------------
def test_pick_providers_prefers_fastest_available():
    import onnxruntime as ort

    assert pick_providers(ort) == ["CPUExecutionProvider"]
    assert pick_providers(ort, ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    with pytest.raises(EsrganInferenceError, match="not available"):
        pick_providers(ort, ["NopeExecutionProvider"])


def test_onnx_backend_upscales_end_to_end(tmp_path: Path, silent_logger):
    import numpy as np

    weights = _make_upsample_onnx(tmp_path / "x4.onnx", scale=4)
    backend = OnnxESRGANBackend(
        weights=weights, model="x4plus", target_size=(32, 32), logger=silent_logger
    )
    backend.load()
    assert backend.providers[0] == "CPUExecutionProvider"
    frame = (np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3) % 251)
    out = backend.upscale(frame)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
    assert (out[0, 0] == frame[0, 0]).all()  # nearest-neighbour exactness
    backend.unload()


def test_onnx_backend_tiled_matches_direct(tmp_path: Path, silent_logger):
    import numpy as np

    weights = _make_upsample_onnx(tmp_path / "x4.onnx", scale=4)
    rng = np.random.RandomState(9)
    frame = rng.randint(0, 256, size=(8, 8, 3)).astype(np.uint8)
    direct = OnnxESRGANBackend(
        weights=weights, model="x4plus", target_size=(32, 32), logger=silent_logger
    )
    direct.load()
    tiled = OnnxESRGANBackend(
        weights=weights, model="x4plus", tile=4, tile_pad=1,
        target_size=(32, 32), logger=silent_logger,
    )
    tiled.load()
    assert (np.asarray(tiled.upscale(frame)) == np.asarray(direct.upscale(frame))).all()


def test_onnx_backend_rejects_wrong_scale(tmp_path: Path, silent_logger):
    import numpy as np

    weights = _make_upsample_onnx(tmp_path / "x2.onnx", scale=2)
    backend = OnnxESRGANBackend(
        weights=weights, model="x4plus", target_size=(32, 32), logger=silent_logger
    )
    backend.load()
    with pytest.raises(EsrganInferenceError, match="does not match"):
        backend.upscale(np.zeros((8, 8, 3), dtype=np.uint8))


def test_onnx_backend_load_errors(tmp_path: Path, silent_logger):
    import numpy as np

    with pytest.raises(EsrganInferenceError, match="not found"):
        OnnxESRGANBackend(
            weights=tmp_path / "missing.onnx", logger=silent_logger
        ).load()
    pth = tmp_path / "m.pth"
    pth.write_bytes(b"x")
    with pytest.raises(EsrganInferenceError, match="needs a .onnx file"):
        OnnxESRGANBackend(weights=pth, logger=silent_logger).load()
    with pytest.raises(EsrganInferenceError, match="Unknown model"):
        OnnxESRGANBackend(weights=tmp_path / "m.onnx", model="nope")
    weights = _make_upsample_onnx(tmp_path / "x4.onnx", scale=4)
    with pytest.raises(EsrganInferenceError, match="not available"):
        OnnxESRGANBackend(
            weights=weights, providers=["NopeExecutionProvider"],
            logger=silent_logger,
        ).load()
    backend = OnnxESRGANBackend(weights=weights, logger=silent_logger)
    with pytest.raises(EsrganInferenceError, match="not loaded"):
        backend.upscale(np.zeros((8, 8, 3), dtype=np.uint8))
    backend.load()
    with pytest.raises(EsrganInferenceError, match="RGB"):
        backend.upscale(np.zeros((8, 8), dtype=np.uint8))


def test_factory_auto_selects_onnx_for_onnx_weights(silent_logger):
    backend = create_upscaler(
        "esrgan", logger=silent_logger, weights="model.onnx",
        target_size=(32, 32),
    )
    assert isinstance(backend, OnnxESRGANBackend)
    explicit = create_upscaler(
        "onnx", logger=silent_logger, weights="model.onnx", target_size=(32, 32)
    )
    assert isinstance(explicit, OnnxESRGANBackend)
    with pytest.raises(EsrganInferenceError, match="'onnx'"):
        create_upscaler("nope", logger=silent_logger)
    with pytest.raises(EsrganInferenceError, match="backend 'onnx'"):
        TorchESRGANBackend(weights="model.onnx", logger=silent_logger)


def test_export_needs_torch_and_cli_returns_2(tmp_path: Path, silent_logger, capsys):
    with pytest.raises(EsrganInferenceError, match="PyTorch"):
        export_esrgan_onnx(tmp_path / "m.pth", "x4plus", logger=silent_logger)
    assert onnx_main(["--weights", "m.pth", "--model", "x4plus"]) == 2
    assert "error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI flags (§1 + §4)
# ---------------------------------------------------------------------------
def test_cli_step9_flags_and_accel_validation(capsys):
    import main as cli

    args = cli.build_parser().parse_args(
        ["--precision", "bf16", "--accel", "none", "--profile",
         "--cprofile", "--torch-profile", "--no-async-transfer"]
    )
    assert args.precision == "bf16"
    assert args.accel == "none"
    assert args.profile and args.cprofile and args.torch_profile
    assert args.no_async_transfer is True
    assert cli.main(["--upscale-backend", "resize", "--accel", "onnx"]) == 2


def test_save_and_print_benchmark_writes_files(tmp_path: Path, silent_logger, capsys):
    import main as cli
    from pipeline.config import PipelineConfig

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    config = PipelineConfig(
        input_video_path=video,
        final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
    )
    config.interpolated_frame_count = 10
    config.upscaled_frame_count = 20
    profiler = Profiler()
    with profiler.stage("step3/inference"):
        pass
    cli._save_and_print_benchmark(
        config, profiler, 3.0, {3: 1.0, 4: 2.0}, None, "cprof-table",
        silent_logger,
    )
    out = capsys.readouterr().out
    assert "Benchmark report" in out
    assert "cprof-table" in out
    data = json.loads((tmp_path / "ws" / "benchmark.json").read_text())
    assert data["steps"]["step3"]["frames"] == 10
    assert (tmp_path / "ws" / "cprofile.txt").is_file()


# ---------------------------------------------------------------------------
# WebUI options (§1 + §4)
# ---------------------------------------------------------------------------
def test_runoptions_step9_defaults_cli_and_validation(tmp_path: Path):
    from pipeline.webui import RunOptions

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    opts = RunOptions(input=str(video))
    assert opts.precision == "auto"
    assert opts.accel == "auto"
    assert opts.profile is False
    assert opts.async_transfers is True
    assert opts.validate() == []
    assert "--precision" not in opts.to_cli_args()

    opts2 = RunOptions(
        input=str(video), precision="fp16", accel="none", profile=True,
        async_transfers=False,
    )
    cli = opts2.to_cli_args()
    assert ["--precision", "fp16", "--accel", "none", "--profile",
            "--no-async-transfer"] == [a for a in cli if a in (
                "--precision", "fp16", "--accel", "none", "--profile",
                "--no-async-transfer")]
    assert RunOptions(input=str(video), precision="fp8").validate() != []
    assert RunOptions(input=str(video), accel="onnx").validate() != []


def test_run_pipeline_profile_writes_benchmark(tmp_path: Path, monkeypatch):
    import pipeline.step01_environment as step01
    import pipeline.step02_frames as step02
    import pipeline.step03_interpolate as step03
    import pipeline.step04_upscale as step04
    import pipeline.step05_assemble as step05
    from pipeline.config import PipelineConfig
    from pipeline.webui import ProgressBus, RunOptions, run_pipeline

    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")

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
        return config

    def _fake_step(tag, extra=None):
        class _Fake:
            CODECS = ("auto", "libx265")

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

    _fake_s3 = _fake_step("s3", _step3)
    _fake_s3.target_count = staticmethod(lambda n, d, f, fps: 2)
    monkeypatch.setattr(step01, "setup_environment", _fake_setup)
    monkeypatch.setattr(step02, "Step02Frames", _fake_step("s2", _step2))
    monkeypatch.setattr(step03, "Step03Interpolate", _fake_s3)
    monkeypatch.setattr(step04, "Step04Upscale", _fake_step("s4"))
    monkeypatch.setattr(step05, "Step05Assemble", _fake_step("s5"))

    opts = RunOptions(
        input=str(video), workspace=str(tmp_path / "ws"), profile=True,
        interp_backend="blend", upscale_backend="resize",
    )
    result = run_pipeline(opts, ProgressBus())
    assert result.status == "done"
    data = json.loads((tmp_path / "ws" / "benchmark.json").read_text())
    assert data["steps"]["step3"]["frames"] == 2


def test_app_run_signature_matches_inputs():
    """Regression: Gradio passes inputs positionally — counts must match."""
    tree = ast.parse((Path(__file__).parent.parent / "app.py").read_text())
    run_fn = click_inputs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run":
            run_fn = node
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "click":
            for kw in node.keywords:
                if kw.arg == "inputs" and isinstance(kw.value, ast.List):
                    click_inputs = kw.value
    assert run_fn is not None and click_inputs is not None
    assert len(run_fn.args.args) == len(click_inputs.elts) == 16
