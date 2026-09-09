"""Unit tests for Step 3 — no torch/cv2/numpy needed (fakes injected).

Only the tests marked with ``importorskip("numpy")`` need numpy (the
static/cut shortcuts and the blend backend genuinely operate on arrays).
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from pipeline.config import PipelineConfig
from pipeline.exceptions import (
    InterpolationError,
    RifeInferenceError,
    RifeWeightsError,
)
from pipeline.rife.backends import RifeBackend, create_backend
from pipeline.rife.io import FrameIO
from pipeline.rife.weights import (
    _confirmation_url,
    _extract_flownet_from_zip,
    _is_generic_zip,
    cache_path,
    ensure_weights,
)
from pipeline.step03_interpolate import Step03Interpolate


# ---------------------------------------------------------------------------
# Fakes (duck-typed frames: plain floats — no numpy anywhere)
# ---------------------------------------------------------------------------
class FakeBackend(RifeBackend):
    name = "fake"

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False
        self.calls: List[tuple] = []

    def load(self) -> None:
        self.loaded = True

    def interpolate_batch(self, batch0: Any, batch1: Any, timestep: float = 0.5) -> Any:
        assert len(batch0) == len(batch1) and len(batch0) > 0
        self.calls.append((list(batch0), list(batch1), timestep))
        return [(1.0 - timestep) * a + timestep * b for a, b in zip(batch0, batch1)]

    def unload(self) -> None:
        self.unloaded = True


class FakeIO(FrameIO):
    def __init__(self, values: Dict[str, float]) -> None:
        self.values = dict(values)
        self.writes: Dict[str, Any] = {}

    def read(self, path: Path) -> Any:
        return self.values[path.name]

    def write(self, path: Path, frame: Any) -> None:
        self.writes[path.name] = frame


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step03")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


@pytest.fixture
def video_and_config(tmp_path: Path) -> PipelineConfig:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    return PipelineConfig(
        input_video_path=video,
        final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
        original_fps=30.0,
        total_frames=4,
        duration_sec=0.1,  # 4 frames @30fps ≈ 0.1s -> T = 100 outputs
        interpolation_factor=1000.0 / 30.0,
    )


def make_sources(config: PipelineConfig, values: List[float]) -> Dict[str, float]:
    """Create empty source files on disk; return FakeIO value map."""
    config.temp_raw_frames.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, value in enumerate(values, start=1):
        name = f"frame_{i:06d}.png"
        (config.temp_raw_frames / name).write_bytes(b"")
        mapping[name] = value
    return mapping


def make_step(silent_logger, **kwargs) -> Step03Interpolate:
    kwargs.setdefault("static_threshold", None)  # deterministic without numpy
    kwargs.setdefault("cut_threshold", None)
    kwargs.setdefault("logger", silent_logger)
    return Step03Interpolate(**kwargs)


# -- Static math ------------------------------------------------------------------
@pytest.mark.parametrize(
    "factor,expected",
    [(33.33, 6), (16.67, 5), (16.0, 4), (2.0, 1), (1.0, 0), (0.5, 0)],
)
def test_compute_exp(factor, expected):
    assert Step03Interpolate.compute_exp(factor) == expected


def test_compute_exp_invalid():
    with pytest.raises(InterpolationError):
        Step03Interpolate.compute_exp(0)
    with pytest.raises(InterpolationError):  # 100x needs exp=7 > max 6
        Step03Interpolate.compute_exp(100.0, max_exp=6)


@pytest.mark.parametrize(
    "num,exp,expected",
    [(60, 6, 59 * 64 + 1), (2, 1, 3), (4, 0, 4)],
)
def test_dense_count(num, exp, expected):
    assert Step03Interpolate.dense_count(num, exp) == expected


def test_target_count():
    assert Step03Interpolate.target_count(60, 2.0, 33.33, 1000.0) == 2000
    assert Step03Interpolate.target_count(4, None, 1000 / 30, 1000.0) == 101


def test_selection_indices_exact_ends_and_monotonic():
    sel = Step03Interpolate.selection_indices(3777, 2000)
    assert len(sel) == 2000
    assert sel[0] == 0 and sel[-1] == 3776
    assert all(b >= a for a, b in zip(sel, sel[1:]))
    assert Step03Interpolate.selection_indices(5, 3) == [0, 2, 4]
    with pytest.raises(InterpolationError):
        Step03Interpolate.selection_indices(0, 5)


# -- Subdivision ---------------------------------------------------------------------
def test_subdivide_pair_values_and_order(silent_logger):
    backend = FakeBackend()
    step = make_step(silent_logger, backend_obj=backend)
    out = step._subdivide_pair(backend, 0.0, 100.0, exp=2)
    assert out == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_subdivide_batching(silent_logger):
    backend = FakeBackend()
    step = make_step(silent_logger, backend_obj=backend, batch_size=2)
    step._subdivide_pair(backend, 0.0, 8.0, exp=2)
    # level 0: 1 pair -> 1 call; level 1: 2 pairs -> 1 call (batch=2)
    assert len(backend.calls) == 2
    assert step._forwards == 2

    backend2 = FakeBackend()
    step2 = make_step(silent_logger, backend_obj=backend2, batch_size=1)
    step2._subdivide_pair(backend2, 0.0, 8.0, exp=2)
    assert len(backend2.calls) == 3  # 1 + 2 micro-batches


# -- Full run (fake backend + fake IO) ----------------------------------------
def test_full_run(video_and_config, silent_logger):
    config = video_and_config
    mapping = make_sources(config, [0.0, 100.0, 200.0, 300.0])
    backend = FakeBackend()
    step = make_step(silent_logger, backend_obj=backend, frame_io=FakeIO(mapping))

    result = step.run(config)

    assert backend.loaded and backend.unloaded  # VRAM freed for Step 4
    assert result.exp == 6
    assert result.source_frames == 4
    assert result.dense_count == 3 * 64 + 1 == 193
    assert result.target_count == 100  # round(0.1 * 1000)
    assert result.written_count == 100
    assert result.validation_ok is True
    # Per pair, batch=4 over levels of 1,2,4,8,16,32 mids:
    # 1+1+1+2+4+8 = 17 forwards x 3 pairs = 51.
    assert result.forwards_run == 51
    assert len(backend.calls) == 51
    # 8-digit names starting at 1 (spec: frame_00000001.png ...).
    writes = step._frame_io.writes
    assert len(writes) == 100
    assert "frame_00000001.png" in writes
    assert "frame_00000100.png" in writes
    # Endpoints preserved through resampling.
    assert writes["frame_00000001.png"] == 0.0
    assert writes["frame_00000100.png"] == 300.0
    # Config enriched + persisted for Step 4.
    assert config.interpolation_exp == 6
    assert config.interpolated_frame_count == 100
    assert config.interpolation_backend == "fake"
    reloaded = PipelineConfig.load(config.workspace_root / "config.json")
    assert reloaded.interpolated_frame_count == 100


def test_full_run_jpg_naming(video_and_config, silent_logger):
    config = video_and_config
    mapping = make_sources(config, [0.0, 10.0])
    step = make_step(
        silent_logger, backend_obj=FakeBackend(), frame_io=FakeIO(mapping),
        output_format="jpg",
    )
    result = step.run(config)
    assert result.frame_pattern == "frame_%08d.jpg"
    assert "frame_00000001.jpg" in step._frame_io.writes


def test_full_run_cleans_stale_outputs(video_and_config, silent_logger):
    config = video_and_config
    mapping = make_sources(config, [0.0, 10.0])
    config.interpolated_720p.mkdir(parents=True, exist_ok=True)
    stale = config.interpolated_720p / "frame_99999999.png"
    stale.write_bytes(b"old")
    step = make_step(silent_logger, backend_obj=FakeBackend(), frame_io=FakeIO(mapping))
    step.run(config)
    assert not stale.exists()


def test_full_run_too_few_frames(video_and_config, silent_logger):
    config = video_and_config
    mapping = make_sources(config, [5.0])
    step = make_step(silent_logger, backend_obj=FakeBackend(), frame_io=FakeIO(mapping))
    with pytest.raises(InterpolationError):
        step.run(config)


def test_full_run_missing_metadata(video_and_config, silent_logger):
    config = video_and_config
    config.original_fps = None
    config.interpolation_factor = None
    mapping = make_sources(config, [0.0, 10.0])
    step = make_step(silent_logger, backend_obj=FakeBackend(), frame_io=FakeIO(mapping))
    with pytest.raises(InterpolationError):
        step.run(config)


def test_full_run_exp_overflow(video_and_config, silent_logger):
    config = video_and_config
    config.original_fps = 5.0  # 200x -> exp=8 > max 6
    config.interpolation_factor = 200.0
    mapping = make_sources(config, [0.0, 10.0])
    step = make_step(silent_logger, backend_obj=FakeBackend(), frame_io=FakeIO(mapping))
    with pytest.raises(InterpolationError):
        step.run(config)


def test_bad_params_raise(silent_logger):
    with pytest.raises(ValueError):
        Step03Interpolate(batch_size=0, logger=silent_logger)
    with pytest.raises(ValueError):
        Step03Interpolate(output_format="bmp", logger=silent_logger)


# -- Shortcuts (need numpy) ----------------------------------------------------------
def test_static_pair_skips_inference(video_and_config, silent_logger):
    np = pytest.importorskip("numpy")
    config = video_and_config
    config.duration_sec = 1 / 30  # P=2 -> T = 33
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    mapping = make_sources(config, [0.0, 0.0])

    class ArrayIO(FakeIO):
        def read(self, path: Path) -> Any:
            return frame.copy()

    backend = FakeBackend()
    step = Step03Interpolate(
        backend_obj=backend, frame_io=ArrayIO(mapping),
        static_threshold=1.0, cut_threshold=60.0, logger=silent_logger,
    )
    result = step.run(config)
    assert result.static_skips == 1
    assert result.forwards_run == 0
    assert result.written_count == result.target_count


def test_cut_pair_copies_first_frame(video_and_config, silent_logger):
    np = pytest.importorskip("numpy")
    config = video_and_config
    config.duration_sec = 1 / 30
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    mapping = make_sources(config, [0.0, 1.0])

    class CutIO(FakeIO):
        def read(self, path: Path) -> Any:
            return black.copy() if "000001" in path.name else white.copy()

    backend = FakeBackend()
    step = Step03Interpolate(
        backend_obj=backend, frame_io=CutIO(mapping),
        static_threshold=1.0, cut_threshold=60.0, logger=silent_logger,
    )
    result = step.run(config)
    assert result.cut_skips == 1
    assert result.forwards_run == 0


# -- Backends ------------------------------------------------------------------------------
def test_create_backend_unknown(silent_logger):
    with pytest.raises(RifeInferenceError):
        create_backend("unknown-thing", logger=silent_logger)


def test_torch_backend_requires_torch(silent_logger):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed -- missing-torch path not testable")
    from pipeline.rife.backends import TorchRifeBackend

    with pytest.raises(RifeInferenceError, match="PyTorch"):
        TorchRifeBackend(logger=silent_logger).load()


def test_torch_backend_bad_device_and_cuda_request(silent_logger):
    from pipeline.rife.backends import TorchRifeBackend

    class NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    backend = TorchRifeBackend(device="cuda", logger=silent_logger)
    with pytest.raises(RifeInferenceError, match="CUDA was requested"):
        backend._resolve_device(NoCuda())
    backend2 = TorchRifeBackend(device="tpu", logger=silent_logger)
    with pytest.raises(RifeInferenceError, match="Unknown device"):
        backend2._resolve_device(NoCuda())


def test_blend_backend_averages(silent_logger):
    np = pytest.importorskip("numpy")
    from pipeline.rife.backends import BlendBackend

    backend = BlendBackend(logger=silent_logger)
    backend.load()
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = backend.interpolate_batch([a], [b], 0.5)
    assert out.shape == (1, 4, 4, 3)
    assert (np.asarray(out)[0] == 50).all()


# -- Weights ---------------------------------------------------------------------------------
def test_ensure_weights_unknown_version(tmp_path, silent_logger):
    with pytest.raises(RifeWeightsError, match="Unknown RIFE version"):
        ensure_weights(version="99", weights_root=tmp_path, logger=silent_logger)


def test_ensure_weights_override_file(tmp_path, silent_logger):
    f = tmp_path / "my.pkl"
    f.write_bytes(b"fake-weights")
    assert ensure_weights(weights_root=tmp_path, weights_override=f,
                          logger=silent_logger) == f


def test_ensure_weights_override_dir(tmp_path, silent_logger):
    d = tmp_path / "train_log"
    d.mkdir()
    (d / "flownet.pkl").write_bytes(b"fake-weights")
    assert ensure_weights(weights_root=tmp_path, weights_override=d,
                          logger=silent_logger) == d / "flownet.pkl"


def test_ensure_weights_override_missing(tmp_path, silent_logger):
    with pytest.raises(RifeWeightsError, match="does not exist"):
        ensure_weights(weights_root=tmp_path,
                       weights_override=tmp_path / "nope.pkl",
                       logger=silent_logger)


def test_ensure_weights_cache_hit(tmp_path, silent_logger):
    cached = cache_path(tmp_path, "4")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"cached-weights")
    assert ensure_weights(version="4", weights_root=tmp_path,
                          logger=silent_logger) == cached


def test_confirmation_url_parsing():
    html = '<a href="/uc?export=download&amp;confirm=tAbC_123&amp;uuid=1-2-3-4-5">'
    url = _confirmation_url("https://drive.google.com/uc?export=download&id=X", html)
    assert url is not None and "confirm=tAbC_123" in url and "uuid=1-2-3-4-5" in url
    assert _confirmation_url("https://x", "<html>no token</html>") is None


def test_is_generic_zip_detection(tmp_path):
    torch_like = tmp_path / "torch.zip"
    with zipfile.ZipFile(torch_like, "w") as zf:
        zf.writestr("archive/data.pkl", b"xx")
    assert _is_generic_zip(torch_like) is False

    archive = tmp_path / "arch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train_log/flownet.pkl", b"yy")
    assert _is_generic_zip(archive) is True

    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"not a zip")
    assert _is_generic_zip(plain) is False


def test_extract_prefers_flownet(tmp_path, silent_logger):
    archive = tmp_path / "models.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("other/model.pkl", b"a")
        zf.writestr("train_log/flownet.pkl", b"b")
    out = _extract_flownet_from_zip(archive, silent_logger)
    assert out.name == "flownet.pkl" and out.read_bytes() == b"b"


# -- Vendored upstream API surface (AST only — no torch import) -----------
def test_vendored_upstream_api_surface():
    repo = Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / "pipeline/rife/vendor/upstream/IFNet.py").read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "IFBlock" in classes and "IFNet" in classes
    ifnet = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == "IFNet")
    methods = {n.name for n in ifnet.body if isinstance(n, ast.FunctionDef)}
    assert {"__init__", "forward"} <= methods

    wrapper = ast.parse((repo / "pipeline/rife/vendor/model.py").read_text())
    wclasses = {n.name for n in ast.walk(wrapper) if isinstance(n, ast.ClassDef)}
    assert "RifeModel" in wclasses

    # Vendored files must not import torch at package-import time paths:
    # (upstream modules import torch at top — they are only loaded lazily).
    pkg_init = (repo / "pipeline/rife/__init__.py").read_text()
    assert "upstream" not in pkg_init
