"""Unit tests for Step 4 — no torch needed (fakes + numpy-gated array tests)."""

from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pipeline.config import PipelineConfig
from pipeline.esrgan.backends import EsrganBackend, create_upscaler
from pipeline.esrgan.tiling import plan_tiles, upscale_tiled_numpy
from pipeline.esrgan.weights import (
    ESRGAN_MODELS,
    _validate,
    cache_path,
    ensure_esrgan_weights,
)
from pipeline.esrgan.writer import AsyncFrameWriter
from pipeline.exceptions import (
    EsrganInferenceError,
    EsrganWeightsError,
    UpscaleError,
)
from pipeline.frame_io import FrameIO
from pipeline.step04_upscale import Step04Upscale


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeUpscaler(EsrganBackend):
    name = "fake-esr"

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False
        self.seen: List[Any] = []

    def load(self) -> None:
        self.loaded = True

    def upscale(self, frame: Any) -> Any:
        self.seen.append(frame)
        return ("upscaled", frame)

    def unload(self) -> None:
        self.unloaded = True


class FakeIO(FrameIO):
    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = dict(values)
        self.writes: Dict[str, Any] = {}

    def read(self, path: Path) -> Any:
        return self.values[path.name]

    def write(self, path: Path, frame: Any) -> None:
        self.writes[path.name] = frame


@pytest.fixture
def silent_logger() -> logging.Logger:
    logger = logging.getLogger("test-step04")
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
        final_output_path=tmp_path / "out.mp4",
        workspace_root=tmp_path / "ws",
        tile_size=256,
    )
    for i in range(1, 4):
        cfg.interpolated_720p.mkdir(parents=True, exist_ok=True)
        (cfg.interpolated_720p / f"frame_{i:08d}.png").write_bytes(b"")
    return cfg


# -- Tiling geometry (pure int math — no numpy) -------------------------------
def _boxes_overlap(a: tuple, b: tuple) -> bool:
    (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = a, b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@pytest.mark.parametrize(
    "h,w,tile,pad,scale",
    [
        (720, 1280, 256, 10, 4),   # production 720p case
        (720, 1280, 512, 10, 4),
        (720, 1280, 1024, 10, 4),
        (100, 70, 32, 5, 4),       # non-divisible both axes
        (32, 32, 64, 8, 4),        # tile bigger than image
        (50, 50, 16, 0, 4),        # zero halo
        (7, 13, 4, 2, 2),          # tiny odd case, scale 2
    ],
)
def test_plan_tiles_exact_partition(h, w, tile, pad, scale):
    plans = plan_tiles(h, w, tile, pad, scale)
    assert len(plans) > 0
    # 1. kept regions cover the canvas exactly (area) without overlap.
    total = sum(
        (p.out_y1 - p.out_y0) * (p.out_x1 - p.out_x0) for p in plans
    )
    assert total == (h * scale) * (w * scale)
    boxes = [(p.out_y0, p.out_y1, p.out_x0, p.out_x1) for p in plans]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _boxes_overlap(boxes[i], boxes[j])
    # 2. input crops stay in-bounds; kept size matches crop math.
    for p in plans:
        assert 0 <= p.in_y0 < p.in_y1 <= h
        assert 0 <= p.in_x0 < p.in_x1 <= w
        assert (p.tile_y1 - p.tile_y0) == (p.out_y1 - p.out_y0)
        assert (p.tile_x1 - p.tile_x0) == (p.out_x1 - p.out_x0)
        assert (p.tile_y1 - p.tile_y0) <= (p.in_y1 - p.in_y0) * scale
        assert (p.tile_x1 - p.tile_x0) <= (p.in_x1 - p.in_x0) * scale


def test_plan_tiles_invalid():
    with pytest.raises(ValueError):
        plan_tiles(64, 64, 0, 10, 4)  # 0 = direct path, not plannable
    with pytest.raises(ValueError):
        plan_tiles(64, 64, 32, -1, 4)
    with pytest.raises(ValueError):
        plan_tiles(0, 64, 32, 10, 4)


def test_tiled_matches_direct_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(42)

    def repeat4(crop):
        a = np.asarray(crop)
        return np.repeat(np.repeat(a, 4, axis=0), 4, axis=1)

    for (h, w, tile, pad) in [(100, 70, 32, 5), (64, 64, 64, 8), (33, 65, 16, 3)]:
        img = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        tiled = upscale_tiled_numpy(img, 4, tile, pad, repeat4)
        assert (np.asarray(tiled) == repeat4(img)).all()


def test_tiled_rejects_bad_fn_output():
    np = pytest.importorskip("numpy")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        upscale_tiled_numpy(img, 4, 8, 2, lambda c: np.zeros((4, 4, 3)))


# -- Async writer -------------------------------------------------------------------
def test_writer_order_and_count(silent_logger):
    io = FakeIO({})
    with AsyncFrameWriter(io, max_queue=2, logger=silent_logger) as writer:
        for i in range(5):
            writer.submit(Path(f"f{i}.png"), i)
    assert writer.count == 5
    assert [io.writes[f"f{i}.png"] for i in range(5)] == [0, 1, 2, 3, 4]


def test_writer_propagates_errors(silent_logger):
    class FailingIO(FrameIO):
        def read(self, path: Path) -> Any:
            raise AssertionError("unused")

        def write(self, path: Path, frame: Any) -> None:
            raise IOError("disk is on fire")

    writer = AsyncFrameWriter(FailingIO(), max_queue=4, logger=silent_logger)
    writer.start()
    writer.submit(Path("x.png"), 1)
    with pytest.raises(RuntimeError, match="disk is on fire"):
        writer.close()


def test_writer_misuse_raises(silent_logger):
    io = FakeIO({})
    writer = AsyncFrameWriter(io, logger=silent_logger)
    with pytest.raises(RuntimeError, match="before start"):
        writer.submit(Path("x.png"), 1)
    writer.start()
    writer.close()
    with pytest.raises(RuntimeError, match="after close"):
        writer.submit(Path("x.png"), 1)
    assert writer.close() == 0  # idempotent
    with pytest.raises(ValueError):
        AsyncFrameWriter(io, max_queue=0, logger=silent_logger)


# -- Full run (fakes) ------------------------------------------------------------------
def test_full_run(config, silent_logger):
    mapping = {f"frame_{i:08d}.png": f"img{i}" for i in range(1, 4)}
    backend = FakeUpscaler()
    fake_io = FakeIO(mapping)
    step = Step04Upscale(backend_obj=backend, frame_io=fake_io, logger=silent_logger)

    result = step.run(config)

    assert backend.loaded and backend.unloaded  # VRAM freed for Step 5
    assert result.source_frames == 3
    assert result.written_count == 3
    assert result.target_size == (7680, 4320)
    assert result.tile_used == 256  # defaulted from Step 1 config.tile_size
    assert result.frame_pattern == "frame_8k_%08d.png"
    assert result.validation_ok is True
    assert backend.seen == ["img1", "img2", "img3"]  # chronological order
    assert set(fake_io.writes) == {
        "frame_8k_00000001.png", "frame_8k_00000002.png", "frame_8k_00000003.png",
    }
    assert config.upscaled_frame_count == 3
    assert config.upscaled_frame_pattern == "frame_8k_%08d.png"
    assert config.esrgan_tile == 256
    reloaded = PipelineConfig.load(config.workspace_root / "config.json")
    assert reloaded.upscaled_frame_count == 3


def test_full_run_tile_override_and_jpg(config, silent_logger):
    mapping = {f"frame_{i:08d}.png": i for i in range(1, 4)}
    step = Step04Upscale(
        backend_obj=FakeUpscaler(), frame_io=FakeIO(mapping),
        tile=0, output_format="jpg", logger=silent_logger,
    )
    result = step.run(config)
    assert result.tile_used == 0
    assert result.frame_pattern == "frame_8k_%08d.jpg"


def test_full_run_custom_target(config, silent_logger):
    config.target_width, config.target_height = 3840, 2160
    mapping = {f"frame_{i:08d}.png": i for i in range(1, 4)}
    step = Step04Upscale(
        backend_obj=FakeUpscaler(), frame_io=FakeIO(mapping), logger=silent_logger
    )
    assert step.run(config).target_size == (3840, 2160)


def test_full_run_cleans_stale(config, silent_logger):
    mapping = {f"frame_{i:08d}.png": i for i in range(1, 4)}
    config.upscaled_8k.mkdir(parents=True, exist_ok=True)
    stale = config.upscaled_8k / "frame_8k_99999999.png"
    stale.write_bytes(b"old")
    step = Step04Upscale(
        backend_obj=FakeUpscaler(), frame_io=FakeIO(mapping), logger=silent_logger
    )
    step.run(config)
    assert not stale.exists()


def test_full_run_no_sources(tmp_path, silent_logger):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    cfg = PipelineConfig(
        input_video_path=video, final_output_path=tmp_path / "o.mp4",
        workspace_root=tmp_path / "ws",
    )
    cfg.interpolated_720p.mkdir(parents=True, exist_ok=True)
    step = Step04Upscale(
        backend_obj=FakeUpscaler(), frame_io=FakeIO({}), logger=silent_logger
    )
    with pytest.raises(UpscaleError, match="No interpolated frames"):
        step.run(cfg)


def test_bad_params_raise(silent_logger):
    with pytest.raises(ValueError):
        Step04Upscale(output_format="bmp", logger=silent_logger)
    with pytest.raises(ValueError):
        Step04Upscale(tile=-1, logger=silent_logger)


# -- Backends ------------------------------------------------------------------------------
def test_create_upscaler_unknown(silent_logger):
    with pytest.raises(EsrganInferenceError):
        create_upscaler("unknown-thing", logger=silent_logger)


def test_torch_backend_requires_torch(silent_logger):
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch installed -- missing-torch path not testable")
    from pipeline.esrgan.backends import TorchESRGANBackend

    with pytest.raises(EsrganInferenceError, match="PyTorch"):
        TorchESRGANBackend(logger=silent_logger).load()


def test_torch_backend_validates_model_and_device(silent_logger):
    from pipeline.esrgan.backends import TorchESRGANBackend

    with pytest.raises(EsrganInferenceError, match="Unknown model"):
        TorchESRGANBackend(model="x8magic", logger=silent_logger)

    class NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    with pytest.raises(EsrganInferenceError, match="CUDA was requested"):
        TorchESRGANBackend(device="cuda", logger=silent_logger)._resolve_device(NoCuda())
    with pytest.raises(EsrganInferenceError, match="Unknown device"):
        TorchESRGANBackend(device="tpu", logger=silent_logger)._resolve_device(NoCuda())


def test_resize_backend_output_size(silent_logger):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from pipeline.esrgan.backends import ResizeBackend

    backend = ResizeBackend(target_size=(320, 160), logger=silent_logger)
    backend.load()
    out = backend.upscale(np.zeros((40, 80, 3), dtype=np.uint8))
    assert np.asarray(out).shape == (160, 320, 3)


# -- Weights ---------------------------------------------------------------------------------
def test_model_table_sanity():
    assert set(ESRGAN_MODELS) == {"x4plus", "x4plus-anime"}
    for name, info in ESRGAN_MODELS.items():
        assert info.url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/")
        assert info.url.endswith(".pth")
        assert info.scale == 4
    assert ESRGAN_MODELS["x4plus"].num_block == 23
    assert ESRGAN_MODELS["x4plus-anime"].num_block == 6


def test_ensure_weights_unknown_model(tmp_path, silent_logger):
    with pytest.raises(EsrganWeightsError, match="Unknown Real-ESRGAN model"):
        ensure_esrgan_weights(model="x8", weights_root=tmp_path, logger=silent_logger)


def test_ensure_weights_override_and_cache(tmp_path, silent_logger):
    f = tmp_path / "mine.pth"
    f.write_bytes(b"x" * 10)
    assert ensure_esrgan_weights(weights_root=tmp_path, weights_override=f,
                                 logger=silent_logger) == f
    with pytest.raises(EsrganWeightsError, match="does not exist"):
        ensure_esrgan_weights(weights_root=tmp_path,
                              weights_override=tmp_path / "nope.pth",
                              logger=silent_logger)
    cached = cache_path(tmp_path, "x4plus")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"y" * 2_000_000)
    assert ensure_esrgan_weights(model="x4plus", weights_root=tmp_path,
                                 logger=silent_logger) == cached


def test_validate_rejects_small_and_bad_magic(tmp_path):
    info = ESRGAN_MODELS["x4plus"]
    small = tmp_path / "small.pth"
    small.write_bytes(b"PK" + b"0" * 100)
    with pytest.raises(EsrganWeightsError, match="suspiciously small"):
        _validate(small, info)
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"<!DO" + b"0" * 2_000_000)
    with pytest.raises(EsrganWeightsError, match="not a torch weights file"):
        _validate(bad, info)


# -- Vendored API surface (AST only — no torch import) ----------------------
def test_vendored_rrdbnet_api_surface():
    repo = Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / "pipeline/esrgan/vendor/upstream/rrdbnet.py").read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert {"ResidualDenseBlock", "RRDB", "RRDBNet"} <= classes
    net = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "RRDBNet")
    methods = {n.name for n in net.body if isinstance(n, ast.FunctionDef)}
    assert {"__init__", "forward"} <= methods

    helpers = ast.parse(
        (repo / "pipeline/esrgan/vendor/upstream/arch_util.py").read_text()
    )
    funcs = {n.name for n in ast.walk(helpers) if isinstance(n, ast.FunctionDef)}
    assert {"default_init_weights", "make_layer", "pixel_unshuffle"} <= funcs

    # No leftover dependency on the basicsr package in vendored code.
    for path in (repo / "pipeline/esrgan/vendor/upstream").glob("*.py"):
        text = path.read_text()
        assert "basicsr" not in text, path
        assert "ARCH_REGISTRY" not in text or "dropped" in text, path


def test_frame_io_shim_identity():
    import pipeline.frame_io as shared
    import pipeline.rife.io as shim

    assert shim.FrameIO is shared.FrameIO
    assert shim.Cv2FrameIO is shared.Cv2FrameIO
