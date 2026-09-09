"""Real-ESRGAN super-resolution toolkit for Step 4 (torch-free imports).

- :mod:`pipeline.esrgan.vendor` — vendored RRDBNet + arch helpers.
- :mod:`pipeline.esrgan.tiling` — torch-free tile geometry (upstream math).
- :mod:`pipeline.esrgan.weights` — weight resolution + GitHub auto-download.
- :mod:`pipeline.esrgan.backends` — ``esrgan`` (PyTorch AI) / ``resize``.
- :mod:`pipeline.esrgan.writer` — background-thread async frame writer.

Importing this package never requires torch/numpy/cv2; heavy dependencies
are loaded lazily inside the backend / FrameIO implementations.
"""

from pipeline.esrgan.backends import (
    EsrganBackend,
    ResizeBackend,
    TorchESRGANBackend,
    create_upscaler,
)
from pipeline.esrgan.tiling import TilePlan, plan_tiles, upscale_tiled_numpy
from pipeline.esrgan.weights import (
    DEFAULT_ESRGAN_MODEL,
    ESRGAN_MODELS,
    ensure_esrgan_weights,
)
from pipeline.esrgan.writer import AsyncFrameWriter

__all__ = [
    "EsrganBackend",
    "ResizeBackend",
    "TorchESRGANBackend",
    "create_upscaler",
    "TilePlan",
    "plan_tiles",
    "upscale_tiled_numpy",
    "DEFAULT_ESRGAN_MODEL",
    "ESRGAN_MODELS",
    "ensure_esrgan_weights",
    "AsyncFrameWriter",
]
