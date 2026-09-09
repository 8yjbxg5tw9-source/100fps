"""RIFE interpolation toolkit for Step 3 (torch-free imports only).

- :mod:`pipeline.rife.vendor` — vendored official RIFE v4 network code (MIT).
- :mod:`pipeline.rife.weights` — weight resolution + Drive auto-download.
- :mod:`pipeline.rife.backends` — ``rife`` (PyTorch AI) / ``blend`` (smoke).
- :mod:`pipeline.rife.io` — frame image I/O.

Importing this package never requires torch/numpy/cv2; heavy dependencies
are loaded lazily inside the backend / FrameIO implementations.
"""

from pipeline.rife.backends import (
    BlendBackend,
    RifeBackend,
    TorchRifeBackend,
    create_backend,
)
from pipeline.rife.io import Cv2FrameIO, FrameIO
from pipeline.rife.weights import (
    DEFAULT_RIFE_VERSION,
    RIFE_VERSIONS,
    ensure_weights,
)

__all__ = [
    "BlendBackend",
    "RifeBackend",
    "TorchRifeBackend",
    "create_backend",
    "Cv2FrameIO",
    "FrameIO",
    "DEFAULT_RIFE_VERSION",
    "RIFE_VERSIONS",
    "ensure_weights",
]
