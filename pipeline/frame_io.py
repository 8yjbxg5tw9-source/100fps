"""Shared frame image I/O (used by Step 3, Step 4, and later steps).

Production uses :class:`Cv2FrameIO` (files <-> RGB ``uint8`` arrays). Unit
tests inject in-memory fakes, so the suite runs without cv2/numpy/torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class FrameIO(ABC):
    """Read/write single frames as RGB ``uint8`` ``(H, W, 3)`` arrays."""

    @abstractmethod
    def read(self, path: Path) -> object:
        """Read an image file into an RGB uint8 array."""

    @abstractmethod
    def write(self, path: Path, frame: object) -> None:
        """Write an RGB uint8 array to an image file."""


class Cv2FrameIO(FrameIO):
    """OpenCV implementation (lazy import with a clear error message)."""

    def _cv2(self):  # noqa: ANN202 - cv2 module
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "This step needs opencv-python for frame I/O. Install it with "
                "'pip install opencv-python' (Step 1 does this automatically "
                "unless --no-auto-install was used)."
            ) from exc
        return cv2

    def read(self, path: Path) -> object:
        cv2 = self._cv2()
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read frame image: {path}")
        return img[:, :, ::-1].copy()  # BGR -> RGB

    def write(self, path: Path, frame: object) -> None:
        import numpy as np

        cv2 = self._cv2()
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"Cannot write frame with shape {arr.shape}: {path}")
        params = []
        if path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 98]
        elif path.suffix.lower() == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast, still lossless
        if not cv2.imwrite(str(path), arr[:, :, ::-1], params):
            raise RuntimeError(f"Could not write frame image: {path}")
