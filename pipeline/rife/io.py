"""Backward-compatible shim: FrameIO lives in :mod:`pipeline.frame_io` now."""

from __future__ import annotations

from pipeline.frame_io import Cv2FrameIO, FrameIO

__all__ = ["Cv2FrameIO", "FrameIO"]
