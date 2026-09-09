"""Background-thread frame writer (I/O never blocks the GPU).

Pattern ported from upstream ``IOConsumer`` (Real-ESRGAN ``realesrgan/utils.py``):
a single worker thread drains a bounded FIFO queue, so ``cv2.imwrite`` latency
(large 8K files!) overlaps with the next frame's inference. Order is preserved
and the first worker error is re-raised on :meth:`close` (fail-loud, no silent
frame loss).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pipeline.logger import get_logger
from pipeline.perf import RingBufferWriter


class AsyncFrameWriter:
    """FIFO background writer around any :class:`FrameIO` implementation.

    Thin Step-4-compatible façade over :class:`RingBufferWriter` (Step 9):
    same constructor, same fail-loud semantics, shared and hardened worker.
    """

    def __init__(
        self,
        frame_io: Any,
        max_queue: int = 8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}.")
        self.frame_io = frame_io
        self.max_queue = max_queue
        self.log = logger or get_logger(__name__)
        self._inner = RingBufferWriter(
            frame_io.write, capacity=max_queue, logger=self.log,
            name="frame-writer",
        )

    @property
    def count(self) -> int:
        return self._inner.count

    def start(self) -> "AsyncFrameWriter":
        self._inner.start()
        return self

    def submit(self, path: Path, frame: Any) -> None:
        """Enqueue one write (blocks when the queue is full: backpressure)."""
        try:
            self._inner.submit(path, frame)
        except RuntimeError as exc:
            # Preserve the historical error wording for callers/tests.
            message = str(exc).replace("frame-writer", "AsyncFrameWriter")
            raise RuntimeError(message) from exc.__cause__

    def close(self) -> int:
        """Flush, stop the worker, re-raise any worker error; return count."""
        try:
            return self._inner.close()
        except RuntimeError as exc:
            if "Background writer failed" in str(exc):
                raise RuntimeError(
                    str(exc).replace(
                        "Background writer failed",
                        "Background frame writer failed",
                    )
                ) from exc.__cause__
            raise

    def __enter__(self) -> "AsyncFrameWriter":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()
