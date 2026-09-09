"""Background-thread frame writer (I/O never blocks the GPU).

Pattern ported from upstream ``IOConsumer`` (Real-ESRGAN ``realesrgan/utils.py``):
a single worker thread drains a bounded FIFO queue, so ``cv2.imwrite`` latency
(large 8K files!) overlaps with the next frame's inference. Order is preserved
and the first worker error is re-raised on :meth:`close` (fail-loud, no silent
frame loss).
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Optional

from pipeline.logger import get_logger

_SENTINEL: Any = object()


class AsyncFrameWriter:
    """FIFO background writer around any :class:`FrameIO` implementation."""

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
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._count = 0
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def start(self) -> "AsyncFrameWriter":
        if self._started:
            return self
        self._thread = threading.Thread(
            target=self._worker, name="frame-writer", daemon=True
        )
        self._thread.start()
        self._started = True
        return self

    def submit(self, path: Path, frame: Any) -> None:
        """Enqueue one write (blocks when the queue is full: backpressure)."""
        if not self._started:
            raise RuntimeError("AsyncFrameWriter.submit() before start().")
        if self._closed:
            raise RuntimeError("AsyncFrameWriter.submit() after close().")
        self._raise_if_failed()
        self._queue.put((path, frame))

    def close(self) -> int:
        """Flush, stop the worker, re-raise any worker error; return count."""
        if not self._started or self._closed:
            return self.count
        self._closed = True
        # Unblock the worker even if a previous submitter is still waiting:
        # the queue always has room for the small sentinel dance below because
        # submit() callers finish once the worker drains one item.
        self._queue.put(_SENTINEL)
        assert self._thread is not None
        self._thread.join()
        self._raise_if_failed()
        return self.count

    def __enter__(self) -> "AsyncFrameWriter":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- internals ------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                path, frame = item
                self.frame_io.write(path, frame)
                with self._lock:
                    self._count += 1
            except BaseException as exc:  # noqa: BLE001 - captured, re-raised on close()
                with self._lock:
                    if self._error is None:
                        self._error = exc
                self.log.error("Frame writer failed on %s: %s", item[0] if item is not _SENTINEL else "?", exc)
                return
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"Background frame writer failed: {error}") from error
