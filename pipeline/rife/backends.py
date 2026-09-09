"""Interpolation backends for Step 3.

- :class:`TorchRifeBackend` (``"rife"``, default) — the real AI path: vendored
  official RIFE v4 network on CUDA with mixed precision (fp16/bf16) +
  ``no_grad``, batched forward passes, 32-px padding (upstream formula),
  a dedicated CUDA stream with non-blocking transfers, and OOM-safe
  batch halving.
- :class:`BlendBackend` (``"blend"``) — plain linear cross-fade. **Not AI**,
  only for smoke-testing the pipeline/orchestration on machines without
  torch/CUDA. Never use for real output.

Backends work on RGB ``uint8`` arrays of shape ``(H, W, 3)`` / ``(B, H, W, 3)``.
``numpy``/``torch`` are imported lazily so this module stays importable on a
bare interpreter (unit tests inject fakes instead).
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pipeline.exceptions import RifeInferenceError
from pipeline.logger import get_logger
from pipeline.perf import resolve_torch_precision
from pipeline.resources import default_weights_root
from pipeline.rife.weights import (
    DEFAULT_RIFE_VERSION,
    WEIGHTS_DIRNAME,
    ensure_weights,
)


class RifeBackend(ABC):
    """Interface every Step 3 backend implements."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load weights / allocate resources (idempotent)."""

    @abstractmethod
    def interpolate_batch(
        self,
        batch0: object,  # (B,H,W,3) RGB uint8 (numpy in practice)
        batch1: object,
        timestep: float = 0.5,
    ) -> object:
        """Return the interpolated middle frame(s) for each pair."""

    def interpolate(self, img0: object, img1: object, timestep: float = 0.5) -> object:
        """Single-pair convenience wrapper around :meth:`interpolate_batch`."""
        import numpy as np

        out = self.interpolate_batch(
            np.asarray(img0)[None], np.asarray(img1)[None], timestep
        )
        return np.asarray(out)[0]

    def empty_cache(self) -> None:
        """Release cached GPU memory (no-op for CPU-only backends)."""

    def unload(self) -> None:
        """Free all resources (called before Step 4 to release VRAM)."""


class TorchRifeBackend(RifeBackend):
    """Official RIFE v4 inference on PyTorch (CUDA fp16 / CPU fp32)."""

    name = "rife"

    def __init__(
        self,
        version: str = DEFAULT_RIFE_VERSION,
        weights: Optional[str | Path] = None,
        weights_root: Optional[str | Path] = None,  # None -> default_weights_root()
        device: Optional[str] = None,  # "cuda" | "cpu" | None (auto)
        fp16: Optional[bool] = None,  # None -> True on CUDA, False on CPU
        precision: Optional[str] = None,  # Step 9: auto|fp32|fp16|bf16 (beats fp16)
        async_transfers: bool = True,  # Step 9: CUDA stream + non-blocking H2D/D2H
        scale_list: Optional[List[float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.version = version
        self.weights_override = weights
        self.weights_root = weights_root
        self.device_request = device
        self.fp16_request = fp16
        self.precision_request = precision
        self.async_transfers = async_transfers
        self.scale_list = list(scale_list) if scale_list else [4.0, 2.0, 1.0]
        self.log = logger or get_logger(__name__)
        self.device: Optional[str] = None
        self.dtype_name: str = "fp32"
        self.fp16: bool = False  # legacy mirror of dtype_name == "fp16"
        self.weights_path: Optional[Path] = None
        self._model: Optional[object] = None
        self._torch: Optional[object] = None
        self._stream: Optional[object] = None  # dedicated CUDA stream (Step 9)

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RifeInferenceError(
                "The 'rife' backend needs PyTorch, which is not installed. "
                "Install it ('pip install torch' — CUDA build from "
                "https://pytorch.org/get-started/locally/ for GPU), or use "
                "--backend blend for a non-AI smoke test."
            ) from exc
        self._torch = torch
        self.device = self._resolve_device(torch)
        self.dtype_name = resolve_torch_precision(
            torch, self.device, self.precision_request, self.fp16_request,
            self.log,
        )
        self.fp16 = self.dtype_name == "fp16"
        self._stream = self._make_stream(torch)
        weights_root = (
            self.weights_root
            if self.weights_root is not None
            else default_weights_root()
        )
        self.weights_path = ensure_weights(
            version=self.version,
            weights_root=weights_root,
            weights_override=self.weights,
            logger=self.log,
        )
        from pipeline.rife.vendor.model import RifeModel

        self.log.info(
            "Loading RIFE v%s on %s (precision=%s, async_transfers=%s) ...",
            self.version, self.device, self.dtype_name,
            self._stream is not None,
        )
        try:
            model = RifeModel(str(self.weights_path)).eval().to(self.device)
        except RuntimeError as exc:
            raise RifeInferenceError(f"RIFE weight load failed: {exc}") from exc
        if self.dtype_name == "fp16":
            model.half()  # mirrors upstream fp16 inference
        elif self.dtype_name == "bf16":
            model.to(torch.bfloat16)
        self._model = model
        self.log.info("RIFE model ready (eval mode, no_grad inference).")

    def _make_stream(self, torch: object) -> Optional[object]:
        """Dedicated compute stream (CUDA + enabled); else None = default."""
        if not self.async_transfers or self.device != "cuda":
            return None
        stream = torch.cuda.Stream()  # type: ignore[attr-defined]
        self.log.info("Async GPU transfers enabled (dedicated CUDA stream).")
        return stream

    def _compute_ctx(self):  # noqa: ANN202 - contextmanager
        """Run a micro-batch on the compute stream (or default stream)."""
        if self._stream is not None and self._torch is not None:
            return self._torch.cuda.stream(self._stream)
        return contextlib.nullcontext()

    def _sync_if_stream(self) -> None:
        if self._stream is not None and self._torch is not None:
            self._torch.cuda.synchronize()

    def _resolve_device(self, torch: object) -> str:
        cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        if self.device_request is None:
            chosen = "cuda" if cuda_available else "cpu"
        elif self.device_request == "cuda":
            if not cuda_available:
                raise RifeInferenceError(
                    "CUDA was requested (--device cuda) but torch reports no "
                    "CUDA device. Install the CUDA-enabled PyTorch build or "
                    "use --device cpu (very slow for 720p->1000fps)."
                )
            chosen = "cuda"
        elif self.device_request == "cpu":
            chosen = "cpu"
        else:
            raise RifeInferenceError(
                f"Unknown device {self.device_request!r}: use 'cuda' or 'cpu'."
            )
        if chosen == "cpu":
            self.log.warning(
                "RIFE on CPU will be extremely slow (minutes per frame pair "
                "at 720p). A CUDA GPU is strongly recommended for Step 3."
            )
        return chosen

    # -- inference ----------------------------------------------------------
    def interpolate_batch(
        self, batch0: object, batch1: object, timestep: float = 0.5
    ) -> object:
        if self._model is None or self._torch is None:
            raise RifeInferenceError("Backend not loaded -- call load() first.")
        import numpy as np

        frames0 = np.asarray(batch0, dtype=np.uint8)
        frames1 = np.asarray(batch1, dtype=np.uint8)
        if frames0.shape != frames1.shape or frames0.ndim != 4:
            raise RifeInferenceError(
                f"Expected two (B,H,W,3) batches, got {frames0.shape} vs "
                f"{frames1.shape}."
            )
        _, h, w, _ = frames0.shape
        ph, pw = _padded_size(h, w)
        torch = self._torch
        with torch.no_grad():  # type: ignore[attr-defined]
            return self._forward_halving_on_oom(
                torch, frames0, frames1, h, w, ph, pw, timestep
            )

    def _forward_halving_on_oom(
        self, torch: object, frames0: object, frames1: object,
        h: int, w: int, ph: int, pw: int, timestep: float,
    ) -> object:
        import numpy as np

        batch_size = int(np.asarray(frames0).shape[0])
        micro = batch_size
        streamed = self._stream is not None
        while True:
            try:
                pending = []
                for start in range(0, batch_size, micro):
                    # H2D copies are non-blocking; on the compute stream they
                    # overlap the previous micro-batch's kernels (Step 9).
                    with self._compute_ctx():
                        t0 = _to_tensor(torch, frames0, start, start + micro, self.dtype_name)  # type: ignore[arg-type]
                        t1 = _to_tensor(torch, frames1, start, start + micro, self.dtype_name)  # type: ignore[arg-type]
                        t0 = _pad(torch, t0, ph, pw, self.device or "cpu")
                        t1 = _pad(torch, t1, ph, pw, self.device or "cpu")
                        mid = self._model.inference(t0, t1, list(self.scale_list), timestep)  # type: ignore[union-attr]
                        mid = mid[:, :, :h, :w]
                        gpu_out = (mid.clamp(0, 1) * 255.0).byte()
                        # D2H copy joins the stream; numpy() below only runs
                        # after the single batch-level synchronisation.
                        out = gpu_out.to("cpu", non_blocking=True) if streamed else gpu_out.cpu()
                        pending.append(out)
                        del t0, t1, mid, gpu_out, out
                self._sync_if_stream()
                outputs = [np.transpose(o.numpy(), (0, 2, 3, 1)) for o in pending]
                return np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or micro <= 1:
                    raise RifeInferenceError(
                        f"RIFE inference failed: {exc} "
                        f"(try a smaller --batch-size)"
                    ) from exc
                micro //= 2
                self.log.warning(
                    "CUDA out of memory -- halving inference batch to %d and retrying ...",
                    micro,
                )
                self.empty_cache()

    # -- memory -------------------------------------------------------------
    def empty_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self) -> None:
        self._model = None
        self._stream = None
        self.empty_cache()
        self.log.info("RIFE backend unloaded (VRAM released for Step 4).")


def _padded_size(h: int, w: int, multiple: int = 32) -> tuple:
    """Upstream padding formula (scale=1.0): round up to 32 px."""
    return ((h - 1) // multiple + 1) * multiple, ((w - 1) // multiple + 1) * multiple


def _to_tensor(torch: object, batch: object, start: int, end: int, dtype_name: str) -> object:
    import numpy as np

    arr = np.asarray(batch)[start:end].transpose(0, 3, 1, 2)  # B,H,W,3 -> B,3,H,W
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0  # type: ignore[attr-defined]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[attr-defined]
    tensor = tensor.to(device, non_blocking=True)
    if dtype_name == "fp16":
        return tensor.half()
    if dtype_name == "bf16":
        return tensor.to(torch.bfloat16)
    return tensor


def _pad(torch: object, tensor: object, ph: int, pw: int, device: str) -> object:
    h, w = tensor.shape[2], tensor.shape[3]  # type: ignore[union-attr]
    padding = (0, pw - w, 0, ph - h)  # left, right, top, bottom
    import torch.nn.functional as F  # noqa: N812 - mirrors upstream pad_image

    return F.pad(tensor, padding)


class BlendBackend(RifeBackend):
    """Linear cross-fade — smoke-test only, explicitly NOT AI interpolation."""

    name = "blend"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.log = logger or get_logger(__name__)
        self.device = "cpu"

    def load(self) -> None:
        self.log.warning(
            "Using 'blend' backend: frames are linear cross-fades, NOT RIFE "
            "optical-flow interpolation. Only for pipeline smoke tests."
        )

    def interpolate_batch(
        self, batch0: object, batch1: object, timestep: float = 0.5
    ) -> object:
        import numpy as np

        a = np.asarray(batch0, dtype=np.float32)
        b = np.asarray(batch1, dtype=np.float32)
        if a.shape != b.shape:
            raise RifeInferenceError(f"Batch shape mismatch: {a.shape} vs {b.shape}.")
        return ((1.0 - timestep) * a + timestep * b).astype(np.uint8)


def create_backend(name: str, logger: Optional[logging.Logger] = None, **kwargs: object) -> RifeBackend:
    """Factory: ``'rife'`` -> TorchRifeBackend, ``'blend'`` -> BlendBackend."""
    if name == TorchRifeBackend.name:
        return TorchRifeBackend(logger=logger, **kwargs)  # type: ignore[arg-type]
    if name == BlendBackend.name:
        if kwargs:
            (logger or get_logger(__name__)).warning(
                "Blend backend ignores extra options: %s", sorted(kwargs)
            )
        return BlendBackend(logger=logger)
    raise RifeInferenceError(
        f"Unknown backend {name!r}. Available: 'rife' (AI, needs torch) "
        f"and 'blend' (non-AI smoke test)."
    )
