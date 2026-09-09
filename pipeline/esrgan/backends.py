"""Super-resolution backends for Step 4.

- :class:`TorchESRGANBackend` (``"esrgan"``, default) — the real AI path: a
  faithful port of upstream ``RealESRGANer`` (pre-pad → tiled/direct RRDBNet
  forward → post-crop → Lanczos outscale) with ``eval`` + ``no_grad`` + CUDA
  fp16, VRAM-safe tiling from :mod:`pipeline.esrgan.tiling`, and OOM-safe
  tile halving.
- :class:`ResizeBackend` (``"resize"``) — plain OpenCV Lanczos upscale.
  **Not AI**, only for smoke-testing the pipeline on machines without
  torch/CUDA. Never use for real output.

Backends map RGB ``uint8`` ``(H, W, 3)`` frames to RGB ``uint8`` frames at the
exact configured target size (default 7680x4320). ``numpy``/``torch``/``cv2``
are imported lazily so this module stays importable on a bare interpreter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from pipeline.esrgan.tiling import plan_tiles
from pipeline.esrgan.weights import (
    DEFAULT_ESRGAN_MODEL,
    ESRGAN_MODELS,
    WEIGHTS_DIRNAME,
    ensure_esrgan_weights,
)
from pipeline.exceptions import EsrganInferenceError
from pipeline.logger import get_logger


class EsrganBackend(ABC):
    """Interface every Step 4 backend implements."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load weights / allocate resources (idempotent)."""

    @abstractmethod
    def upscale(self, frame: object) -> object:
        """Upscale one RGB uint8 frame to the exact target size."""

    def empty_cache(self) -> None:
        """Release cached GPU memory (no-op for CPU-only backends)."""

    def unload(self) -> None:
        """Free all resources (called before Step 5 to release VRAM)."""


class TorchESRGANBackend(EsrganBackend):
    """Official Real-ESRGAN x4 inference on PyTorch (CUDA fp16 / CPU fp32)."""

    name = "esrgan"
    MIN_TILE = 128  # floor for OOM-driven tile halving

    def __init__(
        self,
        model: str = DEFAULT_ESRGAN_MODEL,
        weights: Optional[str | Path] = None,
        weights_root: str | Path = WEIGHTS_DIRNAME,
        device: Optional[str] = None,  # "cuda" | "cpu" | None (auto)
        fp16: Optional[bool] = None,  # None -> True on CUDA, False on CPU
        tile: int = 0,  # 0 = whole-image inference, else tiled (Step 1 value)
        tile_pad: int = 10,  # upstream default halo
        pre_pad: int = 0,  # upstream inference-script default
        target_size: Tuple[int, int] = (7680, 4320),  # (width, height)
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if model not in ESRGAN_MODELS:
            raise EsrganInferenceError(
                f"Unknown model {model!r}. Known: {sorted(ESRGAN_MODELS)}."
            )
        if tile < 0 or tile_pad < 0 or pre_pad < 0:
            raise ValueError("tile/tile_pad/pre_pad must be >= 0.")
        self.model_name = model
        self.info = ESRGAN_MODELS[model]
        self.weights_override = weights
        self.weights_root = weights_root
        self.device_request = device
        self.fp16_request = fp16
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.log = logger or get_logger(__name__)
        self.device: Optional[str] = None
        self.fp16: bool = False
        self.weights_path: Optional[Path] = None
        self._model: Optional[object] = None
        self._torch: Optional[object] = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise EsrganInferenceError(
                "The 'esrgan' backend needs PyTorch, which is not installed. "
                "Install it ('pip install torch' — CUDA build from "
                "https://pytorch.org/get-started/locally/ for GPU), or use "
                "--upscale-backend resize for a non-AI smoke test."
            ) from exc
        self._torch = torch
        self.device = self._resolve_device(torch)
        self.fp16 = self._resolve_fp16()
        self.weights_path = ensure_esrgan_weights(
            model=self.model_name,
            weights_root=self.weights_root,
            weights_override=self.weights,
            logger=self.log,
        )
        from pipeline.esrgan.vendor.upstream.rrdbnet import RRDBNet

        self.log.info(
            "Loading Real-ESRGAN %s on %s (fp16=%s, tile=%s) ...",
            self.model_name, self.device, self.fp16,
            self.tile if self.tile > 0 else "off",
        )
        net = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=self.info.scale,
            num_feat=self.info.num_feat, num_block=self.info.num_block,
            num_grow_ch=self.info.num_grow_ch,
        )
        try:
            state = torch.load(str(self.weights_path), map_location="cpu")
        except Exception as exc:
            raise EsrganInferenceError(
                f"Could not read weights {self.weights_path}: {exc}"
            ) from exc
        # Upstream rule: prefer EMA weights, else plain params, strict load.
        key = "params_ema" if isinstance(state, dict) and "params_ema" in state else "params"
        try:
            params = state[key] if isinstance(state, dict) and key in state else state
            net.load_state_dict(params, strict=True)
        except Exception as exc:
            raise EsrganInferenceError(
                f"Weights {self.weights_path} do not match the {self.model_name} "
                f"architecture (strict load failed): {exc}"
            ) from exc
        net.eval()
        net.to(self.device)
        if self.fp16:
            net.half()
        self._model = net
        self.log.info("Real-ESRGAN model ready (eval mode, no_grad inference).")

    def _resolve_device(self, torch: object) -> str:
        cuda_available = bool(torch.cuda.is_available())  # type: ignore[attr-defined]
        if self.device_request is None:
            chosen = "cuda" if cuda_available else "cpu"
        elif self.device_request == "cuda":
            if not cuda_available:
                raise EsrganInferenceError(
                    "CUDA was requested (--device cuda) but torch reports no "
                    "CUDA device. Install the CUDA-enabled PyTorch build or "
                    "use --device cpu (very slow for 8K upscaling)."
                )
            chosen = "cuda"
        elif self.device_request == "cpu":
            chosen = "cpu"
        else:
            raise EsrganInferenceError(
                f"Unknown device {self.device_request!r}: use 'cuda' or 'cpu'."
            )
        if chosen == "cpu":
            self.log.warning(
                "Real-ESRGAN on CPU will be extremely slow (minutes per frame "
                "at 720p->8K). A CUDA GPU is strongly recommended for Step 4."
            )
        return chosen

    def _resolve_fp16(self) -> bool:
        if self.fp16_request is None:
            return self.device == "cuda"
        if self.fp16_request and self.device == "cpu":
            self.log.warning("fp16 on CPU is unsupported -- falling back to fp32.")
            return False
        return self.fp16_request

    # -- inference (upstream RealESRGANer procedure) ------------------------
    def upscale(self, frame: object) -> object:
        if self._model is None or self._torch is None:
            raise EsrganInferenceError("Backend not loaded -- call load() first.")
        import numpy as np

        img = np.asarray(frame, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise EsrganInferenceError(
                f"Expected an RGB (H, W, 3) frame, got shape {img.shape}."
            )
        torch = self._torch
        with torch.no_grad():  # type: ignore[attr-defined]
            return self._upscale_with_oom_retry(torch, img)

    def _upscale_with_oom_retry(self, torch: object, img: object) -> object:
        while True:
            try:
                tensor = self._pre_process(torch, img)
                if self.tile > 0:
                    out = self._forward_tiled(tensor)
                else:
                    out = self._model(tensor)  # type: ignore[operator]
                out = self._post_process(out)
                return self._to_target_size(out)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise EsrganInferenceError(f"Real-ESRGAN inference failed: {exc}") from exc
                if self.tile > self.MIN_TILE:
                    self.tile //= 2
                    self.log.warning(
                        "CUDA out of memory -- halving tile size to %d and retrying ...",
                        self.tile,
                    )
                    self.empty_cache()
                    continue
                if self.tile == 0:
                    hint = "enable tiling with --upscale-tile (e.g. 256)"
                else:
                    hint = "use a smaller --upscale-tile or --device cpu"
                raise EsrganInferenceError(
                    f"CUDA out of memory during 8K upscale: {exc} ({hint})."
                ) from exc

    def _pre_process(self, torch: object, img: object) -> object:
        import numpy as np

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))  # type: ignore[attr-defined]
        tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)
        if self.fp16:
            tensor = tensor.half()
        if self.pre_pad != 0:  # upstream reflect pre-pad
            import torch.nn.functional as F

            tensor = F.pad(tensor, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        return tensor

    def _forward_tiled(self, img: object) -> object:
        _, _, h, w = img.shape  # type: ignore[union-attr]
        scale = self.info.scale
        out = img.new_zeros((1, 3, h * scale, w * scale))  # type: ignore[union-attr]
        for plan in plan_tiles(h, w, self.tile, self.tile_pad, scale):
            tile_in = img[:, :, plan.in_y0:plan.in_y1, plan.in_x0:plan.in_x1]  # type: ignore[index]
            tile_out = self._model(tile_in)  # type: ignore[operator]
            out[:, :, plan.out_y0:plan.out_y1, plan.out_x0:plan.out_x1] = tile_out[  # type: ignore[index]
                :, :, plan.tile_y0:plan.tile_y1, plan.tile_x0:plan.tile_x1
            ]
        return out

    def _post_process(self, out: object) -> object:
        if self.pre_pad != 0:  # upstream: strip pre-pad * scale
            _, _, h, w = out.shape  # type: ignore[union-attr]
            cut = self.pre_pad * self.info.scale
            out = out[:, :, 0:h - cut, 0:w - cut]  # type: ignore[index]
        return out

    def _to_target_size(self, out: object) -> object:
        import numpy as np

        arr = out.data.squeeze().float().cpu().clamp_(0, 1).numpy()  # type: ignore[union-attr]
        rgb = np.transpose(arr, (1, 2, 0))
        target_w, target_h = self.target_size
        if (rgb.shape[1], rgb.shape[0]) != (target_w, target_h):
            # Upstream outscale step: exact-size Lanczos (4x AI + 1.5x classical
            # for 720p -> 8K).
            try:
                import cv2
            except ImportError as exc:
                raise EsrganInferenceError(
                    "Final resize to 8K needs opencv-python ('pip install opencv-python')."
                ) from exc
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return (rgb * 255.0).round().astype(np.uint8)

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
        self.empty_cache()
        self.log.info("Real-ESRGAN backend unloaded (VRAM released for Step 5).")


class ResizeBackend(EsrganBackend):
    """Plain Lanczos upscale — smoke-test only, explicitly NOT AI."""

    name = "resize"

    def __init__(
        self,
        target_size: Tuple[int, int] = (7680, 4320),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.log = logger or get_logger(__name__)

    def load(self) -> None:
        self.log.warning(
            "Using 'resize' backend: frames are Lanczos-upscaled, NOT "
            "Real-ESRGAN neural upscales. Only for pipeline smoke tests."
        )

    def upscale(self, frame: object) -> object:
        import numpy as np

        try:
            import cv2
        except ImportError as exc:
            raise EsrganInferenceError(
                "The 'resize' backend needs opencv-python ('pip install opencv-python')."
            ) from exc
        img = np.asarray(frame, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise EsrganInferenceError(f"Expected RGB (H, W, 3), got {img.shape}.")
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)


def create_upscaler(
    name: str, logger: Optional[logging.Logger] = None, **kwargs: object
) -> EsrganBackend:
    """Factory: ``'esrgan'`` -> TorchESRGANBackend, ``'resize'`` -> ResizeBackend."""
    if name == TorchESRGANBackend.name:
        return TorchESRGANBackend(logger=logger, **kwargs)  # type: ignore[arg-type]
    if name == ResizeBackend.name:
        allowed = {"target_size"}
        extra = sorted(set(kwargs) - allowed)
        if extra:
            (logger or get_logger(__name__)).warning(
                "Resize backend ignores extra options: %s", extra
            )
        return ResizeBackend(
            logger=logger,
            **{k: v for k, v in kwargs.items() if k in allowed},  # type: ignore[arg-type]
        )
    raise EsrganInferenceError(
        f"Unknown backend {name!r}. Available: 'esrgan' (AI, needs torch) "
        f"and 'resize' (non-AI smoke test)."
    )
