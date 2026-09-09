"""ONNX Runtime backend for Step 4 (Step 9 §2 — no PyTorch required).

:class:`OnnxESRGANBackend` runs an exported ``.onnx`` Real-ESRGAN graph with
`onnxruntime <https://onnxruntime.ai/>`_ instead of PyTorch:

- provider priority **TensorRT → CUDA → CPU** (whatever the installed
  onnxruntime build offers — TensorRT acceleration is automatic when its
  execution provider exists, with graceful fallback otherwise);
- the same halo tiling math as the torch path, via the numpy reference
  :func:`pipeline.esrgan.tiling.upscale_tiled_numpy` (tiled == direct);
- auto-selected by :func:`create_upscaler` whenever the weights file ends in
  ``.onnx`` (or explicitly with backend ``"onnx"``).

Export a checkpoint first (needs torch, once, on any machine)::

    python -m pipeline.esrgan.onnx_backend --weights RealESRGAN_x4plus.pth \\
        --model x4plus --out RealESRGAN_x4plus.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pipeline.esrgan.backends import EsrganBackend
from pipeline.esrgan.tiling import upscale_tiled_numpy
from pipeline.esrgan.weights import DEFAULT_ESRGAN_MODEL, ESRGAN_MODELS
from pipeline.exceptions import EsrganInferenceError
from pipeline.logger import get_logger

#: Provider preference: fastest first; filtered by availability at load().
PREFERRED_PROVIDERS = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def pick_providers(ort: object, requested: Optional[Sequence[str]] = None) -> List[str]:
    """Order providers: explicit request, else fastest-available first."""
    available = list(ort.get_available_providers())  # type: ignore[attr-defined]
    if requested is not None:
        missing = [p for p in requested if p not in available]
        if missing:
            raise EsrganInferenceError(
                f"Requested ONNX provider(s) {missing} are not available "
                f"(available: {available})."
            )
        return list(requested)
    picked = [p for p in PREFERRED_PROVIDERS if p in available]
    if not picked:
        raise EsrganInferenceError(
            "onnxruntime reports no execution providers — reinstall it."
        )
    return picked


class OnnxESRGANBackend(EsrganBackend):
    """Real-ESRGAN x4 inference via ONNX Runtime (torch-free)."""

    name = "onnx"

    def __init__(
        self,
        weights: str | Path,
        model: str = DEFAULT_ESRGAN_MODEL,
        tile: int = 0,  # 0 = whole-image inference, else numpy halo tiling
        tile_pad: int = 10,
        target_size: Tuple[int, int] = (7680, 4320),  # (width, height)
        providers: Optional[Sequence[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if model not in ESRGAN_MODELS:
            raise EsrganInferenceError(
                f"Unknown model {model!r}. Known: {sorted(ESRGAN_MODELS)}."
            )
        if tile < 0 or tile_pad < 0:
            raise ValueError("tile/tile_pad must be >= 0.")
        self.model_name = model
        self.info = ESRGAN_MODELS[model]
        self.weights_path = Path(weights)
        self.tile = tile
        self.tile_pad = tile_pad
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.providers_request = list(providers) if providers else None
        self.log = logger or get_logger(__name__)
        self.providers: List[str] = []
        self._session: Optional[object] = None
        self._input_name: Optional[str] = None
        self._io_checked = False

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EsrganInferenceError(
                "The 'onnx' backend needs onnxruntime, which is not installed. "
                "Install it ('pip install onnxruntime' — or onnxruntime-gpu "
                "for CUDA/TensorRT), or use --upscale-backend esrgan "
                "(PyTorch) instead."
            ) from exc
        if self.weights_path.suffix.lower() != ".onnx":
            raise EsrganInferenceError(
                f"ONNX backend needs a .onnx file, got {self.weights_path}. "
                f"Export one: python -m pipeline.esrgan.onnx_backend "
                f"--weights model.pth --model {self.model_name} --out model.onnx"
            )
        if not self.weights_path.is_file():
            raise EsrganInferenceError(
                f"ONNX weights not found: {self.weights_path}"
            )
        self.providers = pick_providers(ort, self.providers_request)
        self.log.info(
            "Loading ONNX Real-ESRGAN %s (%s, providers=%s, tile=%s) ...",
            self.model_name, self.weights_path.name, self.providers,
            self.tile if self.tile > 0 else "off",
        )
        try:
            self._session = ort.InferenceSession(
                str(self.weights_path), providers=self.providers
            )
        except Exception as exc:
            raise EsrganInferenceError(
                f"Could not load ONNX model {self.weights_path}: {exc}"
            ) from exc
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise EsrganInferenceError(
                f"ONNX model must take a single NCHW input, got "
                f"{[(i.name, i.shape) for i in inputs]}."
            )
        channels = inputs[0].shape[1]
        if isinstance(channels, int) and channels != 3:
            raise EsrganInferenceError(
                f"ONNX model must take 3-channel RGB input, got shape "
                f"{inputs[0].shape}."
            )
        self._input_name = inputs[0].name
        self.log.info(
            "ONNX model ready (input %s %s, provider %s).",
            self._input_name, inputs[0].shape, self.providers[0],
        )

    # -- inference ----------------------------------------------------------
    def upscale(self, frame: object) -> object:
        if self._session is None or self._input_name is None:
            raise EsrganInferenceError("Backend not loaded -- call load() first.")
        import numpy as np

        img = np.asarray(frame, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise EsrganInferenceError(
                f"Expected an RGB (H, W, 3) frame, got shape {img.shape}."
            )
        rgb = img.astype(np.float32) / 255.0  # HWC float, like the torch path
        if self.tile > 0:
            up = upscale_tiled_numpy(
                rgb, self.info.scale, self.tile, self.tile_pad, self._run_tile
            )
        else:
            nchw = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
            up_nchw = self._run_nchw(nchw)
            up = np.transpose(up_nchw[0], (1, 2, 0))
        return self._to_target_size(np.asarray(up, dtype=np.float32))

    def _run_tile(self, crop_hwc: object) -> object:
        import numpy as np

        crop = np.ascontiguousarray(np.asarray(crop_hwc, dtype=np.float32))
        nchw = np.ascontiguousarray(crop.transpose(2, 0, 1))[None]
        out = self._run_nchw(nchw)
        return np.transpose(out[0], (1, 2, 0))

    def _run_nchw(self, nchw: object) -> object:
        import numpy as np

        out = self._session.run(None, {self._input_name: nchw})[0]  # type: ignore[union-attr]
        out = np.asarray(out)
        if not self._io_checked:
            self._io_checked = True
            _, _, in_h, in_w = np.shape(nchw)
            if out.shape[2] != in_h * self.info.scale or out.shape[3] != in_w * self.info.scale:
                raise EsrganInferenceError(
                    f"ONNX model output {out.shape} does not match the "
                    f"{self.model_name} x{self.info.scale} architecture for "
                    f"input {np.shape(nchw)} — wrong model file?"
                )
        return out

    def _to_target_size(self, rgb: object) -> object:
        import numpy as np

        rgb = np.asarray(rgb)
        target_w, target_h = self.target_size
        if (rgb.shape[1], rgb.shape[0]) != (target_w, target_h):
            try:
                import cv2
            except ImportError as exc:
                raise EsrganInferenceError(
                    "Final resize needs opencv-python ('pip install opencv-python')."
                ) from exc
            rgb = cv2.resize(
                rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )
        return (np.clip(rgb, 0, 1) * 255.0).round().astype(np.uint8)

    # -- memory -------------------------------------------------------------
    def empty_cache(self) -> None:
        pass  # ORT manages its own arenas; nothing to flush per frame.

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        self._io_checked = False
        self.log.info("ONNX backend unloaded.")


# ---------------------------------------------------------------------------
# Export: .pth checkpoint -> .onnx graph (needs torch, run once per model)
# ---------------------------------------------------------------------------
def export_esrgan_onnx(
    weights: str | Path,
    model: str = DEFAULT_ESRGAN_MODEL,
    output: Optional[str | Path] = None,
    opset: int = 17,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Export a Real-ESRGAN checkpoint to ONNX (dynamic H/W, fixed NCHW)."""
    log = logger or get_logger(__name__)
    try:
        import torch
    except ImportError as exc:
        raise EsrganInferenceError(
            "ONNX export needs PyTorch ('pip install torch'). Export once on "
            "any machine, then run the .onnx anywhere with onnxruntime."
        ) from exc
    if model not in ESRGAN_MODELS:
        raise EsrganInferenceError(
            f"Unknown model {model!r}. Known: {sorted(ESRGAN_MODELS)}."
        )
    info = ESRGAN_MODELS[model]
    weights_path = Path(weights)
    if weights_path.suffix.lower() == ".onnx":
        raise EsrganInferenceError(
            f"{weights_path} is already an ONNX model — nothing to export."
        )
    if not weights_path.is_file():
        raise EsrganInferenceError(f"Weights not found: {weights_path}")
    out_path = Path(output) if output else weights_path.with_suffix(".onnx")

    from pipeline.esrgan.vendor.upstream.rrdbnet import RRDBNet

    net = RRDBNet(
        num_in_ch=3, num_out_ch=3, scale=info.scale,
        num_feat=info.num_feat, num_block=info.num_block,
        num_grow_ch=info.num_grow_ch,
    )
    state = torch.load(str(weights_path), map_location="cpu")
    key = "params_ema" if isinstance(state, dict) and "params_ema" in state else "params"
    params = state[key] if isinstance(state, dict) and key in state else state
    net.load_state_dict(params, strict=True)
    net.eval()
    dummy = torch.zeros(1, 3, 64, 64)
    log.info("Exporting %s -> %s (opset %d) ...", weights_path.name, out_path, opset)
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"},
                      "output": {2: "height_out", 3: "width_out"}},
        opset_version=opset, do_constant_folding=True,
    )
    log.info("ONNX model written: %s", out_path)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Real-ESRGAN .pth checkpoint to .onnx (Step 9)."
    )
    parser.add_argument("--weights", required=True, help="Input .pth checkpoint.")
    parser.add_argument("--model", default=DEFAULT_ESRGAN_MODEL,
                        choices=sorted(ESRGAN_MODELS))
    parser.add_argument("--out", default=None, help="Output .onnx path.")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        export_esrgan_onnx(args.weights, args.model, args.out, args.opset)
    except EsrganInferenceError as exc:
        print(f"onnx_backend: error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
