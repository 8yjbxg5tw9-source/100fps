# Vendored super-resolution code — provenance & patch notes

Step 4 upscales with the official **Real-ESRGAN x4 RRDBNet** design. The
network itself lives in BasicSR, the inference/tiling procedure in
Real-ESRGAN — both are vendored below so Step 4 has zero extra dependencies.

## Sources

| Component | Source | License |
|-----------|--------|---------|
| `RRDBNet` + arch helpers | [BasicSR](https://github.com/XPixelGroup/BasicSR), `basicsr/archs/{rrdbnet_arch,arch_util}.py` | **Apache-2.0** © XPixelGroup — `LICENSE-BasicSR` |
| Tiling / pre-pad / outscale procedure | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), `realesrgan/utils.py` (`RealESRGANer`) | **BSD-3-Clause** © Xintao Wang — `LICENSE-RealESRGAN` |

## What is vendored (and why)

| File | Role | Fidelity |
|------|------|----------|
| `upstream/rrdbnet.py` | `RRDBNet`, `RRDB`, `ResidualDenseBlock` | Byte-identical class bodies, registry decorator dropped (see patch) |
| `upstream/arch_util.py` | `default_init_weights`, `make_layer`, `pixel_unshuffle` | Verbatim copies of the 3 helpers RRDBNet needs (rest of upstream `arch_util.py` — DCN, flow_warp, … — is unused for inference and not vendored) |

The `RealESRGANer` *procedure* (pre-pad → tiled/direct forward → post-pad-cut →
Lanczos outscale) is **not** copied as a file: it is re-implemented in
`pipeline/esrgan/backends.py` (`TorchESRGANBackend`) because upstream mixes in
BGR/RGBA/16-bit branches, prints, and its own downloader. The tile-index math
is ported line-for-line into the torch-free `pipeline/esrgan/tiling.py`
(`plan_tiles`), which the backend consumes — so the exact geometry upstream
uses is covered by unit tests (`tests/test_step04_upscale.py`).

## Patches applied to vendored files

1. `rrdbnet.py`: dropped `from basicsr.utils.registry import ARCH_REGISTRY`
   and the `@ARCH_REGISTRY.register()` decorator (inference never queries the
   registry); `from .arch_util import …` is now relative. Verified with
   `diff` that every class body is otherwise identical to upstream.
2. `arch_util.py`: the 3 helper functions are verbatim; module imports were
   trimmed to what they use (`torch`, `nn`, `init`, `_BatchNorm`) — upstream
   additionally imports `torchvision`, DCN ops and loggers for helpers we
   don't vendor.

## Weight compatibility

This code implements **RRDBNet x4** (`num_feat=64, num_grow_ch=32`,
`num_block` 23 for `x4plus` / 6 for `x4plus-anime`). Compatible weights are
the official `.pth` files (state dict under `params_ema`, else `params` —
same preference rule as upstream `RealESRGANer`). The loader uses
`strict=True`, so any architecture mismatch fails loudly.
