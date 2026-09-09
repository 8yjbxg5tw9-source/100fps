"""Torch-free tiling geometry for VRAM-safe super-resolution.

This is a line-for-line port of the tile-index math in upstream
``RealESRGANer.tile_process`` (Real-ESRGAN ``realesrgan/utils.py``), factored
out so the production torch backend and the unit tests share one geometry:

- each input tile is cropped **with** a ``pad`` halo on every side,
- the model upscales the padded tile by ``scale``,
- the halo (``pad * scale``) is cropped back off,
- the kept region is pasted into the output canvas — every output pixel is
  written exactly once, so no seam/blending artifacts can occur.

Because the halo gives the network full context at tile borders, results are
bit-close to whole-image inference (up to fp rounding in overlap-free math).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List


@dataclass(frozen=True)
class TilePlan:
    """Coordinates for one tile (row-major order, upstream loop order)."""

    # Input crop WITH halo — what the model sees (input pixel coords).
    in_y0: int
    in_y1: int
    in_x0: int
    in_x1: int
    # Crop INSIDE the upscaled tile output — halo removed (tile-output coords).
    tile_y0: int
    tile_y1: int
    tile_x0: int
    tile_x1: int
    # Paste position in the full upscaled canvas (output coords).
    out_y0: int
    out_y1: int
    out_x0: int
    out_x1: int


def plan_tiles(h: int, w: int, tile: int, pad: int, scale: int) -> List[TilePlan]:
    """Compute the tile plan for an ``h x w`` image (upstream algorithm).

    Args:
        h, w: input image size.
        tile: nominal tile size (from Step 1 ``tile_size``); ``<= 0`` selects
            whole-image inference and raises here (callers branch first).
        pad: halo around each tile (upstream ``tile_pad``, default 10).
        scale: model upscaling factor (4 for Real-ESRGAN x4).
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size {h}x{w}.")
    if tile <= 0:
        raise ValueError("tile must be >= 1 for plan_tiles (use direct inference).")
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}.")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}.")

    plans: List[TilePlan] = []
    tiles_x = math.ceil(w / tile)
    tiles_y = math.ceil(h / tile)
    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile
            ofs_y = y * tile
            # Input tile area on total image (halo-free "kept" footprint).
            in_sx, in_ex = ofs_x, min(ofs_x + tile, w)
            in_sy, in_ey = ofs_y, min(ofs_y + tile, h)
            # Same area extended by the halo, clamped to the image.
            in_sx_p, in_ex_p = max(in_sx - pad, 0), min(in_ex + pad, w)
            in_sy_p, in_ey_p = max(in_sy - pad, 0), min(in_ey + pad, h)
            kept_w, kept_h = in_ex - in_sx, in_ey - in_sy
            # Paste position on the upscaled canvas.
            out_sx, out_ex = in_sx * scale, in_ex * scale
            out_sy, out_ey = in_sy * scale, in_ey * scale
            # Crop inside the upscaled (haloed) tile output.
            tile_sx = (in_sx - in_sx_p) * scale
            tile_sy = (in_sy - in_sy_p) * scale
            plans.append(
                TilePlan(
                    in_y0=in_sy_p, in_y1=in_ey_p, in_x0=in_sx_p, in_x1=in_ex_p,
                    tile_y0=tile_sy, tile_y1=tile_sy + kept_h * scale,
                    tile_x0=tile_sx, tile_x1=tile_sx + kept_w * scale,
                    out_y0=out_sy, out_y1=out_ey, out_x0=out_sx, out_x1=out_ex,
                )
            )
    return plans


def upscale_tiled_numpy(
    image: object,
    scale: int,
    tile: int,
    pad: int,
    upscale_fn: Callable[[object], object],
) -> object:
    """Reference numpy implementation of tiled upscale (tests + fallback).

    ``upscale_fn`` maps an ``(h, w, c)`` crop to ``(h*scale, w*scale, c)``.
    Used by unit tests to prove the seam math (tiled == direct), and
    available to any numpy-based backend.
    """
    import numpy as np

    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"Expected (H, W, C) image, got shape {img.shape}.")
    h, w, c = img.shape
    canvas = np.zeros((h * scale, w * scale, c), dtype=img.dtype)
    for plan in plan_tiles(h, w, tile, pad, scale):
        crop = img[plan.in_y0:plan.in_y1, plan.in_x0:plan.in_x1]
        up = np.asarray(upscale_fn(crop))
        expected = (
            (plan.in_y1 - plan.in_y0) * scale,
            (plan.in_x1 - plan.in_x0) * scale,
            c,
        )
        if up.shape != expected:
            raise ValueError(
                f"upscale_fn returned {up.shape}, expected {expected}."
            )
        kept = up[plan.tile_y0:plan.tile_y1, plan.tile_x0:plan.tile_x1]
        canvas[plan.out_y0:plan.out_y1, plan.out_x0:plan.out_x1] = kept
    return canvas
