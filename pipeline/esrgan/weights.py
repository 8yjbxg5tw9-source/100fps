"""Real-ESRGAN weight resolution: explicit path, cache, or GitHub download.

Resolution order used by :func:`ensure_esrgan_weights`:

1. Explicit ``--esrgan-weights`` path (a ``.pth`` file).
2. Project cache ``weights/esrgan/<filename>.pth`` (reused across runs).
3. Automatic download from the official v0.1.0 GitHub release (direct
   ``https://`` links — plain redirect-following download, no tokens needed).

Downloads are sanity-checked (size + ``PK`` zip magic of ``torch.save`` files)
and the backend performs a ``strict=True`` state-dict load, so truncated or
mismatched files fail loudly instead of producing silent garbage.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pipeline.exceptions import EsrganWeightsError
from pipeline.logger import get_logger

WEIGHTS_DIRNAME = "weights"
ESRGAN_WEIGHTS_SUBDIR = "esrgan"
_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0"


@dataclass(frozen=True)
class EsrganModelInfo:
    url: str
    filename: str
    scale: int
    num_feat: int
    num_block: int
    num_grow_ch: int
    description: str


#: Official weights + the exact RRDBNet arch params each file needs
#: (from upstream ``inference_realesrgan.py``).
ESRGAN_MODELS: Dict[str, EsrganModelInfo] = {
    "x4plus": EsrganModelInfo(
        url=f"{_RELEASE}/RealESRGAN_x4plus.pth",
        filename="RealESRGAN_x4plus.pth",
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        description="General photo/video x4 (default).",
    ),
    "x4plus-anime": EsrganModelInfo(
        url=f"{_RELEASE}/RealESRGAN_x4plus_anime_6B.pth",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        scale=4,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        description="Anime/illustration x4 (6-block lite arch).",
    ),
}

DEFAULT_ESRGAN_MODEL = "x4plus"
_MIN_WEIGHT_BYTES = 1_000_000  # anything smaller is an error page / truncation


def cache_path(weights_root: str | Path, model: str) -> Path:
    return Path(weights_root) / ESRGAN_WEIGHTS_SUBDIR / ESRGAN_MODELS[model].filename


def ensure_esrgan_weights(
    model: str = DEFAULT_ESRGAN_MODEL,
    weights_root: str | Path = WEIGHTS_DIRNAME,
    weights_override: Optional[str | Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Return a ready-to-load ``.pth`` path (download if needed)."""
    log = logger or get_logger(__name__)

    if weights_override is not None:
        override = Path(weights_override).expanduser()
        if override.is_file():
            log.info("Using explicit Real-ESRGAN weights: %s", override)
            return override
        raise EsrganWeightsError(f"--esrgan-weights path does not exist: {override}")

    if model not in ESRGAN_MODELS:
        raise EsrganWeightsError(
            f"Unknown Real-ESRGAN model {model!r}. Known: {sorted(ESRGAN_MODELS)} "
            f"(or pass --esrgan-weights with a local file)."
        )
    info = ESRGAN_MODELS[model]
    cached = cache_path(weights_root, model)
    if cached.is_file() and cached.stat().st_size > _MIN_WEIGHT_BYTES:
        log.info("Using cached Real-ESRGAN %s weights: %s", model, cached)
        return cached

    log.info("Downloading Real-ESRGAN %s weights (%s) ...", model, info.description)
    cached.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download(info.url, cached, log, desc=info.filename)
        _validate(cached, info)
    except EsrganWeightsError:
        _silent_unlink(cached)
        raise
    except Exception as exc:  # noqa: BLE001 - wrap transport errors clearly
        _silent_unlink(cached)
        raise EsrganWeightsError(
            f"Failed to download Real-ESRGAN {model} weights ({exc}). "
            f"Download manually from {info.url} and pass the file via "
            f"--esrgan-weights."
        ) from exc
    log.info("Real-ESRGAN %s weights ready: %s", model, cached)
    return cached


def _download(url: str, dest: Path, log: logging.Logger, desc: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (100fps-pipeline)"})
    response = urllib.request.urlopen(req, timeout=60)
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total and total.isdigit() else None
    try:
        from tqdm import tqdm

        bar = tqdm(total=total_bytes, desc=desc, unit="B", unit_scale=True)
    except ImportError:
        log.warning("tqdm not installed -- download progress hidden.")
        bar = None
    try:
        with open(dest, "wb") as fh:
            if bar is not None:
                with bar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        bar.update(len(chunk))
            else:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
    finally:
        try:
            response.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


def _validate(path: Path, info: EsrganModelInfo) -> None:
    size = path.stat().st_size
    if size < _MIN_WEIGHT_BYTES:
        raise EsrganWeightsError(
            f"Downloaded {info.filename} is suspiciously small ({size} bytes) "
            f"-- likely an error page. URL: {info.url}"
        )
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] != b"PK":  # torch.save() files are zip containers
        raise EsrganWeightsError(
            f"Downloaded {info.filename} is not a torch weights file "
            f"(bad magic {magic!r}). URL: {info.url}"
        )


def _silent_unlink(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
