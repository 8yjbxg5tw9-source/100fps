"""RIFE weight resolution: local cache, explicit path, or Google Drive download.

Resolution order used by :func:`ensure_weights`:

1. Explicit ``--weights`` path (a ``flownet.pkl`` / ``rife*.pth`` file, or a
   directory containing ``flownet.pkl`` — same layout as upstream ``train_log/``).
2. Project cache ``weights/rife/<version>/flownet.pkl`` (reused across runs).
3. Automatic download from the official/community Google Drive links below.

Drive downloads need no extra dependency (a small ``urllib`` client handles
the virus-scan confirmation token) and work with both single ``.pkl`` files
and ``.zip`` archives (auto-extracted). Every download is sanity-checked
before use: HTML error pages are rejected, and the backend later performs a
strict state-dict load so architecture mismatches fail loudly.
"""

from __future__ import annotations

import http.cookiejar
import logging
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pipeline.exceptions import RifeWeightsError
from pipeline.logger import get_logger
from pipeline.weights_manifest import verify_against_manifest

WEIGHTS_DIRNAME = "weights"
RIFE_WEIGHTS_SUBDIR = "rife"
FLOWNET_FILENAME = "flownet.pkl"


@dataclass(frozen=True)
class RifeVersionInfo:
    drive_id: str
    kind: str  # "pkl" (single torch file) or "zip" (archive to extract)
    description: str


#: Known-good weight sources. Drive IDs were taken from the official
#: ECCV2022-RIFE README (v4 paper model + HD archive) and the
#: REAL-Video-Enhancer wiki (community v4.6 mirror). If a link rots, place
#: weights manually and pass --weights (see module docstring).
RIFE_VERSIONS: Dict[str, RifeVersionInfo] = {
    "4": RifeVersionInfo(
        drive_id="1h42aGYPNJn2q8j_GVkS_yDu__G_UZ2GX",
        kind="pkl",
        description="Official ECCV paper RIFE v4 model (default, matches "
        "the vendored v4 architecture by construction).",
    ),
    "4.6": RifeVersionInfo(
        drive_id="1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE",
        kind="pkl",
        description="Community-mirrored RIFE v4.6 weights (v4 arch; key "
        "compatibility verified with a strict load).",
    ),
    "hd": RifeVersionInfo(
        drive_id="1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_",
        kind="zip",
        description="Official 'pretrained HD models' archive (first "
        "flownet.pkl inside is auto-extracted).",
    ),
}

DEFAULT_RIFE_VERSION = "4"


def cache_path(weights_root: str | Path, version: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "", version) or "unknown"
    return Path(weights_root) / RIFE_WEIGHTS_SUBDIR / f"rife{safe}" / FLOWNET_FILENAME


def ensure_weights(
    version: str = DEFAULT_RIFE_VERSION,
    weights_root: str | Path = WEIGHTS_DIRNAME,
    weights_override: Optional[str | Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Return a ready-to-load ``flownet.pkl`` path (download if needed)."""
    log = logger or get_logger(__name__)

    if weights_override is not None:
        override = Path(weights_override).expanduser()
        if override.is_dir():
            candidate = override / FLOWNET_FILENAME
            if candidate.is_file():
                log.info("Using RIFE weights from directory: %s", candidate)
                return candidate
            raise RifeWeightsError(
                f"--weights directory has no {FLOWNET_FILENAME}: {override}"
            )
        if override.is_file():
            log.info("Using explicit RIFE weights: %s", override)
            return override
        raise RifeWeightsError(f"--weights path does not exist: {override}")

    if version not in RIFE_VERSIONS:
        raise RifeWeightsError(
            f"Unknown RIFE version {version!r}. Known: {sorted(RIFE_VERSIONS)} "
            f"(or pass --weights with a local file)."
        )
    info = RIFE_VERSIONS[version]
    cached = cache_path(weights_root, version)
    if cached.is_file() and cached.stat().st_size > 0:
        if _manifest_ok(cached, weights_root, log):
            log.info("Using cached RIFE v%s weights: %s", version, cached)
            return cached
        # Tampered cache was unlinked inside _manifest_ok — re-download below.

    log.info("Downloading RIFE v%s weights (%s) ...", version, info.description)
    cached.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_drive_file(info.drive_id, cached, log, desc=f"rife{version}")
        _finalize_download(cached, info, log)
        if verify_against_manifest(cached, weights_root) is False:
            raise RifeWeightsError(
                f"Downloaded RIFE v{version} weights failed the SHA-256 "
                f"manifest check — refusing to use them. Refresh the manifest "
                f"with packaging/fetch_weights.py."
            )
    except RifeWeightsError:
        _silent_unlink(cached)
        raise
    except Exception as exc:  # noqa: BLE001 - wrap transport errors clearly
        _silent_unlink(cached)
        raise RifeWeightsError(
            f"Failed to download RIFE v{version} weights ({exc}). "
            f"Download manually from "
            f"https://drive.google.com/file/d/{info.drive_id}/view and pass "
            f"the file via --weights."
        ) from exc
    log.info("RIFE v%s weights ready: %s", version, cached)
    return cached


# -- Google Drive client (no extra deps) ---------------------------------------
def _build_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (100fps-pipeline)")]
    return opener


def _download_drive_file(
    file_id: str, dest: Path, log: logging.Logger, desc: str = "weights"
) -> None:
    opener = _build_opener()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = opener.open(url, timeout=60)
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # Large-file virus-scan confirmation page: extract tokens and retry.
        html = response.read().decode("utf-8", errors="ignore")
        url = _confirmation_url(url, html)
        if url is None:
            raise RifeWeightsError(
                "Google Drive returned an HTML page instead of the weights "
                "(quota exceeded or file removed?). Download manually from "
                f"https://drive.google.com/file/d/{file_id}/view and pass "
                "the file via --weights."
            )
        log.info("Drive virus-scan confirmation required -- retrying ...")
        response = opener.open(url, timeout=60)
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total and total.isdigit() else None
    bar = _progress_bar(total_bytes, desc, log)
    try:
        with open(dest, "wb") as fh, bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))
    finally:
        try:
            response.close()
        except Exception:  # noqa: BLE001 - best effort
            pass


def _confirmation_url(url: str, html: str) -> Optional[str]:
    confirm = re.search(r"confirm=([0-9A-Za-z_\-]+)", html)
    uuid = re.search(r"uuid=([0-9a-f\-]+)", html)
    if not confirm:
        return None
    sep = "&" if "?" in url else "?"
    retry = f"{url}{sep}confirm={confirm.group(1)}"
    if uuid:
        retry += f"&uuid={uuid.group(1)}"
    return retry


def _progress_bar(total: Optional[int], desc: str, log: logging.Logger) -> object:
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="B", unit_scale=True)
    except ImportError:
        log.warning("tqdm not installed -- download progress hidden.")

        class _Null:
            def update(self, n: int) -> None:
                pass

            def __enter__(self) -> "_Null":
                return self

            def __exit__(self, *args: object) -> None:
                pass

        return _Null()


def _manifest_ok(cached: Path, weights_root: str | Path, log: logging.Logger) -> bool:
    """Manifest check on a cache hit (missing manifest = legacy OK)."""
    if verify_against_manifest(cached, weights_root) is False:
        log.warning(
            "Cached %s failed its SHA-256 manifest check (corrupt/tampered?) "
            "-- re-downloading ...",
            cached,
        )
        _silent_unlink(cached)
        return False
    return True


# -- Post-download handling ------------------------------------------------------
def _finalize_download(dest: Path, info: RifeVersionInfo, log: logging.Logger) -> None:
    if dest.stat().st_size == 0:
        raise RifeWeightsError("Downloaded weights file is empty.")
    with open(dest, "rb") as fh:
        magic = fh.read(4)
    if magic.startswith(b"<!DO") or magic.startswith(b"<htm"):
        raise RifeWeightsError(
            "Downloaded an HTML page instead of weights (Drive quota/block?). "
            "Download manually and pass the file via --weights."
        )
    if info.kind == "zip" or (magic[:2] == b"PK" and _is_generic_zip(dest)):
        extracted = _extract_flownet_from_zip(dest, log)
        _silent_unlink(dest)
        shutil.move(str(extracted), str(dest))
    # else: single torch file (also a zip by format) -- keep as is.


def _is_generic_zip(path: Path) -> bool:
    """True for plain archives; False for torch.save() zip containers."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return False
    if "archive/data.pkl" in names:
        return False  # torch.save() format -- not an archive to unpack
    return any(name.lower().endswith(".pkl") for name in names)


def _extract_flownet_from_zip(archive: Path, log: logging.Logger) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="rife_weights_"))
    with zipfile.ZipFile(archive) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
        if not candidates:
            raise RifeWeightsError(
                f"Weights archive {archive} contains no .pkl file."
            )
        # Prefer flownet.pkl, else the largest .pkl (most likely the model).
        candidates.sort(key=lambda n: (FLOWNET_FILENAME not in n.lower(), n))
        chosen = candidates[0]
        log.info("Extracting %s from weights archive ...", chosen)
        zf.extract(chosen, tmpdir)
    return tmpdir / chosen


def _silent_unlink(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
