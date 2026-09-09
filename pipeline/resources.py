"""Frozen/dev resource resolution (Step 10 — standalone EXE/App).

When the pipeline runs from source, resources (weights, ``bin/``) resolve
relative to the working tree. When frozen with PyInstaller (``sys.frozen``),
read-only data lives in the bundle (``sys._MEIPASS``) while writable data
(models, workspaces) lives next to the executable — ``_MEIPASS`` is a
per-run temp dir in onefile mode, so weights must NEVER resolve there.

Environment overrides (power users, CI)::

    100FPS_WEIGHTS   explicit weights root (beats every default)
    100FPS_FFMPEG    explicit ffmpeg binary (beats PATH lookup)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

WEIGHTS_ENV_VAR = "100FPS_WEIGHTS"
FFMPEG_ENV_VAR = "100FPS_FFMPEG"
DEV_WEIGHTS_DIRNAME = "weights"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def app_base_dir() -> Path:
    """Bundle dir (frozen) or repo root (dev) — for READ-ONLY data."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        # Paranoia: frozen without _MEIPASS (other freezers) — exe dir.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def exe_dir() -> Path:
    """Directory of the running executable (frozen) or repo root (dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Absolute path of a bundled read-only resource (``bin/ffmpeg``, ...)."""
    return app_base_dir().joinpath(*parts)


def default_weights_root() -> Path:
    """Where model weights live unless the caller overrides them.

    Frozen apps keep a writable ``models/`` folder next to the executable
    (the installer is per-user, so this is always writable); dev runs keep
    the historical relative ``weights/`` directory.
    """
    env = os.environ.get(WEIGHTS_ENV_VAR)
    if env:
        return Path(env).expanduser()
    if is_frozen():
        return exe_dir() / "models"
    return Path(DEV_WEIGHTS_DIRNAME)


def bundled_ffmpeg() -> Optional[Path]:
    """The ffmpeg shipped inside the bundle (``bin/ffmpeg[.exe]``)."""
    suffix = ".exe" if os.name == "nt" else ""
    candidate = resource_path("bin", f"ffmpeg{suffix}")
    return candidate if candidate.is_file() else None


def bundled_ffprobe() -> Optional[Path]:
    """The ffprobe shipped inside the bundle (may be absent — we fall back)."""
    suffix = ".exe" if os.name == "nt" else ""
    candidate = resource_path("bin", f"ffprobe{suffix}")
    return candidate if candidate.is_file() else None


def find_ffmpeg() -> Optional[str]:
    """Best ffmpeg: explicit env var → bundled → PATH. Never raises."""
    env = os.environ.get(FFMPEG_ENV_VAR)
    if env and Path(env).is_file():
        return env
    bundled = bundled_ffmpeg()
    if bundled is not None:
        return str(bundled)
    return shutil.which("ffmpeg")


_PREPARED = False


def prepare_frozen_environment() -> bool:
    """One-time bundle setup; returns True when it did any work.

    Prepends the bundled ``bin/`` to ``PATH`` so every ``shutil.which()``
    call in the pipeline (ffmpeg, ffprobe) finds the shipped binaries, and
    on Windows registers the bundle dir for DLL lookup (torch/CUDA DLLs).
    A no-op (returns False) on dev runs; idempotent everywhere.
    """
    global _PREPARED
    if _PREPARED or not is_frozen():
        return False
    _PREPARED = True
    bin_dir = resource_path("bin")
    if bin_dir.is_dir():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt":
        try:
            os.add_dll_directory(str(app_base_dir()))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass  # old Python / locked-down host — bootloader PATH covers it
    return True
