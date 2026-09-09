"""One-command standalone builder (Step 10).

Produces a portable directory (and ZIP) with the frozen ``100fps`` CLI +
``100fps-gui``, bundled ffmpeg/ffprobe, docs, and optionally pre-downloaded
model weights — no Python needed on the target machine::

    python packaging/build.py                        # onedir + ffmpeg + smoke
    python packaging/build.py --target onefile       # single CLI exe, no GUI
    python packaging/build.py --weights download     # also bundle AI weights
    python packaging/build.py --ffmpeg skip          # use system ffmpeg
    python packaging/build.py --ffmpeg-local /usr/bin  # offline: copy local

Prerequisites: Python 3.10+ with the project requirements installed, plus
``pyinstaller`` and ``pyinstaller-hooks-contrib`` (auto-installed unless
``--no-auto-install``). Build on the OS you ship for (Windows builds need
Windows, codesigning/notarisation for macOS is documented in the README).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pipeline.logger import get_logger  # noqa: E402


def _sibling(name: str):
    """Import a sibling helper by path.

    ``packaging/`` deliberately has NO ``__init__.py``: setuptools ships a
    top-level ``packaging`` package that would shadow (or be shadowed by)
    ours depending on import order. Path-based loading is unambiguous in
    every process, fresh or polluted.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"pack10_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_fetch_ffmpeg = _sibling("fetch_ffmpeg")
_fetch_weights = _sibling("fetch_weights")
detect_platform = _fetch_ffmpeg.detect_platform
fetch_ffmpeg = _fetch_ffmpeg.fetch_ffmpeg
fetch_all_weights = _fetch_weights.fetch_all

log = get_logger("build")

_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def read_version() -> str:
    """Project version without importing the package (bare build hosts)."""
    text = (ROOT / "pipeline" / "__init__.py").read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if not match:
        raise RuntimeError("Could not parse __version__ from pipeline/__init__.py")
    return match.group(1)


def platform_tag() -> str:
    if sys.platform == "win32":
        return "win64"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


# pip distribution name -> importable top-level module.
BUILD_TOOLS = {
    "PyInstaller": "PyInstaller",
    "pyinstaller-hooks-contrib": "_pyinstaller_hooks_contrib",
}


def _pip_install(package: str) -> None:
    log.info("Installing build prerequisite: %s ...", package)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        # Debian/Ubuntu sandboxes (PEP 668) need the override flag; plain
        # pip everywhere else. Retry once before giving up.
        log.warning("Plain pip install failed — retrying with PEP 668 override ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--break-system-packages", package],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def check_prereqs(auto_install: bool = True) -> None:
    """Fail loudly (with the fix) when the build host is not ready."""
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Building needs Python 3.10+, this is {sys.version.split()[0]}."
        )
    for package, module in BUILD_TOOLS.items():
        try:
            __import__(module)
        except ImportError:
            if not auto_install:
                raise RuntimeError(
                    f"Missing build tool {package!r}. Install it: "
                    f"pip install {package}"
                )
            _pip_install(package)
    for optional, why in (
        ("torch", "AI backends (RIFE/ESRGAN)"),
        ("gradio", "desktop GUI"),
        ("onnxruntime", "ONNX acceleration"),
        ("cv2", "frame I/O"),
    ):
        try:
            __import__(optional)
            log.info("Optional dep present: %-12s (will be bundled)", optional)
        except ImportError:
            log.warning(
                "Optional dep missing: %-12s — %s will NOT be in the bundle. "
                "Install it before building for a full release.",
                optional, why,
            )


def prepare_vendor_ffmpeg(mode: str, from_local: Optional[str], plat: str) -> None:
    """Fetch ffmpeg bins into packaging/vendor/bin/<plat>/ (unless skipped)."""
    if mode == "skip":
        log.warning("Skipping ffmpeg bundling (--ffmpeg skip): the app will "
                    "need a system ffmpeg on PATH.")
        return
    out = HERE / "vendor" / "bin" / plat
    if mode == "local":
        if from_local:
            local = from_local
        else:
            try:
                import imageio_ffmpeg

                local = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
                log.info("Using imageio-ffmpeg binary: %s", local)
            except ImportError:
                found = shutil.which("ffmpeg")
                if not found:
                    raise RuntimeError(
                        "No local ffmpeg: pass --ffmpeg-local <dir> or install "
                        "imageio-ffmpeg."
                    )
                local = str(Path(found).parent)
        fetch_ffmpeg(out, detect_platform(), from_local=local)
    else:  # auto: official static builds over the network
        fetch_ffmpeg(out, plat)


def prepare_vendor_weights(mode: str) -> None:
    """Pre-download AI weights into packaging/vendor/models/ (unless none)."""
    if mode == "none":
        log.info("Skipping weight bundling: first run downloads them "
                 "(with progress bar + hash manifest).")
        return
    fetch_all_weights(HERE / "vendor" / "models", rife=["4"], esrgan=["x4plus"])


def run_pyinstaller(target: str, dist_dir: Path, work_parent: Path) -> Path:
    """Run the .spec file; return the produced app dir (onedir) or exe."""
    env = dict(os.environ)
    env["FPS_TARGET"] = target
    work_dir = work_parent / "pyinstaller-work"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean", "--noconfirm",
        "--distpath", str(dist_dir),
        "--workpath", str(work_dir),
        str(HERE / "100fps.spec"),
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    if target == "onefile":
        exe = dist_dir / ("100fps.exe" if sys.platform == "win32" else "100fps")
    else:
        exe = dist_dir / "100fps" / ("100fps.exe" if sys.platform == "win32" else "100fps")
    if not exe.is_file():
        raise RuntimeError(f"PyInstaller finished but {exe} is missing.")
    return exe


def assemble_portable(exe: Path, version: str) -> Path:
    """Copy docs (+ pre-bundled models) next to the frozen app; return dir."""
    portable = exe.parent if exe.parent.name == "100fps" else exe.parent
    if (HERE / "dist-readme.txt").is_file():
        shutil.copy2(HERE / "dist-readme.txt", portable / "README.txt")
    if (HERE / "THIRD_PARTY_LICENSES.txt").is_file():
        shutil.copy2(
            HERE / "THIRD_PARTY_LICENSES.txt", portable / "THIRD_PARTY_LICENSES.txt"
        )
    vendor_models = HERE / "vendor" / "models"
    if vendor_models.is_dir() and any(vendor_models.iterdir()):
        # Frozen default_weights_root() points here (<exe-dir>/models).
        shutil.copytree(vendor_models, portable / "models", dirs_exist_ok=True)
        log.info("Bundled pre-downloaded weights into %s", portable / "models")
    log.info("Portable app assembled: %s", portable)
    return portable


def smoke_test(exe: Path, version: str) -> None:
    """The frozen binary must answer --version/--help (real boot test)."""
    for flag in ("--version", "--help"):
        proc = subprocess.run(
            [str(exe), flag], capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Smoke test FAILED: {exe.name} {flag} exited "
                f"{proc.returncode}:\n{proc.stderr[-2000:]}"
            )
    proc = subprocess.run([str(exe), "--version"], capture_output=True, text=True)
    if version not in proc.stdout:
        raise RuntimeError(
            f"Smoke test FAILED: --version printed {proc.stdout!r}, "
            f"expected {version!r}."
        )
    log.info("Smoke test passed: %s", proc.stdout.strip())


def make_zip(portable: Path, version: str, out_dir: Path) -> Path:
    """Zip the portable dir -> 100fps-<ver>-<plat>.zip; return the archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"100fps-{version}-{platform_tag()}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(portable.rglob("*")):
            zf.write(path, path.relative_to(portable.parent))
    size_mb = archive.stat().st_size / 1e6
    log.info("Portable archive: %s (%.0f MB)", archive, size_mb)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=["onedir", "onefile"], default="onedir",
        help="onedir: folder with CLI+GUI (default, recommended). "
        "onefile: single CLI exe, no GUI.",
    )
    parser.add_argument(
        "--ffmpeg", choices=["auto", "local", "skip"], default="auto",
        help="auto: download static builds (default). local: copy a local "
        "ffmpeg (offline). skip: rely on system ffmpeg.",
    )
    parser.add_argument(
        "--ffmpeg-local", default=None,
        help="Binary or directory to copy with --ffmpeg local.",
    )
    parser.add_argument(
        "--weights", choices=["none", "download"], default="none",
        help="none: first run downloads weights (default). download: "
        "pre-bundle RIFE v4 + ESRGAN x4plus (~130 MB extra).",
    )
    parser.add_argument(
        "--dist", default=None,
        help="Output directory (default: packaging/dist).",
    )
    parser.add_argument(
        "--no-smoke", action="store_true", help="Skip the frozen smoke test.",
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Skip the portable ZIP archive.",
    )
    parser.add_argument(
        "--no-auto-install", action="store_true",
        help="Do not pip-install missing build tools (fail instead).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        version = read_version()
        plat = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
        log.info("Building 100fps %s for %s (%s) ...", version, plat, args.target)
        check_prereqs(auto_install=not args.no_auto_install)
        prepare_vendor_ffmpeg(args.ffmpeg, args.ffmpeg_local, plat)
        prepare_vendor_weights(args.weights)
        dist_dir = Path(args.dist) if args.dist else HERE / "dist"
        with tempfile.TemporaryDirectory(prefix="100fps_build_") as tmp:
            exe = run_pyinstaller(args.target, dist_dir, Path(tmp))
        portable = assemble_portable(exe, version)
        if not args.no_smoke:
            smoke_test(exe, version)
        if not args.no_zip:
            make_zip(portable, version, dist_dir)
        log.info("BUILD OK: %s", portable)
        return 0
    except subprocess.CalledProcessError as exc:
        log.error("BUILD FAILED: command exited %s: %s", exc.returncode, exc.cmd)
        return 2
    except Exception as exc:  # noqa: BLE001 - operator-facing tool
        log.error("BUILD FAILED: %s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
