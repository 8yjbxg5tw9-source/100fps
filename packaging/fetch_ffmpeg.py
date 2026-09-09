"""Fetch static ffmpeg+ffprobe binaries for bundling (Step 10).

Usage (run from the repo root)::

    python packaging/fetch_ffmpeg.py                    # auto platform
    python packaging/fetch_ffmpeg.py --platform windows
    python packaging/fetch_ffmpeg.py --from-local /usr/bin  # offline: copy

Sources are the well-known static builds (GyanDossun/gyan.dev for Windows,
johnvansickle.com for Linux, evermeet.cx for macOS). Every fetched binary
is verified by executing ``<bin> -version`` before it is accepted; if a
URL rots, use ``--from-local`` with distro binaries instead.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.logger import get_logger  # noqa: E402

log = get_logger("fetch-ffmpeg")

#: filename (or glob inside the archive) -> name in our ``bin/`` dir.
SOURCES = {
    "windows": {
        "url": "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        "kind": "zip",
        "members": {
            "*/bin/ffmpeg.exe": "ffmpeg.exe",
            "*/bin/ffprobe.exe": "ffprobe.exe",
        },
    },
    "linux": {
        "url": "https://johnvansickle.com/ffmpeg/releases/"
        "ffmpeg-release-amd64-static.tar.xz",
        "kind": "tarxz",
        "members": {
            "ffmpeg-*-amd64-static/ffmpeg": "ffmpeg",
            "ffmpeg-*-amd64-static/ffprobe": "ffprobe",
        },
    },
    "macos": {
        "url": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",
        "kind": "zip",
        "members": {"ffmpeg": "ffmpeg"},
        "extra_url": "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip",
        "extra_members": {"ffprobe": "ffprobe"},
    },
}

_MIN_BIN_BYTES = 1_000_000  # real static ffmpeg is 25-80 MB


def detect_platform() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _download(url: str, dest: Path, desc: str) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (100fps-packaging)"}
    )
    log.info("Downloading %s ...", url)
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as fh:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        bar = _progress_bar(total_bytes, desc)
        with bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))


def _progress_bar(total: Optional[int], desc: str):  # noqa: ANN202 - tiny helper
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="B", unit_scale=True)
    except ImportError:

        class _Null:
            def update(self, n: int) -> None:
                pass

            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(self, *args: object) -> None:
                pass

        return _Null()


def _extract_members(
    archive: Path, kind: str, members: Dict[str, str], out_dir: Path
) -> List[Path]:
    """Extract wanted members (glob -> target name); return written paths."""
    written: List[Path] = []
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            for pattern, target in members.items():
                match = next(
                    (n for n in names if fnmatch.fnmatch(n, pattern)
                     and not n.endswith("/")),
                    None,
                )
                if match is None:
                    raise FileNotFoundError(
                        f"{archive.name} has no member matching {pattern!r}"
                    )
                dest = out_dir / target
                with zf.open(match) as src, open(dest, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                written.append(dest)
    elif kind == "tarxz":
        with tarfile.open(archive, "r:xz") as tf:
            infos = {m.name: m for m in tf.getmembers() if m.isfile()}
            for pattern, target in members.items():
                match = next(
                    (n for n in infos if fnmatch.fnmatch(n, pattern)), None
                )
                if match is None:
                    raise FileNotFoundError(
                        f"{archive.name} has no member matching {pattern!r}"
                    )
                dest = out_dir / target
                with tf.extractfile(infos[match]) as src, open(dest, "wb") as fh:  # type: ignore[union-attr]
                    shutil.copyfileobj(src, fh)
                written.append(dest)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown archive kind: {kind}")
    return written


def verify_binary(path: Path) -> str:
    """Run ``<bin> -version``; return the first output line (raises if bad)."""
    if path.stat().st_size < _MIN_BIN_BYTES:
        raise ValueError(f"{path} is suspiciously small — not a real binary?")
    if sys.platform != "win32":
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    proc = subprocess.run(
        [str(path), "-version"], capture_output=True, text=True, timeout=60
    )
    first = (proc.stdout or proc.stderr).splitlines()[0] if proc.stdout or proc.stderr else ""
    if proc.returncode != 0 or "ffmpeg version" not in first.lower() and "ffprobe version" not in first.lower():
        raise ValueError(f"{path} failed -version check: {first!r}")
    return first.strip()


def _from_local(src: Path, out_dir: Path) -> List[Path]:
    """Copy ffmpeg (+ ffprobe sibling when present) from a local path.

    Accepts an exact binary, a directory of distro binaries, or an
    imageio-ffmpeg style ``ffmpeg-<plat>-<ver>`` file (renamed to the
    canonical ``ffmpeg`` on copy).
    """
    src = src.expanduser()
    directory = src if src.is_dir() else src.parent
    names = ["ffmpeg", "ffmpeg.exe", "ffprobe", "ffprobe.exe"]
    candidates = [src] if src.is_file() else [directory / n for n in names]
    if src.is_dir():  # imageio-ffmpeg layout: ffmpeg-linux-x86_64-v7.0.2, ...
        for pattern, canonical in (("ffmpeg-*", "ffmpeg"), ("ffprobe-*", "ffprobe")):
            matches = sorted(
                p for p in directory.glob(pattern)
                if p.is_file() and not p.suffix == ".py"
            )
            for match in matches:
                candidates.append(match)
    copied: List[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.name in names:
            dest = out_dir / candidate.name
        elif candidate.name.startswith("ffmpeg-"):
            dest = out_dir / "ffmpeg"
        elif candidate.name.startswith("ffprobe-"):
            dest = out_dir / "ffprobe"
        else:
            continue
        if dest not in copied:
            shutil.copy2(candidate, dest)
            copied.append(dest)
    if not any(p.name == "ffmpeg" or p.name == "ffmpeg.exe" for p in copied):
        raise FileNotFoundError(f"No ffmpeg binary found at {src}")
    return copied


def fetch_ffmpeg(
    out_dir: str | Path,
    platform: Optional[str] = None,
    from_local: Optional[str | Path] = None,
) -> List[Path]:
    """Fetch + verify ffmpeg/ffprobe into ``out_dir``; return the binaries."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if from_local is not None:
        bins = _from_local(Path(from_local), out)
    else:
        plat = platform or detect_platform()
        if plat not in SOURCES:
            raise ValueError(f"Unknown platform {plat!r}. Known: {sorted(SOURCES)}.")
        source = SOURCES[plat]
        with tempfile.TemporaryDirectory(prefix="ffmpeg_fetch_") as tmp:
            archive = Path(tmp) / "ffmpeg_dl"
            _download(source["url"], archive, desc=f"ffmpeg-{plat}")
            bins = _extract_members(archive, source["kind"], source["members"], out)
            if "extra_url" in source:
                archive2 = Path(tmp) / "ffprobe_dl"
                _download(source["extra_url"], archive2, desc=f"ffprobe-{plat}")
                bins += _extract_members(
                    archive2, source["kind"], source["extra_members"], out
                )
    for binary in bins:
        log.info("%s: %s", binary.name, verify_binary(binary))
    return bins


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", choices=["auto", "windows", "linux", "macos"], default="auto",
        help="Target platform binaries (default: auto = this machine).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Target bin dir (default: packaging/vendor/bin/<platform>).",
    )
    parser.add_argument(
        "--from-local", default=None,
        help="Copy from a local ffmpeg binary/dir instead of downloading.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plat = detect_platform() if args.platform == "auto" else args.platform
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "vendor" / "bin" / plat
    )
    try:
        fetch_ffmpeg(out, plat, args.from_local)
    except Exception as exc:  # noqa: BLE001 - operator-facing tool
        log.error("FFmpeg fetch FAILED: %s: %s", type(exc).__name__, exc)
        return 2
    log.info("FFmpeg binaries ready in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
