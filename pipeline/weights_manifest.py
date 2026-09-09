"""SHA-256 manifest for model weights (Step 10 — tamper-evident models).

``packaging/fetch_weights.py`` downloads the official weights once and
records ``{relative path -> sha256}`` in ``weights-manifest.json`` next to
them. At runtime :func:`verify_against_manifest` re-checks any weight file
against that manifest:

- no manifest (or no entry) → ``None``: nothing to check, legacy behaviour;
- hash matches → ``True``: use the file;
- hash differs → ``False``: the file is corrupt/tampered — the caller
  deletes it and re-downloads once, then fails loudly if still bad.

Hashes are recorded by the maintainer at bundle time (trust-on-first-
download); users verify against the shipped manifest, never against
hard-coded strings that would rot with every upstream re-release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_FILENAME = "weights-manifest.json"
_CHUNK = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file (streamed — weight files are hundreds of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(weights_root: str | Path) -> Path:
    return Path(weights_root) / MANIFEST_FILENAME


def load_manifest(weights_root: str | Path) -> Dict[str, Any]:
    """Parse the manifest; ``{}`` when missing/unreadable (legacy dirs)."""
    path = manifest_path(weights_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    files = data.get("files", {})
    return files if isinstance(files, dict) else {}


def build_manifest(weights_root: str | Path) -> Dict[str, Dict[str, Any]]:
    """Hash every weight file under ``weights_root`` (for the bundler)."""
    root = Path(weights_root)
    entries: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        rel = path.relative_to(root).as_posix()
        entries[rel] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return entries


def write_manifest(weights_root: str | Path) -> Path:
    """(Re)generate ``weights-manifest.json``; return its path."""
    from pipeline import __version__ as app_version  # lazy: avoid import cycles

    root = Path(weights_root)
    data = {"app_version": app_version, "files": build_manifest(root)}
    path = manifest_path(root)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def verify_against_manifest(
    path: str | Path, weights_root: str | Path
) -> Optional[bool]:
    """Check one file against the manifest.

    Returns ``None`` when there is no manifest/entry (nothing to enforce),
    otherwise whether the on-disk hash matches the recorded one.
    """
    path, root = Path(path), Path(weights_root)
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return None  # file outside the weights root — not our manifest's business
    entries = load_manifest(root)
    entry = entries.get(rel)
    if not isinstance(entry, dict) or "sha256" not in entry:
        return None
    if not path.is_file():
        return False
    return file_sha256(path) == entry["sha256"]
