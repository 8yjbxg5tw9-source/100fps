"""Pre-download AI weights for offline installers + write the hash manifest.

Usage (run from the repo root)::

    python packaging/fetch_weights.py --out packaging/vendor/models
    python packaging/fetch_weights.py --out models --rife all --esrgan all
    python packaging/fetch_weights.py --list

The default set (RIFE v4 + Real-ESRGAN x4plus, ~130 MB) covers the
out-of-the-box experience; ``--write-manifest`` (on by default) records
SHA-256 hashes next to the files so frozen apps can verify them at
runtime (see :mod:`pipeline.weights_manifest`).

When a Real-CUGAN weights module lands, register its fetcher in
:func:`fetch_all` — the manifest format already supports any layout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.esrgan.weights import DEFAULT_ESRGAN_MODEL, ESRGAN_MODELS  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402
from pipeline.rife.weights import DEFAULT_RIFE_VERSION, RIFE_VERSIONS  # noqa: E402
from pipeline.weights_manifest import write_manifest  # noqa: E402

log = get_logger("fetch-weights")


def parse_selection(value: str, known: List[str]) -> List[str]:
    """Expand ``all``/``none``/``a,b`` against the known model names."""
    value = value.strip().lower()
    if value == "all":
        return list(known)
    if value in ("none", ""):
        return []
    chosen = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in known]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Known: {known}.")
    return chosen


def fetch_all(
    out_dir: str | Path,
    rife: List[str],
    esrgan: List[str],
    write_hash_manifest: bool = True,
) -> List[Path]:
    """Download every selected weight file into ``out_dir``; return paths."""
    from pipeline.esrgan.weights import ensure_esrgan_weights
    from pipeline.rife.weights import ensure_weights

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fetched: List[Path] = []
    for version in rife:
        fetched.append(
            ensure_weights(version=version, weights_root=out, logger=log)
        )
    for model in esrgan:
        fetched.append(
            ensure_esrgan_weights(model=model, weights_root=out, logger=log)
        )
    if write_hash_manifest and fetched:
        manifest = write_manifest(out)
        log.info("Wrote hash manifest: %s", manifest)
    total_mb = sum(p.stat().st_size for p in fetched) / 1e6
    log.info("Fetched %d file(s), %.0f MB, into %s", len(fetched), total_mb, out)
    return fetched


def list_models() -> str:
    lines = ["RIFE versions (--rife):"]
    for version in sorted(RIFE_VERSIONS):
        marker = " [default]" if version == DEFAULT_RIFE_VERSION else ""
        lines.append(f"  {version}{marker}: {RIFE_VERSIONS[version].description}")
    lines.append("Real-ESRGAN models (--esrgan):")
    for model in sorted(ESRGAN_MODELS):
        marker = " [default]" if model == DEFAULT_ESRGAN_MODEL else ""
        lines.append(f"  {model}{marker}: {ESRGAN_MODELS[model].description}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="packaging/vendor/models",
        help="Target weights root (default: packaging/vendor/models).",
    )
    parser.add_argument(
        "--rife", default=DEFAULT_RIFE_VERSION,
        help=f"RIFE versions: {sorted(RIFE_VERSIONS)} / all / none "
        f"(default: {DEFAULT_RIFE_VERSION}).",
    )
    parser.add_argument(
        "--esrgan", default=DEFAULT_ESRGAN_MODEL,
        help=f"ESRGAN models: {sorted(ESRGAN_MODELS)} / all / none "
        f"(default: {DEFAULT_ESRGAN_MODEL}).",
    )
    parser.add_argument(
        "--no-manifest", action="store_true",
        help="Skip writing weights-manifest.json.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List fetchable models and exit.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(list_models())
        return 0
    try:
        rife = parse_selection(args.rife, sorted(RIFE_VERSIONS))
        esrgan = parse_selection(args.esrgan, sorted(ESRGAN_MODELS))
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if not rife and not esrgan:
        log.error("Nothing selected (both --rife and --esrgan are 'none').")
        return 2
    try:
        fetch_all(args.out, rife, esrgan, not args.no_manifest)
    except Exception as exc:  # noqa: BLE001 - operator-facing tool
        log.error("Weight fetch FAILED: %s: %s", type(exc).__name__, exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
