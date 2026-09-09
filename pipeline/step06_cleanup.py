"""STEP 6 — Safe purge of temporary frames/audio + disk-space accounting.

After Step 5 produces the final video, the workspace holds potentially
hundreds of GB of intermediate data (raw 720p frames, interpolated frames,
8K frames, extracted audio). This step reclaims that space — **safety first**:

1. **Pre-cleanup safety check**: the final video must exist and be non-empty,
   otherwise cleanup is *refused* with a loud message and nothing is deleted
   (a missing ``assembly_verified`` flag additionally warns but does not
   block — the file check is authoritative).
2. **Containment**: only paths strictly inside ``workspace_root`` are ever
   deleted (resolved + verified, so a crafted config can never wipe outside
   data). ``config.json`` and ``*.log`` files are always kept for provenance
   and Step 7 (resume/checkpoints).
3. **Purge**: ``temp_raw_frames``, ``interpolated_720p``, ``upscaled_8k`` and
   the extracted audio file(s) are removed (``shutil.rmtree`` for dirs).
   Categories can be individually kept (``--keep-raw`` etc.) or previewed
   with ``--dry-run`` (deletes nothing, reports the would-be savings).
4. **Accounting**: freed bytes are measured *before* deletion and logged in
   GB (e.g. ``Cleaned up 142.50 GB of temporary frame data.``).
5. **Failure tolerance**: locked/unreadable files are collected per-path
   (``try/except`` + ``rmtree`` error handler) and reported as warnings —
   one stuck file never aborts the whole cleanup.

Typical usage::

    from pipeline.config import PipelineConfig
    from pipeline.step06_cleanup import Step06Cleanup

    config = PipelineConfig.load("workspace/config.json")
    result = Step06Cleanup(dry_run=False).run(config)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.base import PipelineStep
from pipeline.config import CONFIG_FILENAME, PipelineConfig
from pipeline.exceptions import CleanupSafetyError
from pipeline.logger import get_logger


@dataclass
class CleanupTarget:
    label: str            # "raw 720p frames" / "audio" / ...
    path: Path
    kind: str             # "dir" | "file"
    size_bytes: int = 0
    file_count: int = 0


@dataclass
class Step06Result:
    targets: List[CleanupTarget] = field(default_factory=list)
    freed_bytes: int = 0
    deleted_files: int = 0
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (path, error)
    dry_run: bool = False

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / (1024**3)


class Step06Cleanup(PipelineStep[Step06Result]):
    """Step 6 implementation. See module docstring for the full contract."""

    name = "step06_cleanup"

    def __init__(
        self,
        dry_run: bool = False,
        keep_raw: bool = False,
        keep_interpolated: bool = False,
        keep_upscaled: bool = False,
        keep_audio: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dry_run = dry_run
        self.keep_raw = keep_raw
        self.keep_interpolated = keep_interpolated
        self.keep_upscaled = keep_upscaled
        self.keep_audio = keep_audio
        self.log = logger or get_logger(__name__)

    # -- Orchestration -----------------------------------------------------------
    def run(self, config: PipelineConfig) -> Step06Result:
        mode = "DRY-RUN (nothing will be deleted)" if self.dry_run else "purge"
        self.log.info("=== Step 6: Temporary data cleanup (%s) started ===", mode)

        self._safety_check(config)
        targets = self._collect_targets(config)
        if not targets:
            self.log.info("Nothing to clean (workspace already empty or all kept).")

        result = Step06Result(targets=targets, dry_run=self.dry_run)
        for target in targets:
            self.log.info(
                "%s %s: %d file(s), %.2f GB -- %s",
                "Would delete" if self.dry_run else "Deleting",
                target.label, target.file_count, target.size_bytes / (1024**3),
                target.path,
            )
            if self.dry_run:
                result.freed_bytes += target.size_bytes
                result.deleted_files += target.file_count
                continue
            try:
                self._delete_target(target, result)
            except Exception as exc:  # noqa: BLE001 - one target never kills cleanup
                message = f"{type(exc).__name__}: {exc}"
                self.log.warning("Failed to delete %s: %s", target.path, message)
                result.failed.append((str(target.path), message))
            else:
                result.freed_bytes += target.size_bytes
                result.deleted_files += target.file_count

        self._report(result)
        config.cleanup_completed = not self.dry_run and not result.failed
        # On dry-run / partial failure the flags stay informative, not "done".
        if self.dry_run:
            config.cleanup_completed = None
        config.cleanup_freed_bytes = result.freed_bytes
        config.cleanup_failed_count = len(result.failed)
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 7: %s", saved)
        return result

    # -- 1. Safety gate ------------------------------------------------------------------
    def _safety_check(self, config: PipelineConfig) -> None:
        final = config.final_output_path
        if not final.is_file():
            raise CleanupSafetyError(
                f"REFUSING cleanup: final video is missing ({final}). "
                f"Temporary frames are kept so nothing is lost. Produce the "
                f"final video (Step 5) first."
            )
        size = final.stat().st_size
        if size <= 0:
            raise CleanupSafetyError(
                f"REFUSING cleanup: final video is empty (0 bytes): {final}. "
                f"Temporary frames are kept so the video can be re-assembled."
            )
        self.log.info("Safety check passed: final video exists (%.2f MB).",
                      size / (1024 * 1024))
        if not config.assembly_verified:
            self.log.warning(
                "Final video was not formally verified (assembly_verified=%s); "
                "proceeding because the file exists and is non-empty.",
                config.assembly_verified,
            )

    # -- 2. Target collection (contained to workspace) -------------------------
    def _collect_targets(self, config: PipelineConfig) -> List[CleanupTarget]:
        root = config.workspace_root.resolve()
        candidates: List[Tuple[str, Path, str, bool]] = [
            ("raw 720p frames", config.temp_raw_frames, "dir", self.keep_raw),
            ("interpolated 720p frames", config.interpolated_720p, "dir",
             self.keep_interpolated),
            ("upscaled 8K frames", config.upscaled_8k, "dir", self.keep_upscaled),
        ]
        if config.audio_path is not None:
            candidates.append(
                ("extracted audio", Path(config.audio_path), "file", self.keep_audio)
            )
        targets: List[CleanupTarget] = []
        for label, path, kind, keep in candidates:
            if keep:
                self.log.info("Keeping %s (--keep flag): %s", label, path)
                continue
            if not self._is_within_workspace(path, root):
                self.log.warning(
                    "SKIPPING %s: %s is outside the workspace (%s) -- refusing "
                    "to delete out-of-workspace data.",
                    label, path, root,
                )
                continue
            if not path.exists():
                self.log.info("Already absent, skipping %s: %s", label, path)
                continue
            if kind == "dir" and not path.is_dir():
                self.log.warning("SKIPPING %s: %s is not a directory.", label, path)
                continue
            if kind == "file" and not path.is_file():
                self.log.warning("SKIPPING %s: %s is not a file.", label, path)
                continue
            size, count = self._measure(path, kind)
            targets.append(CleanupTarget(label, path, kind, size, count))
        return targets

    @staticmethod
    def _is_within_workspace(path: Path, root: Path) -> bool:
        """True only for strict workspace children (never root itself)."""
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return False
        return resolved != root and root in resolved.parents

    @staticmethod
    def _measure(path: Path, kind: str) -> Tuple[int, int]:
        if kind == "file":
            try:
                return path.stat().st_size, 1
            except OSError:
                return 0, 0
        total, count = 0, 0
        try:
            for entry in path.rglob("*"):
                try:
                    if entry.is_file() and not entry.is_symlink():
                        total += entry.stat().st_size
                        count += 1
                except OSError:
                    continue
        except OSError:
            pass
        return total, count

    # -- 3. Deletion (failure-tolerant) ---------------------------------------------------
    def _delete_target(self, target: CleanupTarget, result: Step06Result) -> None:
        if target.kind == "file":
            try:
                target.path.unlink()
            except OSError as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.log.warning("Could not delete file %s: %s", target.path, message)
                result.failed.append((str(target.path), message))
            return
        errors: List[Tuple[str, str]] = []

        def _on_error(func: object, path: str, exc: object) -> None:
            err = exc[1] if isinstance(exc, tuple) else exc
            errors.append((str(path), f"{type(err).__name__}: {err}"))

        # shutil.rmtree(exc-handler): `onexc` on 3.12+, `onerror` before.
        try:
            shutil.rmtree(target.path, onexc=_on_error)  # type: ignore[call-arg]
        except TypeError:
            shutil.rmtree(target.path, onerror=_on_error)  # type: ignore[call-arg]
        for path, message in errors:
            self.log.warning("Could not delete %s: %s", path, message)
            result.failed.append((path, message))
        if target.path.exists():
            self.log.warning(
                "Directory partially remains after purge: %s", target.path
            )

    # -- 4. Report ------------------------------------------------------------------------------------
    def _report(self, result: Step06Result) -> None:
        if self.dry_run:
            self.log.info(
                "DRY-RUN: would free %.2f GB (%d files). Nothing was deleted.",
                result.freed_gb, result.deleted_files,
            )
            return
        self.log.info(
            "Cleaned up %.2f GB of temporary frame data (%d files).",
            result.freed_gb, result.deleted_files,
        )
        if result.failed:
            self.log.warning(
                "%d path(s) could not be deleted (locked/permissions?) -- see "
                "warnings above. Re-run Step 6 after closing other programs.",
                len(result.failed),
            )
        else:
            self.log.info("Workspace is clean (config.json and logs kept).")
