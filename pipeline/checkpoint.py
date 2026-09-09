"""STEP 7 — Checkpoint & resume: crash-proof progress tracking for the pipeline.

8K @ 1000 FPS processing runs for hours, so a power cut, a VRAM overflow or
an accidental Ctrl+C must never mean "start over from zero". This module is
the pipeline's flight recorder:

- :class:`PipelineState` — JSON-serialisable run state (``pipeline_state.json``
  in the workspace): which steps finished, how far the frame loops got, and a
  snapshot of the run identity (input video, targets, source metadata).
- :class:`CheckpointManager` — atomic state persistence, startup resume
  decisions (resume / overwrite / fresh), progress recording called from the
  Step 3/4 frame loops, and safe progress discarding.
- :func:`scan_frame_indices` / :func:`continuous_prefix_length` — disk truth
  for frame-level resume: already-written ``frame_*.png`` files are never
  re-processed; the AI backends restart exactly where the files stop.
- :func:`is_oom_error` — torch-free OOM detection so Steps 3/4 can halve
  batch/tile sizes and retry the *same* frame instead of crashing.

Design rule: **the disk is the truth, the state file is the accelerator.**
Resume frontiers are always re-derived by scanning output directories (a
checkpoint entry alone never marks frames done), so even a torn write or a
manually deleted frame converges to the correct restart point.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pipeline.config import (
    AUDIO_FILENAME_AAC,
    AUDIO_FILENAME_WAV,
    DIR_INTERPOLATED,
    DIR_RAW_FRAMES,
    DIR_UPSCALED_8K,
)
from pipeline.exceptions import CheckpointError
from pipeline.logger import get_logger

STATE_FILENAME = "pipeline_state.json"
STATE_VERSION = 1

#: Step number -> PipelineStep.name key used inside the state file.
STEP_KEYS = {
    1: "step01_environment",
    2: "step02_frames",
    3: "step03_interpolate",
    4: "step04_upscale",
    5: "step05_assemble",
    6: "step06_cleanup",
}

#: Snapshot keys that identify "the same run" for resume validation.
IDENTITY_KEYS = (
    "input_video",
    "final_output",
    "target_fps",
    "target_width",
    "target_height",
)

ResumeAction = Literal["resume", "overwrite", "fresh"]


# ---------------------------------------------------------------------------
# Disk truth helpers (pure functions — the core of frame-level resume)
# ---------------------------------------------------------------------------
def scan_frame_indices(directory: str | Path, prefix: str, suffix: str) -> List[int]:
    """Sorted unique frame numbers present as ``prefix + N + suffix`` files.

    Only non-empty regular files count: a zero-byte file is a torn write from
    a crash and must be recomputed, not trusted.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = set()
    for path in directory.glob(f"{prefix}*{suffix}"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        middle = path.name[len(prefix):len(path.name) - len(suffix)] if suffix else path.name[len(prefix):]
        try:
            found.add(int(middle))
        except ValueError:
            continue  # e.g. "frame_preview.png" — not a numbered frame
    return sorted(found)


def continuous_prefix_length(indices: List[int]) -> int:
    """Length of the ``1..N`` prefix with no gaps (0 when file #1 is missing).

    Callers treat the last prefix file as *suspect* (it may be torn by a
    kill -9 mid-write) and restart emission at file ``N`` itself, i.e. only
    files ``1..N-1`` are trusted without recomputation.
    """
    present = set(indices)
    count = 0
    while count + 1 in present:
        count += 1
    return count


def is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) signals a GPU/CUDA OOM.

    Deliberately torch-free: ``torch.cuda.OutOfMemoryError`` subclasses
    ``RuntimeError`` with a ``"... out of memory ..."`` message, and our
    backends wrap failures in ``*InferenceError`` preserving that text, so a
    message + cause-chain scan recognises OOM on any machine.
    """
    seen = 0
    current: Optional[BaseException] = exc
    while current is not None and seen < 8:
        if type(current).__name__ == "OutOfMemoryError":
            return True
        try:
            text = str(current)
        except Exception:  # noqa: BLE001 - exotic __str__ must not break recovery
            text = ""
        if "out of memory" in text.lower():
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


def steps_to_run(from_step: int, to_step: int, skip_through: int) -> List[int]:
    """Steps in ``[from_step, to_step]`` not already completed (``<= skip``)."""
    return [n for n in range(from_step, to_step + 1) if n > skip_through]


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepProgress:
    status: str = "in_progress"  # "in_progress" | "complete"
    last_processed_frame_index: int = 0
    target_count: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepProgress":
        known = {k: data.get(k) for k in (
            "status", "last_processed_frame_index", "target_count",
            "params", "updated_at",
        )}
        return cls(
            status=known["status"] or "in_progress",
            last_processed_frame_index=int(known["last_processed_frame_index"] or 0),
            target_count=known["target_count"],
            params=dict(known["params"] or {}),
            updated_at=known["updated_at"] or "",
        )


@dataclass
class PipelineState:
    version: int = STATE_VERSION
    updated_at: str = ""
    workspace: str = ""
    snapshot: Dict[str, Any] = field(default_factory=dict)
    last_completed_step: int = 0
    steps: Dict[str, StepProgress] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "workspace": self.workspace,
            "snapshot": self.snapshot,
            "last_completed_step": self.last_completed_step,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        steps = {
            k: StepProgress.from_dict(v)
            for k, v in (data.get("steps") or {}).items()
            if isinstance(v, dict)
        }
        return cls(
            version=int(data.get("version") or STATE_VERSION),
            updated_at=data.get("updated_at") or "",
            workspace=data.get("workspace") or "",
            snapshot=dict(data.get("snapshot") or {}),
            last_completed_step=int(data.get("last_completed_step") or 0),
            steps=steps,
        )

    def save(self, path: str | Path) -> Path:
        """Atomically persist (temp file + rename: readers never see halves)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utcnow()
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        """Strict load: missing file -> ``FileNotFoundError`` passthrough,
        corrupt JSON -> :class:`CheckpointError`."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, ValueError) as exc:
            raise CheckpointError(
                f"Checkpoint file {path} is unreadable/corrupt: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise CheckpointError(f"Checkpoint file {path} is not a JSON object.")
        return cls.from_dict(data)

    def describe_progress(self) -> str:
        parts = [f"last completed step: {self.last_completed_step}"]
        for key, prog in self.steps.items():
            if prog.status == "in_progress" and prog.last_processed_frame_index:
                target = f"/{prog.target_count}" if prog.target_count else ""
                parts.append(
                    f"{key}: frame {prog.last_processed_frame_index}{target}"
                )
        return "; ".join(parts)


@dataclass
class ResumeDecision:
    action: ResumeAction
    skip_through_step: int          # steps <= this are already done (resume only)
    state: Optional[PipelineState]  # attached checkpoint (resume only)
    reason: str


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Owns ``workspace/pipeline_state.json`` for one pipeline invocation."""

    def __init__(
        self,
        workspace_root: str | Path,
        state_filename: str = STATE_FILENAME,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.state_path = self.workspace_root / state_filename
        self.log = logger or get_logger(__name__)
        self._state: Optional[PipelineState] = None

    @property
    def state(self) -> Optional[PipelineState]:
        return self._state

    def _require_state(self) -> PipelineState:
        if self._state is None:
            raise CheckpointError(
                "No active checkpoint run: call begin_run() (fresh) or "
                "attach() (resume) before recording progress."
            )
        return self._state

    # -- Snapshots ---------------------------------------------------------------
    @staticmethod
    def _norm_path(value: Any) -> Any:
        if value is None:
            return None
        return os.path.normcase(os.path.abspath(os.fspath(value)))

    @classmethod
    def snapshot_from_values(
        cls,
        input_video: Any,
        final_output: Any,
        target_fps: float,
        target_width: int,
        target_height: int,
        **extra: Any,
    ) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "input_video": str(input_video),
            "final_output": str(final_output),
            "target_fps": float(target_fps),
            "target_width": int(target_width),
            "target_height": int(target_height),
        }
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, Path):
                value = str(value)
            snapshot[key] = value
        return snapshot

    @classmethod
    def snapshot_from_config(cls, config: Any) -> Dict[str, Any]:
        """Identity + source metadata (metadata enables resume even if the
        config file itself was deleted — Step 2's probe results live here)."""
        return cls.snapshot_from_values(
            config.input_video_path,
            config.final_output_path,
            config.target_fps,
            config.target_width,
            config.target_height,
            original_fps=config.original_fps,
            duration_sec=config.duration_sec,
            total_frames=config.total_frames,
            source_width=config.source_width,
            source_height=config.source_height,
            interpolation_factor=config.interpolation_factor,
            has_audio=config.has_audio,
            audio_path=config.audio_path,
            extracted_frame_count=config.extracted_frame_count,
            frame_pattern=config.frame_pattern,
        )

    @classmethod
    def validate_against(
        cls, state_snapshot: Dict[str, Any], desired: Dict[str, Any]
    ) -> List[str]:
        """Identity mismatches between a checkpoint and the requested run."""
        diffs = []
        for key in IDENTITY_KEYS:
            old, new = state_snapshot.get(key), desired.get(key)
            if old is None or new is None:
                continue
            if key in ("input_video", "final_output"):
                same = cls._norm_path(old) == cls._norm_path(new)
            elif key == "target_fps":
                same = float(old) == float(new)
            else:
                same = old == new
            if not same:
                diffs.append(f"{key}: checkpoint={old!r} vs run={new!r}")
        return diffs

    # -- Lifecycle -------------------------------------------------------------------
    def load_existing(self) -> Optional[PipelineState]:
        """Read the checkpoint, tolerating absence AND corruption.

        A corrupt file (power cut mid-write — rare thanks to atomic saves) is
        quarantined to ``*.corrupt-<timestamp>.json`` for forensics and
        treated as "no checkpoint", never as a fatal error: recovery tooling
        must not itself need recovery.
        """
        if not self.state_path.exists():
            return None
        try:
            return PipelineState.load(self.state_path)
        except CheckpointError as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            backup = self.state_path.with_name(f"{self.state_path.name}.corrupt-{stamp}")
            try:
                os.replace(self.state_path, backup)
            except OSError:
                backup = self.state_path
            self.log.warning(
                "%s Quarantined to %s; starting fresh.", exc, backup
            )
            return None

    def decide(
        self,
        desired_snapshot: Dict[str, Any],
        policy: str = "ask",
        interactive: Optional[bool] = None,
    ) -> ResumeDecision:
        """Startup decision: resume the checkpoint, overwrite it, or go fresh.

        ``policy``: ``"ask"`` (prompt on TTY, auto-resume otherwise),
        ``"yes"`` (always resume when possible), ``"no"`` (discard + scratch).
        """
        if policy not in ("ask", "yes", "no"):
            raise ValueError(
                f"policy must be 'ask', 'yes' or 'no', got {policy!r}."
            )
        state = self.load_existing()
        if state is None:
            return ResumeDecision("fresh", 0, None, "no checkpoint found — starting fresh")
        diffs = self.validate_against(state.snapshot, desired_snapshot)
        if diffs:
            reason = (
                "checkpoint is for a different run "
                f"({'; '.join(diffs)}) — starting fresh"
            )
            self.log.warning("Checkpoint mismatch: %s.", "; ".join(diffs))
            return ResumeDecision("fresh", 0, None, reason)
        has_progress = state.last_completed_step > 0 or any(
            p.status == "in_progress" and p.last_processed_frame_index > 0
            for p in state.steps.values()
        )
        if not has_progress:
            return ResumeDecision(
                "fresh", 0, None,
                "checkpoint holds no progress yet — starting fresh",
            )
        if policy == "no":
            return ResumeDecision(
                "overwrite", 0, None,
                "progress discarded by --resume no — starting from scratch",
            )
        if policy == "ask":
            if interactive is None:
                try:
                    interactive = sys.stdin.isatty()
                except (AttributeError, ValueError):
                    interactive = False
            if interactive:
                choice = self._prompt_choice(state)
                if choice == "overwrite":
                    return ResumeDecision(
                        "overwrite", 0, None,
                        "progress discarded by user choice — starting from scratch",
                    )
            else:
                self.log.info(
                    "Non-interactive shell: auto-resuming checkpoint "
                    "(%s). Pass --resume no to start over.",
                    state.describe_progress(),
                )
        return ResumeDecision(
            "resume",
            state.last_completed_step,
            state,
            f"resuming: {state.describe_progress()}",
        )

    def _prompt_choice(self, state: PipelineState) -> ResumeAction:
        self.log.info("Found an unfinished run in %s:", self.workspace_root)
        self.log.info("  %s", state.describe_progress())
        self.log.info("  [1] Resume from where it stopped (recommended)")
        self.log.info("  [2] Discard progress and start from scratch")
        for _ in range(3):
            try:
                answer = input("Choice [1/2] (default 1): ").strip()
            except EOFError:
                self.log.info("No answer (EOF) — resuming.")
                return "resume"
            if answer in ("", "1", "resume"):
                return "resume"
            if answer in ("2", "overwrite"):
                return "overwrite"
            self.log.warning("Please answer 1 or 2.")
        self.log.info("No valid answer — resuming.")
        return "resume"

    def begin_run(self, snapshot: Dict[str, Any], from_step: int = 1) -> PipelineState:
        """Start tracking a fresh run (trusts explicit ``--from-step`` ranges:
        jumping in at step N implies steps ``< N`` are already done)."""
        state = PipelineState(
            workspace=str(self.workspace_root),
            snapshot=dict(snapshot),
            last_completed_step=max(0, from_step - 1),
        )
        state.save(self.state_path)
        self._state = state
        return state

    def attach(self, state: PipelineState) -> PipelineState:
        self._state = state
        return state

    # -- Recording ---------------------------------------------------------------------
    def record_progress(
        self,
        step_key: str,
        frame_index: int,
        target_count: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self._require_state()
        entry = state.steps.get(step_key) or StepProgress()
        entry.status = "in_progress"
        entry.last_processed_frame_index = int(frame_index)
        if target_count is not None:
            entry.target_count = target_count
        if params is not None:
            entry.params = {k: self._jsonable(v) for k, v in params.items()}
        entry.updated_at = _utcnow()
        state.steps[step_key] = entry
        state.save(self.state_path)

    def mark_step_complete(
        self, step_num: int, info: Optional[Dict[str, Any]] = None
    ) -> None:
        state = self._require_state()
        key = STEP_KEYS.get(step_num, f"step{step_num:02d}")
        entry = state.steps.get(key) or StepProgress()
        entry.status = "complete"
        if info:
            entry.params.update({k: self._jsonable(v) for k, v in info.items()})
        entry.updated_at = _utcnow()
        state.steps[key] = entry
        # Monotonic: a re-run step range must never move the frontier back.
        state.last_completed_step = max(state.last_completed_step, step_num)
        state.save(self.state_path)

    def refresh_snapshot(self, config: Any) -> None:
        """Re-snapshot after a step (picks up Step 2 probe metadata, ...)."""
        state = self._require_state()
        state.snapshot = self.snapshot_from_config(config)
        state.save(self.state_path)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    # -- Resume validation ---------------------------------------------------------------
    def check_step_params(
        self, step_key: str, params: Dict[str, Any]
    ) -> List[str]:
        """Diffs between checkpoint-recorded and current step params.

        Empty when the checkpoint has no entry for the step (nothing recorded
        yet) — callers then fall back to pure disk scanning.
        """
        state = self._require_state()
        entry = state.steps.get(step_key)
        if entry is None or not entry.params:
            return []
        diffs = []
        for key, current in params.items():
            stored = entry.params.get(key)
            if self._jsonable(current) != stored:
                diffs.append(f"{key}: checkpoint={stored!r} vs run={current!r}")
        return diffs

    def restore_metadata(self, config: Any, state: PipelineState) -> List[str]:
        """Copy Step 2 probe results from the snapshot into a fresh config.

        Used when resuming with a lost ``config.json``: the state file is the
        backup of ``original_fps``/``duration``/audio info. Only fills fields
        that are currently ``None``; returns the restored field names.
        """
        restored = []
        for key in (
            "original_fps", "duration_sec", "total_frames",
            "source_width", "source_height", "interpolation_factor",
            "has_audio", "extracted_frame_count", "frame_pattern",
        ):
            if getattr(config, key, None) is None and state.snapshot.get(key) is not None:
                setattr(config, key, state.snapshot[key])
                restored.append(key)
        if getattr(config, "audio_path", None) is None and state.snapshot.get("audio_path"):
            config.audio_path = Path(state.snapshot["audio_path"])
            restored.append("audio_path")
        return restored

    # -- Discard -------------------------------------------------------------------------------
    def discard_progress(self, from_step: int = 1) -> tuple:
        """Delete the state file + intermediate frames for steps >= from_step.

        Only fixed workspace-relative globs are ever removed (containment by
        construction); the final video and ``config.json`` are always kept.
        Returns ``(deleted_files, deleted_bytes)``.
        """
        deleted_files = 0
        deleted_bytes = 0
        try:
            if self.state_path.exists():
                deleted_bytes += self.state_path.stat().st_size
                self.state_path.unlink()
                deleted_files += 1
        except OSError as exc:
            self.log.warning("Could not delete %s: %s", self.state_path, exc)

        wipes: List[tuple] = []
        if from_step <= 2:
            wipes.append((DIR_RAW_FRAMES, "frame_*.*"))
        if from_step <= 3:
            wipes.append((DIR_INTERPOLATED, "frame_*.*"))
        if from_step <= 4:
            wipes.append((DIR_UPSCALED_8K, "frame_8k_*.*"))
        for dirname, glob in wipes:
            folder = self.workspace_root / dirname
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob(glob)):
                if not path.is_file():
                    continue
                try:
                    deleted_bytes += path.stat().st_size
                    path.unlink()
                    deleted_files += 1
                except OSError as exc:
                    self.log.warning("Could not delete %s: %s", path, exc)
        if from_step <= 2:
            for audio_name in (AUDIO_FILENAME_WAV, AUDIO_FILENAME_AAC):
                audio = self.workspace_root / audio_name
                if audio.is_file():
                    try:
                        deleted_bytes += audio.stat().st_size
                        audio.unlink()
                        deleted_files += 1
                    except OSError as exc:
                        self.log.warning("Could not delete %s: %s", audio, exc)
        self._state = None
        self.log.info(
            "Discarded progress from step %d: removed %d file(s) (%.2f GB).",
            from_step, deleted_files, deleted_bytes / (1024**3),
        )
        return deleted_files, deleted_bytes
