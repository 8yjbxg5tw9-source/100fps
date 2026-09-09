"""STEP 5 — Assemble the 8K @ target-FPS video with FFmpeg (+ audio mux).

Reads the 8K frames from Step 4 (``upscaled_8k/frame_8k_%08d.*``) and the
original audio from Step 2 (``input_audio.wav``/``.aac``), encodes a single
high-quality file (``final_output_path``) and verifies it:

1. **Dynamic command**: image-sequence input at exactly ``target_fps``
   (``-framerate`` + ``-r``), optional ``-c:a copy`` audio mux (gracefully
   skipped when the source had no audio), ``+faststart`` for MP4.
2. **Codec**: HEVC by default — ``hevc_nvenc`` on CUDA machines
   (``-preset p6 -tune hq -rc vbr -cq <crf>``), ``libx265`` otherwise
   (``-crf <crf> -preset medium``); AV1 variants opt-in. Always
   ``yuv420p`` for player compatibility. Encoder availability is probed in
   the local FFmpeg build with automatic fallback.
3. **Streamed encode**: frames stream from disk through FFmpeg (never fully
   loaded into RAM); ``-progress`` output drives a ``tqdm`` bar.
4. **Verification**: resolution + FPS are confirmed via FFprobe (OpenCV
   fallback); mismatches raise, count/duration drift warns.

Typical usage::

    from pipeline.config import PipelineConfig
    from pipeline.step05_assemble import Step05Assemble

    config = PipelineConfig.load("workspace/config.json")
    result = Step05Assemble(video_codec="auto", crf=19).run(config)
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.base import PipelineStep
from pipeline.config import CONFIG_FILENAME, PipelineConfig
from pipeline.exceptions import (
    AssemblyError,
    FFmpegNotFoundError,
    OutputVerificationError,
)
from pipeline.logger import get_logger
from pipeline.step01_environment import FFMPEG_INSTALL_GUIDE


@dataclass
class Step05Result:
    output_path: Path
    encoder: str
    encoded_frames: int
    fps: float
    resolution: tuple  # (width, height)
    has_audio: bool
    file_size_mb: float
    verified: bool


class Step05Assemble(PipelineStep[Step05Result]):
    """Step 5 implementation. See module docstring for the full contract."""

    name = "step05_assemble"

    CODECS = ("auto", "hevc_nvenc", "libx265", "av1_nvenc", "libsvtav1")
    #: Preference order for ``video_codec="auto"`` on CUDA machines.
    AUTO_CUDA_ORDER = ("hevc_nvenc", "libx265")
    DEFAULT_CRF = 19  # visually lossless per spec (17-20)

    def __init__(
        self,
        video_codec: str = "auto",
        crf: float = DEFAULT_CRF,
        encoder_preset: Optional[str] = None,  # override codec default preset
        overwrite: bool = True,
        ffmpeg_args: Optional[str] = None,  # extra raw args (shlex-split)
        verify: bool = True,
        ffmpeg_bin: Optional[str] = None,
        ffprobe_bin: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if video_codec not in self.CODECS:
            raise ValueError(
                f"video_codec must be one of {self.CODECS}, got {video_codec!r}."
            )
        if not 0 <= crf <= 63:
            raise ValueError(f"crf must be in 0..63, got {crf}.")
        self.video_codec = video_codec
        self.crf = crf
        self.encoder_preset = encoder_preset
        self.overwrite = overwrite
        self.ffmpeg_args = ffmpeg_args
        self.verify = verify
        self.ffmpeg: Optional[str] = ffmpeg_bin
        self.ffprobe: Optional[str] = ffprobe_bin
        self.log = logger or get_logger(__name__)
        self.last_command: List[str] = []

    # -- Orchestration -----------------------------------------------------------
    def run(self, config: PipelineConfig) -> Step05Result:
        self.log.info("=== Step 5: Assembling %s started ===", config.final_output_path)
        sources = self._discover_sources(config)
        self._resolve_tools(config)
        assert self.ffmpeg is not None

        audio = self._resolve_audio(config)
        encoder = self._select_encoder(config)
        self._warn_if_low_memory(encoder)
        command = self.build_command(config, encoder, audio, len(sources))
        self.last_command = command
        self.log.info("FFmpeg command:\n  %s", " ".join(command))

        encoded = self._run_encode(command, len(sources), config)
        output = config.final_output_path
        size_mb = output.stat().st_size / (1024 * 1024)
        self.log.info("Encoded %s (%.1f MB, %s).", output, size_mb, encoder)

        verified = self._verify_output(config, output, len(sources), audio is not None)

        config.assembly_codec = encoder
        config.assembled_frame_count = encoded
        config.assembly_verified = verified
        config.assembly_file_size_mb = size_mb
        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 6: %s", saved)
        self.log.info(
            "Step 5 completed: %s (%dx%d @ %.1f FPS, %s).",
            output, config.target_width, config.target_height,
            config.target_fps, "verified" if verified else "UNVERIFIED",
        )
        self.log.info("\n%s", config.summary())
        return Step05Result(
            output_path=output,
            encoder=encoder,
            encoded_frames=encoded,
            fps=config.target_fps,
            resolution=(config.target_width, config.target_height),
            has_audio=audio is not None,
            file_size_mb=size_mb,
            verified=verified,
        )

    # -- Discovery / tools ---------------------------------------------------------------
    def _discover_sources(self, config: PipelineConfig) -> List[Path]:
        sources = sorted(
            p for p in config.upscaled_8k.glob("frame_8k_*.*") if p.is_file()
        )
        if not sources:
            raise AssemblyError(
                f"No 8K frames (frame_8k_*) in {config.upscaled_8k}. Run Step 4 first."
            )
        suffixes = {p.suffix.lower() for p in sources}
        if len(suffixes) > 1:
            self.log.warning(
                "Mixed 8K frame extensions %s -- using '%s' only.",
                sorted(suffixes), config.upscaled_frame_pattern or "first",
            )
        self.log.info("Discovered %d 8K frame(s) in %s.", len(sources), config.upscaled_8k)
        return sources

    def _resolve_tools(self, config: PipelineConfig) -> None:
        self.ffmpeg = self.ffmpeg or shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise FFmpegNotFoundError(
                "FFmpeg is required for Step 5 (video assembly) but was not "
                f"found.\n{FFMPEG_INSTALL_GUIDE}"
            )
        self.ffprobe = self.ffprobe or shutil.which("ffprobe")
        config.ffmpeg_available = True
        config.ffmpeg_path = self.ffmpeg
        if self.ffprobe:
            self.log.info("Using ffprobe: %s", self.ffprobe)
        else:
            self.log.warning("ffprobe NOT found -- verification falls back to OpenCV.")

    def _resolve_audio(self, config: PipelineConfig) -> Optional[Path]:
        audio = config.audio_path
        if config.has_audio and audio is not None and Path(audio).is_file():
            self.log.info("Muxing original audio: %s", audio)
            return Path(audio)
        self.log.info("No audio to mux (video-only output).")
        return None

    # -- Encoder selection -------------------------------------------------------------------
    def probe_encoder(self, encoder: str) -> bool:
        """Check whether the local FFmpeg build supports ``encoder``."""
        assert self.ffmpeg is not None
        try:
            proc = subprocess.run(
                [self.ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0 and "Unknown encoder" not in (proc.stderr or "")

    def _select_encoder(self, config: PipelineConfig) -> str:
        if self.video_codec != "auto":
            if not self.probe_encoder(self.video_codec):
                raise AssemblyError(
                    f"Encoder '{self.video_codec}' is not available in {self.ffmpeg}. "
                    f"Use --video-codec auto (recommended) or install an FFmpeg "
                    f"build with that encoder."
                )
            self.log.info("Using explicitly requested encoder: %s", self.video_codec)
            return self.video_codec
        candidates = (
            self.AUTO_CUDA_ORDER if config.device == "cuda" else ("libx265",)
        )
        for encoder in candidates:
            if self.probe_encoder(encoder):
                self.log.info("Auto-selected encoder: %s", encoder)
                return encoder
            self.log.warning("Encoder %s not in this FFmpeg build -- trying next.", encoder)
        raise AssemblyError(
            f"None of {candidates} is available in {self.ffmpeg}. Install a "
            f"fuller FFmpeg build (https://ffmpeg.org/download.html)."
        )

    # -- Low-RAM guardrail -----------------------------------------------------------------------
    #: Software 8K HEVC/AV1 encoders need several GB of working set; below this
    #: we warn (8K on a 4 GB box is reliably OOM-killed — verified in testing).
    LOW_RAM_WARN_GB = 8.0
    SOFTWARE_ENCODERS = ("libx265", "libsvtav1")

    @staticmethod
    def _total_ram_gb() -> Optional[float]:
        """Best-effort total system RAM (None when undetectable)."""
        try:
            if os.name == "posix" and hasattr(os, "sysconf"):
                pages = os.sysconf("SC_PHYS_PAGES")
                size = os.sysconf("SC_PAGE_SIZE")
                if pages > 0 and size > 0:
                    return pages * size / (1024**3)
        except (OSError, ValueError):
            pass
        return None

    def _warn_if_low_memory(self, encoder: str) -> None:
        if encoder not in self.SOFTWARE_ENCODERS:
            return
        total = self._total_ram_gb()
        if total is not None and total < self.LOW_RAM_WARN_GB:
            self.log.warning(
                "Only %.1f GiB system RAM detected -- software 8K %s encoding "
                "may be OOM-killed (exit -9). If it fails: use GPU encoding "
                "(--video-codec hevc_nvenc), an .mkv target, or lean settings "
                "via --ffmpeg-args \"-x265-params pools=1:frame-threads=1:"
                "rc-lookahead=10\".",
                total, encoder,
            )

    # -- Command construction ------------------------------------------------------------------
    @staticmethod
    def _fps_token(fps: float) -> str:
        return str(int(fps)) if float(fps).is_integer() else str(fps)

    def _codec_args(self, encoder: str) -> List[str]:
        crf = self._fps_token(self.crf)
        if encoder == "libx265":
            return ["-c:v", "libx265", "-preset", self.encoder_preset or "medium",
                    "-crf", crf, "-pix_fmt", "yuv420p"]
        if encoder == "hevc_nvenc":
            return ["-c:v", "hevc_nvenc", "-preset", self.encoder_preset or "p6",
                    "-tune", "hq", "-rc", "vbr", "-cq", crf, "-pix_fmt", "yuv420p"]
        if encoder == "libsvtav1":
            return ["-c:v", "libsvtav1", "-preset", self.encoder_preset or "6",
                    "-crf", crf, "-pix_fmt", "yuv420p"]
        if encoder == "av1_nvenc":
            return ["-c:v", "av1_nvenc", "-preset", self.encoder_preset or "p6",
                    "-tune", "hq", "-rc", "vbr", "-cq", crf, "-pix_fmt", "yuv420p"]
        raise AssemblyError(f"Unsupported encoder: {encoder}.")

    def build_command(
        self,
        config: PipelineConfig,
        encoder: str,
        audio: Optional[Path],
        num_frames: Optional[int] = None,
    ) -> List[str]:
        assert self.ffmpeg is not None
        pattern = config.upscaled_frame_pattern or "frame_8k_%08d.png"
        input_path = str(config.upscaled_8k / pattern)
        fps = self._fps_token(config.target_fps)
        cmd: List[str] = [
            self.ffmpeg, "-y" if self.overwrite else "-n",
            "-framerate", fps,
            "-start_number", "1",
            "-i", input_path,
        ]
        if audio is not None:
            cmd += ["-i", str(audio)]
        cmd += self._codec_args(encoder)
        cmd += ["-r", fps]  # force exact CFR pacing on the output
        if audio is not None:
            # Trim audio to the exact video duration (packet-precise for PCM).
            # NOTE: `-shortest` is deliberately NOT used: on very short clips
            # it can drop the audio stream entirely (verified with FFmpeg 7).
            frames = num_frames or config.upscaled_frame_count
            if not frames:
                raise AssemblyError(
                    "Cannot compute audio trim duration: frame count unknown."
                )
            duration = frames / float(config.target_fps)
            cmd += ["-c:a", "copy", "-t", f"{duration:.6f}"]
        if config.final_output_path.suffix.lower() == ".mp4":
            cmd += ["-movflags", "+faststart"]
        if self.ffmpeg_args:
            cmd += shlex.split(self.ffmpeg_args)
        cmd += ["-progress", "pipe:1", "-nostats", str(config.final_output_path)]
        return cmd

    # -- Encode with progress -----------------------------------------------------------------------
    def _run_encode(self, command: List[str], total_frames: int, config: PipelineConfig) -> int:
        err_log = config.workspace_root / "ffmpeg_assemble.log"
        bar = self._make_progress_bar(total_frames, desc="Encoding")
        encoded = 0
        try:
            with open(err_log, "w", encoding="utf-8") as err_fh:
                proc = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=err_fh,
                    text=True, bufsize=1,
                )
                last = 0
                with bar:
                    for line in proc.stdout or []:
                        line = line.strip()
                        if line.startswith("frame="):
                            try:
                                current = int(line.split("=", 1)[1].strip())
                            except ValueError:
                                continue
                            bar.update(max(0, current - last))
                            last = current
                            encoded = current
                        elif line == "progress=end":
                            break
                proc.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            raise AssemblyError(f"Failed to launch ffmpeg: {exc}") from exc
        if proc.returncode != 0:
            raise AssemblyError(
                f"FFmpeg assembly failed (exit {proc.returncode}). "
                f"Full log: {err_log}\nLast lines:\n{self._tail(err_log)}"
            )
        if not config.final_output_path.is_file():
            raise AssemblyError(
                f"FFmpeg exited 0 but {config.final_output_path} is missing. Log: {err_log}"
            )
        self.log.info("Encode progress: %d/%d frames.", encoded or total_frames, total_frames)
        return encoded or total_frames

    @staticmethod
    def _tail(path: Path, lines: int = 10) -> str:
        try:
            return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
        except OSError:
            return "(unreadable)"

    def _make_progress_bar(self, total: int, desc: str = "") -> Any:
        try:
            from tqdm import tqdm
        except ImportError:
            self.log.warning("tqdm not installed -- progress bar disabled.")
            return _NullProgress()
        return tqdm(total=total, desc=desc, unit="frame")

    # -- Verification ----------------------------------------------------------------------------------
    def _verify_output(
        self, config: PipelineConfig, output: Path, expected_frames: int, expect_audio: bool
    ) -> bool:
        if not self.verify:
            self.log.warning("Verification disabled (--skip-verify).")
            return False
        props = self._probe_ffprobe(output) if self.ffprobe else None
        source = "ffprobe"
        if props is None:
            props = self._probe_opencv(output)
            source = "opencv"
        if props is None:
            self.log.warning(
                "Could not verify %s (no ffprobe, OpenCV fallback failed). "
                "Output exists but is UNVERIFIED.",
                output,
            )
            return False
        self.log.info(
            "Verification via %s: %dx%d @ %.2f FPS, %s frames%s.",
            source, props["width"], props["height"], props["fps"],
            props["frames"] if props["frames"] is not None else "?",
            f", {props['duration']:.2f}s" if props.get("duration") else "",
        )
        errors: List[str] = []
        if (props["width"], props["height"]) != (config.target_width, config.target_height):
            errors.append(
                f"resolution is {props['width']}x{props['height']}, "
                f"expected {config.target_width}x{config.target_height}"
            )
        if abs(props["fps"] - config.target_fps) > max(0.5, config.target_fps * 0.01):
            errors.append(
                f"FPS is {props['fps']:.2f}, expected {config.target_fps}"
            )
        if errors:
            raise OutputVerificationError(
                f"Output verification FAILED for {output} ({source}): "
                + "; ".join(errors)
                + ". The file was still written -- inspect ffmpeg_assemble.log."
            )
        if props["frames"] is not None and abs(props["frames"] - expected_frames) > max(1, expected_frames // 100):
            self.log.warning(
                "Frame count drift: container reports %d, encoded %d.",
                props["frames"], expected_frames,
            )
        if expect_audio and not props.get("has_audio", True):
            self.log.warning("Expected audio track is missing from %s.", output)
        self.log.info("Verification PASSED: %s is correct.", output.name)
        return True

    def _probe_ffprobe(self, output: Path) -> Optional[Dict[str, Any]]:
        cmd = [
            self.ffprobe, "-v", "error",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,codec_type:format=duration",
            "-of", "json", str(output),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return None
        video = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
        )
        if video is None:
            return None
        fps = self._parse_rate(str(video.get("avg_frame_rate", "")))
        if not fps:
            return None
        try:
            width, height = int(video["width"]), int(video["height"])
        except (KeyError, TypeError, ValueError):
            return None
        nb = str(video.get("nb_frames", "") or "")
        frames = int(nb) if nb.isdigit() else None
        duration = None
        try:
            duration = float(data.get("format", {}).get("duration"))
        except (TypeError, ValueError):
            pass
        has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
        return {"width": width, "height": height, "fps": fps,
                "frames": frames, "duration": duration, "has_audio": has_audio}

    def _probe_opencv(self, output: Path) -> Optional[Dict[str, Any]]:
        try:
            import cv2
        except ImportError:
            return None
        cap = cv2.VideoCapture(str(output))
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width <= 0 or height <= 0 or fps <= 0:
            return None
        return {"width": width, "height": height, "fps": fps,
                "frames": count or None,
                "duration": (count / fps) if count else None,
                "has_audio": True}  # cv2 can't see audio: don't warn falsely

    @staticmethod
    def _parse_rate(value: str) -> Optional[float]:
        value = (value or "").strip()
        if not value or value == "0/0":
            return None
        try:
            fps = (
                float(value.split("/", 1)[0]) / float(value.split("/", 1)[1])
                if "/" in value else float(value)
            )
        except (ValueError, ZeroDivisionError):
            return None
        return fps if fps > 0 else None


class _NullProgress:
    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
