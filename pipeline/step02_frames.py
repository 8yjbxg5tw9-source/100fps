"""STEP 2 — Video analysis, lossless audio extraction, frame extraction.

Consumes the :class:`PipelineConfig` produced by Step 1 and prepares everything
Step 3 (RIFE interpolation to 1000 FPS) needs:

1. **Metadata probe** (FFprobe first, OpenCV fallback): resolution, original
   FPS, total frame count, duration. From the FPS it derives the
   ``interpolation_factor`` needed to reach the target (e.g. 30 -> 1000 FPS
   is a ~33.33x multiplier) and stores all of it back on the config object
   (typed attributes **and** the ``metadata`` dict).
2. **Audio extraction** (auxiliary): ``input_audio.wav`` (lossless, default)
   or ``input_audio.aac`` next to the workspace. Missing audio is *not* an
   error -- the log notes it and Step 8 will simply produce silent output.
3. **Frame extraction** (critical): every frame as lossless ``frame_%06d.png``
   (or high-quality JPEG) into ``temp_raw_frames``, with a ``tqdm`` progress
   bar fed by FFmpeg's ``-progress`` output.
4. **Validation**: extracted file count vs. probed total; mismatches (VFR
   sources, dropped packets, ...) produce a warning, not a crash.

Typical usage::

    from pipeline.config import PipelineConfig
    from pipeline.step02_frames import Step02Frames

    config = PipelineConfig.load("workspace/config.json")
    result = Step02Frames().run(config)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.base import PipelineStep
from pipeline.config import (
    AUDIO_FILENAME_AAC,
    AUDIO_FILENAME_WAV,
    CONFIG_FILENAME,
    FRAME_PATTERN_JPG,
    FRAME_PATTERN_PNG,
    PipelineConfig,
)
from pipeline.exceptions import (
    FFmpegNotFoundError,
    FrameExtractionError,
    InputVideoNotFoundError,
    MetadataProbeError,
)
from pipeline.logger import get_logger
from pipeline.step01_environment import FFMPEG_INSTALL_GUIDE, Step01Environment


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: Optional[int]      # None when neither probe could count
    duration_sec: Optional[float]
    video_codec: Optional[str] = None
    pix_fmt: Optional[str] = None
    probe_source: str = "ffprobe"    # "ffprobe" | "opencv"
    total_frames_estimated: bool = False  # True if derived, not read


@dataclass
class Step02Result:
    metadata: VideoMetadata
    interpolation_factor: float
    estimated_interpolated_frames: Optional[int]
    has_audio: bool
    audio_path: Optional[Path]
    frame_dir: Path
    frame_pattern: str
    extracted_frame_count: int
    validation_ok: bool


class _NullProgress:
    """Tiny tqdm stand-in used when tqdm is not installed."""

    def __init__(self, total: Optional[int] = None, desc: str = "") -> None:
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class Step02Frames(PipelineStep[Step02Result]):
    """Step 2 implementation. See module docstring for the full contract."""

    name = "step02_frames"

    IMAGE_FORMATS = ("png", "jpg")
    AUDIO_FORMATS = ("wav", "aac")

    def __init__(
        self,
        image_format: str = "png",
        jpeg_quality: int = 2,  # ffmpeg -q:v scale: 1 (best) .. 31
        audio_format: str = "wav",
        clean_frame_dir: bool = True,  # drop stale frame_*.* before extracting
        overwrite: bool = True,        # pass -y (else -n) to ffmpeg
        ffmpeg_bin: Optional[str] = None,   # override PATH lookup (tests/tools)
        ffprobe_bin: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if image_format not in self.IMAGE_FORMATS:
            raise ValueError(
                f"image_format must be one of {self.IMAGE_FORMATS}, "
                f"got {image_format!r}"
            )
        if audio_format not in self.AUDIO_FORMATS:
            raise ValueError(
                f"audio_format must be one of {self.AUDIO_FORMATS}, "
                f"got {audio_format!r}"
            )
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.audio_format = audio_format
        self.clean_frame_dir = clean_frame_dir
        self.overwrite = overwrite
        # Explicit overrides win immediately; PATH lookup happens in
        # _resolve_tools() for whatever is still None.
        self.ffmpeg: Optional[str] = ffmpeg_bin
        self.ffprobe: Optional[str] = ffprobe_bin
        self.log = logger or get_logger(__name__)

    # -- Orchestration -----------------------------------------------------------
    def run(self, config: PipelineConfig) -> Step02Result:
        self.log.info("=== Step 2: Video analysis & frame extraction started ===")

        if not config.input_video_path.is_file():
            raise InputVideoNotFoundError(
                f"Input video not found: {config.input_video_path}"
            )
        config.ensure_directories()
        self._resolve_tools(config)

        # 1. Metadata -------------------------------------------------------
        metadata = self.probe_metadata(config.input_video_path)
        factor = config.target_fps / metadata.fps
        if metadata.duration_sec:
            estimated = int(round(metadata.duration_sec * config.target_fps))
        elif metadata.total_frames:
            estimated = int(round(metadata.total_frames * factor))
        else:
            estimated = None

        config.source_width = metadata.width
        config.source_height = metadata.height
        config.original_fps = metadata.fps
        config.total_frames = metadata.total_frames
        config.duration_sec = metadata.duration_sec
        config.interpolation_factor = factor
        config.estimated_interpolated_frames = estimated
        config.metadata = {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "total_frames": metadata.total_frames,
            "total_frames_estimated": metadata.total_frames_estimated,
            "duration_sec": metadata.duration_sec,
            "video_codec": metadata.video_codec,
            "pix_fmt": metadata.pix_fmt,
            "probe_source": metadata.probe_source,
            "interpolation_factor": factor,
        }
        self.log.info(
            "Source: %dx%d @ %.2f FPS, %s frames%s -> target %.1f FPS "
            "(interpolation %.2fx, ~%s output frames)",
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.total_frames if metadata.total_frames is not None else "?",
            f", {metadata.duration_sec:.2f}s" if metadata.duration_sec else "",
            config.target_fps,
            factor,
            estimated if estimated is not None else "?",
        )

        # 2. Audio (auxiliary -- never raises) --------------------------------
        audio_path = self.extract_audio(config)
        config.has_audio = audio_path is not None
        config.audio_path = audio_path

        # 3. Frames (critical) ---------------------------------------------------
        pattern = FRAME_PATTERN_PNG if self.image_format == "png" else FRAME_PATTERN_JPG
        extracted = self.extract_frames(config, pattern)
        config.extracted_frame_count = extracted
        config.frame_pattern = pattern

        # 4. Validation --------------------------------------------------------------
        validation_ok = self.validate_frame_count(
            metadata.total_frames, extracted, metadata.total_frames_estimated
        )

        saved = config.save(config.workspace_root / CONFIG_FILENAME)
        self.log.info("Updated config saved for Step 3: %s", saved)
        self.log.info("Step 2 completed: %d raw frames ready for interpolation.", extracted)
        self.log.info("\n%s", config.summary())

        return Step02Result(
            metadata=metadata,
            interpolation_factor=factor,
            estimated_interpolated_frames=estimated,
            has_audio=config.has_audio,
            audio_path=audio_path,
            frame_dir=config.temp_raw_frames,
            frame_pattern=pattern,
            extracted_frame_count=extracted,
            validation_ok=validation_ok,
        )

    # -- Tool resolution --------------------------------------------------------------
    def _resolve_tools(self, config: PipelineConfig) -> None:
        self.ffmpeg = self.ffmpeg or shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise FFmpegNotFoundError(
                "FFmpeg is required for Step 2 (frame/audio extraction) but was "
                f"not found.\n{FFMPEG_INSTALL_GUIDE}"
            )
        self.ffprobe = self.ffprobe or shutil.which("ffprobe")
        config.ffmpeg_available = True
        config.ffmpeg_path = self.ffmpeg
        if not config.ffmpeg_version:
            config.ffmpeg_version = Step01Environment._read_ffmpeg_version(self.ffmpeg)
        self.log.info("Using ffmpeg: %s", self.ffmpeg)
        if self.ffprobe:
            self.log.info("Using ffprobe: %s", self.ffprobe)
        else:
            self.log.warning("ffprobe NOT found -- metadata falls back to OpenCV.")

    # -- 1. Metadata -----------------------------------------------------------------------
    def probe_metadata(self, video: Path) -> VideoMetadata:
        """Probe with FFprobe, fall back to OpenCV, else raise."""
        if self.ffprobe:
            metadata = self._probe_with_ffprobe(video)
            if metadata is not None:
                return metadata
            self.log.warning("FFprobe failed -- trying OpenCV fallback.")
        else:
            self.log.info("Skipping FFprobe (not installed) -- using OpenCV.")
        metadata = self._probe_with_opencv(video)
        if metadata is not None:
            return metadata
        raise MetadataProbeError(
            f"Could not read metadata from {video}: FFprobe unavailable/failed "
            "and OpenCV fallback failed too. Install ffprobe (comes with FFmpeg) "
            "or opencv-python ('pip install opencv-python'), and verify the file "
            "is a valid video."
        )

    def _probe_with_ffprobe(self, video: Path) -> Optional[VideoMetadata]:
        cmd = [
            self.ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,"
            "codec_name,pix_fmt:format=duration",
            "-of", "json",
            str(video),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.warning("ffprobe launch failed: %s", exc)
            return None
        if proc.returncode != 0:
            self.log.warning("ffprobe error: %s", proc.stderr.strip()[:300])
            return None
        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            self.log.warning("ffprobe returned invalid JSON.")
            return None
        streams = data.get("streams", [])
        if not streams:
            # File parsed fine but has no video stream (e.g. audio-only).
            raise MetadataProbeError(f"No video stream found in {video}.")
        return self._parse_ffprobe_stream(streams[0], data.get("format", {}))

    @staticmethod
    def _parse_ffprobe_stream(
        stream: Dict[str, Any], fmt: Dict[str, Any]
    ) -> Optional[VideoMetadata]:
        try:
            width = int(stream["width"])
            height = int(stream["height"])
        except (KeyError, TypeError, ValueError):
            return None
        fps = Step02Frames.parse_frame_rate(
            str(stream.get("avg_frame_rate", ""))
        ) or Step02Frames.parse_frame_rate(str(stream.get("r_frame_rate", "")))
        if width <= 0 or height <= 0 or not fps or fps <= 0:
            return None

        total: Optional[int] = None
        estimated = False
        nb_frames = str(stream.get("nb_frames", "") or "")
        if nb_frames.isdigit():
            total = int(nb_frames)

        duration: Optional[float] = None
        for candidate in (stream.get("duration"), fmt.get("duration")):
            try:
                duration = float(candidate)  # type: ignore[arg-type]
                if duration > 0:
                    break
            except (TypeError, ValueError):
                continue
        else:
            duration = None

        if total is None and duration:
            total = int(round(duration * fps))
            estimated = True

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total,
            duration_sec=duration,
            video_codec=stream.get("codec_name"),
            pix_fmt=stream.get("pix_fmt"),
            probe_source="ffprobe",
            total_frames_estimated=estimated,
        )

    @staticmethod
    def parse_frame_rate(value: str) -> Optional[float]:
        """Parse ffprobe rates like ``'30000/1001'`` or ``'30'`` to float."""
        value = (value or "").strip()
        if not value or value == "0/0":
            return None
        try:
            if "/" in value:
                num, den = value.split("/", 1)
                fps = float(num) / float(den)
            else:
                fps = float(value)
        except (ValueError, ZeroDivisionError):
            return None
        return fps if fps > 0 else None

    def _probe_with_opencv(self, video: Path) -> Optional[VideoMetadata]:
        try:
            import cv2  # lazy: opencv may be missing when auto-install is off
        except ImportError:
            self.log.warning("OpenCV (cv2) not importable -- cannot probe metadata.")
            return None
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                self.log.warning("OpenCV could not open %s.", video)
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width <= 0 or height <= 0 or fps <= 0:
            self.log.warning("OpenCV returned invalid properties for %s.", video)
            return None
        total = count if count > 0 else None
        duration = (total / fps) if total else None
        self.log.info(
            "Metadata via OpenCV: %dx%d @ %.2f FPS, %s frames (frame count via "
            "OpenCV can be approximate for some codecs).",
            width, height, fps, total if total is not None else "?",
        )
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total,
            duration_sec=duration,
            probe_source="opencv",
            total_frames_estimated=total is None,
        )

    # -- 2. Audio -------------------------------------------------------------------------------
    def has_audio_stream(self, video: Path) -> Optional[bool]:
        """True/False via ffprobe, or None when ffprobe is unavailable."""
        if not self.ffprobe:
            return None
        cmd = [
            self.ffprobe, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index,codec_name",
            "-of", "json",
            str(video),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        try:
            return len(json.loads(proc.stdout or "{}").get("streams", [])) > 0
        except json.JSONDecodeError:
            return None

    def extract_audio(self, config: PipelineConfig) -> Optional[Path]:
        """Extract audio to ``workspace/input_audio.{wav,aac}`` (never raises).

        Returns the audio path, or None when the video has no audio / extraction
        failed. Audio is auxiliary: Step 8 muxes it only if the file exists.
        """
        known = self.has_audio_stream(config.input_video_path)
        if known is False:
            self.log.info("No audio stream in input -- skipping audio extraction.")
            return None
        if known is None:
            self.log.info("Audio presence unknown (no ffprobe) -- attempting extraction.")

        filename = (
            AUDIO_FILENAME_WAV if self.audio_format == "wav" else AUDIO_FILENAME_AAC
        )
        target = config.workspace_root / filename
        cmd: List[str] = [
            self.ffmpeg, "-y" if self.overwrite else "-n",
            "-i", str(config.input_video_path),
            "-vn",
        ]
        if self.audio_format == "wav":
            cmd += ["-c:a", "pcm_s16le"]  # lossless
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(target))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            self.log.warning("Audio extraction failed to launch (%s) -- continuing "
                             "without audio.", exc)
            return None
        if proc.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            mb = target.stat().st_size / (1024 * 1024)
            self.log.info("Audio extracted: %s (%.2f MB, %s).", target, mb, self.audio_format)
            return target

        # Distinguish "no audio stream" from a genuine failure for a clear log.
        stderr = proc.stderr or ""
        if "does not contain any stream" in stderr or "Stream map" in stderr:
            self.log.info("No audio stream in input -- skipping audio extraction.")
        else:
            self.log.warning(
                "Audio extraction failed (non-critical, continuing without audio). "
                "ffmpeg said: %s",
                stderr.strip().splitlines()[-1][:300] if stderr.strip() else "unknown",
            )
        try:
            if target.is_file():
                target.unlink()  # remove partial/corrupt output
        except OSError:
            pass
        return None

    # -- 3. Frames ---------------------------------------------------------------------------------
    def extract_frames(self, config: PipelineConfig, pattern: str) -> int:
        """Extract all frames with a tqdm progress bar; return file count."""
        frame_dir = config.temp_raw_frames
        frame_dir.mkdir(parents=True, exist_ok=True)

        if self.clean_frame_dir:
            stale = [p for p in frame_dir.glob("frame_*.*") if p.is_file()]
            for p in stale:
                try:
                    p.unlink()
                except OSError:
                    pass
            if stale:
                self.log.info("Removed %d stale frame(s) from %s.", len(stale), frame_dir)

        output = str(frame_dir / pattern)
        cmd: List[str] = [
            self.ffmpeg, "-y" if self.overwrite else "-n",
            "-i", str(config.input_video_path),
            "-vsync", "passthrough",  # keep every frame incl. VFR duplicates
        ]
        if self.image_format == "jpg":
            cmd += ["-q:v", str(self.jpeg_quality)]
        cmd += ["-progress", "pipe:1", "-nostats", output]

        total = config.total_frames
        self.log.info("Extracting frames -> %s ...", output)
        bar = self._make_progress_bar(total, desc="Frames")
        err_log = config.workspace_root / "ffmpeg_frames.log"
        try:
            with open(err_log, "w", encoding="utf-8") as err_fh:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=err_fh,
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
                        elif line == "progress=end":
                            break
                proc.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            raise FrameExtractionError(f"Failed to launch ffmpeg: {exc}") from exc

        if proc.returncode != 0:
            tail = self._tail(err_log)
            raise FrameExtractionError(
                f"ffmpeg frame extraction failed (exit {proc.returncode}). "
                f"Full log: {err_log}\nLast lines:\n{tail}"
            )

        count = self._count_frame_files(frame_dir, self.image_format)
        self.log.info("Extracted %d frame(s) -> %s", count, frame_dir)
        return count

    @staticmethod
    def _count_frame_files(frame_dir: Path, ext: str) -> int:
        return sum(1 for p in frame_dir.glob(f"frame_*.{ext}") if p.is_file())

    @staticmethod
    def _tail(path: Path, lines: int = 10) -> str:
        try:
            return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
        except OSError:
            return "(unreadable)"

    def _make_progress_bar(self, total: Optional[int], desc: str = "") -> Any:
        try:
            from tqdm import tqdm
        except ImportError:
            self.log.warning("tqdm not installed -- progress bar disabled.")
            return _NullProgress(total=total, desc=desc)
        return tqdm(total=total, desc=desc, unit="frame")

    # -- 4. Validation -----------------------------------------------------------------------------
    def validate_frame_count(
        self, expected: Optional[int], actual: int, estimated: bool = False
    ) -> bool:
        """Compare extracted files vs. probed total; warn (don't raise)."""
        if expected is None:
            self.log.info(
                "Original frame count unknown -- extracted %d frame(s), "
                "skipping count validation.", actual,
            )
            return True
        if actual == expected:
            self.log.info("Validation OK: extracted %d/%d frames.", actual, expected)
            return True
        self.log.warning(
            "Frame count MISMATCH: extracted %d but metadata reported %d%s. "
            "Possible causes: VFR source, dropped/corrupt packets, or leftover "
            "files. Step 3 will interpolate whatever frames exist.",
            actual, expected, " (estimated)" if estimated else "",
        )
        return False
