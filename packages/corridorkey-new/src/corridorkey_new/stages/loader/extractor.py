"""Stage 1 - video extraction.

Extracts video files to PNG image sequences using PyAV (Python bindings to
FFmpeg C libraries). PyAV bundles its own FFmpeg — no system install required.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import av
import cv2
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v"})


class VideoMetadata(BaseModel):
    """Source video metadata captured at extraction time.

    Carried through the pipeline so stage 6 can re-encode output with
    matching properties.

    Attributes:
        filename: Original video filename (stem + suffix).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps_num: Framerate numerator.
        fps_den: Framerate denominator.
        pix_fmt: Pixel format string (e.g. "yuv420p").
        codec_name: Video codec name (e.g. "h264", "prores").
        duration_s: Total duration in seconds. None if not reported by container.
        has_audio: True if the source container has at least one audio stream.
        color_space: Color space string (e.g. "bt709"). None if not reported.
        color_transfer: Transfer characteristic (e.g. "bt709"). None if not reported.
        color_primaries: Color primaries (e.g. "bt709"). None if not reported.
    """

    filename: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    pix_fmt: str
    codec_name: str
    duration_s: float | None = None
    has_audio: bool = False
    color_space: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None

    @property
    def fps(self) -> float:
        """Framerate as a float."""
        return self.fps_num / self.fps_den if self.fps_den else 0.0


def is_video(path: Path) -> bool:
    """Check if a path points to a video file."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def read_video_metadata(video_path: Path) -> VideoMetadata:
    """Read metadata from a video file without extracting frames.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata populated from the container and video stream.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    try:
        container = av.open(str(video_path))
    except av.FFmpegError as e:
        raise RuntimeError(f"Cannot open video '{video_path}': {e}") from e

    try:
        return _extract_metadata(container, video_path)
    finally:
        container.close()


def extract_video(
    video_path: Path,
    output_dir: Path,
    pattern: str = "frame_{:06d}.png",
    on_frame: Callable[[int, int], None] | None = None,
) -> VideoMetadata:
    """Extract a video file to a PNG image sequence using PyAV.

    Frames are written as lossless PNGs. The output filenames follow
    ``pattern`` (default: ``frame_000000.png``, ``frame_000001.png``, ...).

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into (created if needed).
        pattern: Python format string for frame filenames. Must contain one
            integer placeholder (e.g. ``"frame_{:06d}.png"``).
        on_frame: Optional callback(frame_index, total_frames) called after
            each frame is written. total_frames is 0 when the container does
            not report a frame count.

    Returns:
        VideoMetadata captured from the source container.

    Raises:
        RuntimeError: If the video cannot be opened or a frame cannot be decoded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting video: %s -> %s", video_path, output_dir)

    try:
        container = av.open(str(video_path))
    except av.FFmpegError as e:
        raise RuntimeError(f"Cannot open video '{video_path}': {e}") from e

    stream = container.streams.video[0]
    stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO

    metadata = _extract_metadata(container, video_path)

    # Best-effort total frame count from container metadata.
    total_frames = stream.frames or 0

    frame_index = 0
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                bgr = frame.to_ndarray(format="bgr24")
                out_path = output_dir / pattern.format(frame_index)
                cv2.imwrite(str(out_path), bgr)
                if on_frame:
                    on_frame(frame_index, total_frames)
                frame_index += 1
    except av.FFmpegError as e:
        raise RuntimeError(f"Error decoding '{video_path}' at frame {frame_index}: {e}") from e
    finally:
        container.close()

    logger.info("Extracted %d frames to %s", frame_index, output_dir)
    return metadata


def save_video_metadata(metadata: VideoMetadata, clip_root: Path) -> Path:
    """Write VideoMetadata as JSON into the clip root directory.

    Args:
        metadata: Metadata to serialise.
        clip_root: Clip root directory (the folder containing Input/, Output/, etc.).

    Returns:
        Path to the written JSON file.
    """
    out_path = clip_root / "video_meta.json"
    out_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved video metadata: %s", out_path)
    return out_path


def load_video_metadata(clip_root: Path) -> VideoMetadata | None:
    """Read VideoMetadata from the clip root directory, if present.

    Args:
        clip_root: Clip root directory.

    Returns:
        VideoMetadata if video_meta.json exists, None otherwise.
    """
    meta_path = clip_root / "video_meta.json"
    if not meta_path.exists():
        return None
    return VideoMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_metadata(container: av.container.InputContainer, video_path: Path) -> VideoMetadata:
    """Pull metadata from an open PyAV container."""
    stream = container.streams.video[0]
    ctx = stream.codec_context

    rate = stream.average_rate or stream.base_rate or stream.guessed_rate
    fps_num = int(rate.numerator) if rate else 0
    fps_den = int(rate.denominator) if rate else 1

    duration_s: float | None = None
    if container.duration is not None:
        duration_s = float(container.duration) / av.time_base

    return VideoMetadata(
        filename=video_path.name,
        width=ctx.width,
        height=ctx.height,
        fps_num=fps_num,
        fps_den=fps_den,
        pix_fmt=ctx.pix_fmt or "unknown",
        codec_name=ctx.name or "unknown",
        duration_s=duration_s,
        has_audio=len(container.streams.audio) > 0,
        color_space=str(ctx.colorspace) if ctx.colorspace else None,
        color_transfer=str(ctx.color_trc) if ctx.color_trc else None,
        color_primaries=str(ctx.color_primaries) if ctx.color_primaries else None,
    )
