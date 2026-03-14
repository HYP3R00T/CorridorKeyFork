"""Typed exceptions for the CorridorKey pipeline.

All exceptions inherit from CorridorKeyError so callers can catch the
base class when they don't need to distinguish between subtypes.
"""

import sys


class CorridorKeyError(Exception):
    """Base exception for all CorridorKey errors."""

    pass


class ClipScanError(CorridorKeyError):
    """Raised when a clip directory cannot be scanned or is malformed."""

    pass


class FrameMismatchError(CorridorKeyError):
    """Raised when input and alpha frame counts do not match."""

    def __init__(self, clip_name: str, input_count: int, alpha_count: int):
        self.clip_name = clip_name
        self.input_count = input_count
        self.alpha_count = alpha_count
        super().__init__(f"Clip '{clip_name}': frame count mismatch - input has {input_count}, alpha has {alpha_count}")


class FrameReadError(CorridorKeyError):
    """Raised when a frame file cannot be read."""

    def __init__(self, clip_name: str, frame_index: int, path: str):
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.path = path
        super().__init__(f"Clip '{clip_name}': failed to read frame {frame_index} ({path})")


class WriteFailureError(CorridorKeyError):
    """Raised when a write operation fails."""

    def __init__(self, clip_name: str, frame_index: int, path: str):
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.path = path
        super().__init__(f"Clip '{clip_name}': failed to write frame {frame_index} ({path})")


class MaskChannelError(CorridorKeyError):
    """Raised when a mask has an unexpected channel count."""

    def __init__(self, clip_name: str, frame_index: int, channels: int):
        self.clip_name = clip_name
        self.frame_index = frame_index
        self.channels = channels
        super().__init__(f"Clip '{clip_name}': mask frame {frame_index} has {channels} channels, expected 1 or 3+")


class VRAMInsufficientError(CorridorKeyError):
    """Raised when there is not enough GPU VRAM for the requested operation."""

    def __init__(self, required_gb: float, available_gb: float):
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(f"Insufficient VRAM: {required_gb:.1f}GB required, {available_gb:.1f}GB available")


class InvalidStateTransitionError(CorridorKeyError):
    """Raised when a clip state transition is not allowed."""

    def __init__(self, clip_name: str, current_state: str, target_state: str):
        self.clip_name = clip_name
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(f"Clip '{clip_name}': invalid state transition {current_state} -> {target_state}")


class JobCancelledError(CorridorKeyError):
    """Raised when a GPU job is cancelled by the user."""

    def __init__(self, clip_name: str, frame_index: int | None = None):
        self.clip_name = clip_name
        self.frame_index = frame_index
        msg = f"Clip '{clip_name}': job cancelled"
        if frame_index is not None:
            msg += f" at frame {frame_index}"
        super().__init__(msg)


class FFmpegNotFoundError(CorridorKeyError):
    """Raised when FFmpeg/FFprobe binaries cannot be located.

    The error message includes platform-specific install instructions so
    users know exactly what to do without consulting documentation.
    """

    def __init__(self, binary: str = "ffmpeg") -> None:
        if sys.platform == "darwin":
            hint = "brew install ffmpeg"
        elif sys.platform.startswith("linux"):
            hint = "sudo apt install ffmpeg  # or: sudo dnf install ffmpeg"
        else:
            hint = "choco install ffmpeg  # or: winget install ffmpeg"
        super().__init__(
            f"{binary} not found on PATH.\n"
            f"Install FFmpeg and ensure it is available system-wide:\n"
            f"  {hint}\n"
            f"Then restart your terminal and try again."
        )


class ExtractionError(CorridorKeyError):
    """Raised when video frame extraction fails."""

    def __init__(self, clip_name: str, detail: str):
        self.clip_name = clip_name
        self.detail = detail
        super().__init__(f"Clip '{clip_name}': extraction failed - {detail}")
