"""Typed exceptions for the CorridorKey pipeline.

All exceptions inherit from CorridorKeyError so callers can catch the
base class when they don't need to distinguish between subtypes.

Hierarchy
---------
CorridorKeyError
├── ClipScanError               — scanner: path/structure problems
├── ExtractionError             — loader: video extraction failures
├── FrameMismatchError          — loader: input/alpha count mismatch
├── FrameReadError              — preprocessor: frame file unreadable
├── WriteFailureError           — writer: cv2.imwrite failure
├── VRAMInsufficientError       — inference: not enough GPU memory
├── DeviceError                 — infra: requested device unavailable
├── ModelError                  — infra: model download or checksum failure
├── InvalidStateTransitionError — clip state machine: illegal transition
└── JobCancelledError           — pipeline: job cancelled by the caller
"""

from __future__ import annotations


class CorridorKeyError(Exception):
    """Base exception for all CorridorKey errors."""


class ClipScanError(CorridorKeyError):
    """Raised when a clip path cannot be scanned or has an unrecognised structure."""


class ExtractionError(CorridorKeyError):
    """Raised when video frame extraction fails."""

    def __init__(self, clip_name: str, detail: str) -> None:
        self.clip_name = clip_name
        self.detail = detail
        super().__init__(f"Clip '{clip_name}': extraction failed — {detail}")


class FrameMismatchError(CorridorKeyError):
    """Raised when input and alpha frame counts do not match."""

    def __init__(self, clip_name: str, input_count: int, alpha_count: int) -> None:
        self.clip_name = clip_name
        self.input_count = input_count
        self.alpha_count = alpha_count
        super().__init__(
            f"Clip '{clip_name}': frame count mismatch — {input_count} input frames vs {alpha_count} alpha frames"
        )


class FrameReadError(CorridorKeyError):
    """Raised when a frame file cannot be read or decoded."""


class WriteFailureError(CorridorKeyError):
    """Raised when a write operation fails."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"cv2.imwrite failed: {path}")


class VRAMInsufficientError(CorridorKeyError):
    """Raised when there is not enough GPU VRAM for the requested operation."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(f"Insufficient VRAM: {required_gb:.1f} GB required, {available_gb:.1f} GB available")


class DeviceError(CorridorKeyError):
    """Raised when the requested compute device is unavailable."""


class ModelError(CorridorKeyError):
    """Raised when a model cannot be downloaded or fails checksum verification."""


class InvalidStateTransitionError(CorridorKeyError):
    """Raised when a clip state transition is not permitted by the state machine."""

    def __init__(self, clip_name: str, current_state: str, target_state: str) -> None:
        self.clip_name = clip_name
        self.current_state = current_state
        self.target_state = target_state
        super().__init__(f"Clip '{clip_name}': invalid state transition {current_state} -> {target_state}")


class JobCancelledError(CorridorKeyError):
    """Raised when a pipeline job is cancelled by the caller."""

    def __init__(self, clip_name: str, frame_index: int | None = None) -> None:
        self.clip_name = clip_name
        self.frame_index = frame_index
        msg = f"Clip '{clip_name}': job cancelled"
        if frame_index is not None:
            msg += f" at frame {frame_index}"
        super().__init__(msg)
