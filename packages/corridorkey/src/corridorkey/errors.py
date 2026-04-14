"""Typed exceptions for the CorridorKey pipeline.

All exceptions inherit from CorridorKeyError so callers can catch the
base class when they don't need to distinguish between subtypes.

Hierarchy
---------
CorridorKeyError
├── EngineError                 — engine: contract violations before processing starts
├── AlphaGeneratorError         — engine: alpha slot unfilled or bad return
├── ClipScanError               — scanner: path/structure problems, permission errors
├── ClipLoadError               — loader: empty input, output dir creation failure
├── ExtractionError             — loader: video extraction failures
├── FrameMismatchError          — loader: input/alpha count mismatch
├── FrameReadError              — preprocessor: frame file unreadable
├── InferenceError              — inference: model forward pass failure, OOM
├── PostprocessError            — postprocessor: despill/despeckle/composite failure
├── WriteFailureError           — writer: write failure (cv2 or OS)
├── VRAMInsufficientError       — inference: not enough GPU memory (CUDA OOM)
├── DeviceError                 — infra: requested device unavailable
├── ModelError                  — infra: model download, load, or checksum failure
├── InvalidStateTransitionError — clip state machine: illegal transition
└── JobCancelledError           — pipeline: job cancelled by the caller
"""

from __future__ import annotations


class CorridorKeyError(Exception):
    """Base exception for all CorridorKey errors."""


class ClipScanError(CorridorKeyError):
    """Raised when a clip path cannot be scanned or has an unrecognised structure.

    Also raised for permission errors and filesystem errors during scanning.
    """


class ClipLoadError(CorridorKeyError):
    """Raised when a clip cannot be loaded into a manifest.

    Examples: input directory has no frames, output directory cannot be
    created, clip structure is invalid after extraction.
    """

    def __init__(self, clip_name: str, detail: str) -> None:
        self.clip_name = clip_name
        self.detail = detail
        super().__init__(f"Clip '{clip_name}': load failed — {detail}")


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


class InferenceError(CorridorKeyError):
    """Raised when the model forward pass fails for a frame.

    Wraps RuntimeError from PyTorch (including CUDA errors). When the
    failure is specifically an out-of-memory condition, VRAMInsufficientError
    is raised instead.
    """

    def __init__(self, frame_index: int, detail: str) -> None:
        self.frame_index = frame_index
        self.detail = detail
        super().__init__(f"Inference failed at frame {frame_index}: {detail}")


class PostprocessError(CorridorKeyError):
    """Raised when postprocessing fails for a frame.

    Covers despill, despeckle, composite, and resize failures.
    """

    def __init__(self, frame_index: int, detail: str) -> None:
        self.frame_index = frame_index
        self.detail = detail
        super().__init__(f"Postprocess failed at frame {frame_index}: {detail}")


class WriteFailureError(CorridorKeyError):
    """Raised when writing an output file fails.

    Covers both cv2.imwrite returning False and OS-level write errors
    (disk full, permission denied).
    """

    def __init__(self, path: str, detail: str = "") -> None:
        self.path = path
        self.detail = detail
        msg = f"Write failed: {path}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class VRAMInsufficientError(CorridorKeyError):
    """Raised when a CUDA out-of-memory error occurs during inference.

    Indicates the GPU does not have enough VRAM for the current configuration.
    Try reducing img_size, enabling tiled refiner mode, or using a lower
    model_precision.
    """

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        msg = "Insufficient VRAM for inference"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class DeviceError(CorridorKeyError):
    """Raised when the requested compute device is unavailable."""


class ModelError(CorridorKeyError):
    """Raised when a model cannot be downloaded, loaded, or fails verification."""


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


class EngineError(CorridorKeyError):
    """Raised when the Engine detects a contract violation before processing starts.

    Examples: required slot unfilled, plugin type mismatch, plugin config invalid.
    """


class AlphaGeneratorError(CorridorKeyError):
    """Raised when a clip needs alpha but no AlphaGenerator is registered,
    or when the registered generator returns an invalid manifest."""

    def __init__(self, clip_name: str) -> None:
        self.clip_name = clip_name
        super().__init__(
            f"Clip '{clip_name}' requires alpha frames but no AlphaGenerator is registered. "
            "Register one via engine.set_alpha_generator() before calling engine.run()."
        )
