"""CorridorKey pipeline — public API.

Single import surface for any interface (CLI, GUI, TUI, plugin).
Do not import from submodules directly.

Startup::

    config = load_config()
    engine = Engine(config)

See ENGINE.md and PUBLIC_API.md for full integration guidance.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("corridorkey")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from corridorkey.errors import (
    AlphaGeneratorError,
    ClipScanError,
    CorridorKeyError,
    DeviceError,
    EngineError,
    ExtractionError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    ModelError,
    VRAMInsufficientError,
    WriteFailureError,
)
from corridorkey.infra import (
    APP_NAME,
    CorridorKeyConfig,
    GPUInfo,
    InferenceSettings,
    LoggingSettings,
    PostprocessSettings,
    PreprocessSettings,
    SettingsMetadata,
    WriterSettings,
    clear_device_cache,
    default_checkpoint_path,
    detect_gpu,
    ensure_config_file,
    get_config_path,
    load_config,
    load_config_with_metadata,
    resolve_device,
    resolve_devices,
    write_config,
)
from corridorkey.protocols import AlphaGenerator
from corridorkey.runtime.clip_state import ClipRecord, ClipState, FrameRange, get_clip_state
from corridorkey.stages.inference import (
    InferenceConfig,
    InferenceResult,
    ModelBackend,
    load_model_backend,
)
from corridorkey.stages.loader import (
    ClipManifest,
    attach_alpha,
    load,
)
from corridorkey.stages.loader.validator import list_frames
from corridorkey.stages.postprocessor import PostprocessConfig, ProcessedFrame, postprocess_frame
from corridorkey.stages.preprocessor import (
    FrameMeta,
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from corridorkey.stages.scanner import Clip, ScanResult, SkippedClip, scan
from corridorkey.stages.writer import WriteConfig, write_frame

__all__ = [
    "__version__",
    # Configuration
    "load_config",
    "load_config_with_metadata",
    "SettingsMetadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    # Infrastructure
    "APP_NAME",
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    "GPUInfo",
    "default_checkpoint_path",
    # Clip lifecycle
    "scan",
    "load",
    "attach_alpha",
    "list_frames",
    "Clip",
    "ScanResult",
    "SkippedClip",
    "ClipManifest",
    "ClipRecord",
    "ClipState",
    "FrameRange",
    "get_clip_state",
    # Protocols
    "AlphaGenerator",
    "ModelBackend",
    # Layer 2 — frame loop
    "load_model_backend",
    "preprocess_frame",
    "postprocess_frame",
    "write_frame",
    "PreprocessedFrame",
    "FrameMeta",
    "InferenceResult",
    "ProcessedFrame",
    # Internal stage configs (advanced — prefer config.to_*_config())
    "PreprocessConfig",
    "InferenceConfig",
    "PostprocessConfig",
    "WriteConfig",
    # Errors
    "CorridorKeyError",
    "EngineError",
    "AlphaGeneratorError",
    "DeviceError",
    "ModelError",
    "ClipScanError",
    "ExtractionError",
    "FrameMismatchError",
    "InvalidStateTransitionError",
    "JobCancelledError",
    "FrameReadError",
    "WriteFailureError",
    "VRAMInsufficientError",
]
