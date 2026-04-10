"""CorridorKey pipeline — public API.

Single import surface for any interface (CLI, GUI, TUI, plugin).
Do not import from submodules directly.

Two integration layers are available. See the integration guide in the docs
for full usage examples: docs/corridorkey/integration/
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("corridorkey")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from corridorkey.errors import (
    ClipScanError,
    CorridorKeyError,
    DeviceError,
    ExtractionError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    ModelError,
    VRAMInsufficientError,
    WriteFailureError,
)
from corridorkey.events import PipelineEvents
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
    ensure_model,
    get_config_path,
    load_config,
    load_config_with_metadata,
    resolve_device,
    resolve_devices,
    setup_logging,
    write_config,
)
from corridorkey.runtime.clip_state import ClipEntry, ClipState, InOutRange, resolve_clip_state
from corridorkey.runtime.runner import PipelineConfig, Runner
from corridorkey.stages.inference import (
    InferenceConfig,
    InferenceResult,
    ModelBackend,
    load_backend,
)
from corridorkey.stages.loader import (
    ClipManifest,
    load,
    resolve_alpha,
)
from corridorkey.stages.loader.validator import list_clip_frames
from corridorkey.stages.postprocessor import PostprocessConfig, PostprocessedFrame, postprocess_frame
from corridorkey.stages.preprocessor import (
    FrameMeta,
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from corridorkey.stages.scanner import Clip, ScanResult, SkippedPath, scan
from corridorkey.stages.writer import WriteConfig, write_frame

__all__ = [
    "__version__",
    # -- Shared foundation (both layers) -------------------------------- #
    "load_config",
    "load_config_with_metadata",
    "SettingsMetadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    "APP_NAME",
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    "GPUInfo",
    "ensure_model",
    "default_checkpoint_path",
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    "scan",
    "load",
    "resolve_alpha",
    "Clip",
    "ScanResult",
    "SkippedPath",
    "ClipManifest",
    "ClipEntry",
    "ClipState",
    "InOutRange",
    "resolve_clip_state",
    # Layer 1 — Managed pipeline
    "Runner",
    "PipelineConfig",
    "PipelineEvents",
    # Layer 2 — Frame loop
    "load_backend",
    "preprocess_frame",
    "postprocess_frame",
    "write_frame",
    "PreprocessConfig",
    "InferenceConfig",
    "PostprocessConfig",
    "WriteConfig",
    "list_clip_frames",
    "PreprocessedFrame",
    "FrameMeta",
    "InferenceResult",
    "PostprocessedFrame",
    "ModelBackend",
    # Errors
    "CorridorKeyError",
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
