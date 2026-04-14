"""CorridorKey — public API.

Single import surface for any interface (CLI, GUI, TUI, plugin).

Quickstart::

    from corridorkey import Engine, load_config

    config = load_config()
    engine = Engine(config)
    engine.set_alpha_generator(MyAlphaGenerator())
    engine.on("frame_done", lambda i, total: print(f"{i}/{total}"))
    stats = engine.run([Path("/clips")])
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("corridorkey")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from corridorkey.engine import Engine
from corridorkey.errors import (
    AlphaGeneratorError,
    ClipScanError,
    CorridorKeyError,
    DeviceError,
    EngineError,
    ExtractionError,
    FrameMismatchError,
    FrameReadError,
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
from corridorkey.runtime.job_stats import JobStats
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.scanner.contracts import Clip, SkippedClip

__all__ = [
    "__version__",
    # Config
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
    # Engine
    "Engine",
    "JobStats",
    # Data contracts (flow through events)
    "Clip",
    "SkippedClip",
    "ClipManifest",
    # Alpha slot protocol
    "AlphaGenerator",
    # Errors
    "CorridorKeyError",
    "EngineError",
    "AlphaGeneratorError",
    "DeviceError",
    "ModelError",
    "ClipScanError",
    "ExtractionError",
    "FrameMismatchError",
    "JobCancelledError",
    "FrameReadError",
    "WriteFailureError",
    "VRAMInsufficientError",
]
