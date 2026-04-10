"""CorridorKey pipeline — public API.

Single import surface for any interface (CLI, GUI, TUI, plugin).
Do not import from submodules directly.

Configuration entry point
-------------------------
``CorridorKeyConfig`` is the single entry point for all configuration, in both
integration layers. Load it once at startup, then pass it (or the configs it
produces) into the pipeline. Never construct internal stage configs directly.

    config = load_config()          # reads TOML + env, resolves "auto" values
    setup_logging(config)

Layer 1 — Managed runtime (recommended)
----------------------------------------
Hand the whole clip to the runner. Device resolution, queue management, and
multi-GPU dispatch are handled for you.

    manifest = load(clip)
    pipeline_config = config.to_pipeline_config(device=resolve_device(config.device))
    Runner(manifest, pipeline_config, events=PipelineEvents(...)).run()

Layer 2 — Frame loop (custom pipelines)
-----------------------------------------
Use individual stage functions when you need to own the loop — e.g. to feed
frames to a GUI preview, integrate with a host application's threading model,
or swap out a stage entirely.

    manifest = load(clip)
    backend = load_backend(config.to_inference_config(device=device))
    preprocess_cfg = config.to_preprocess_config(device=device)
    postprocess_cfg = config.to_postprocess_config()
    write_cfg = config.to_writer_config(manifest.output_dir)

    for i in range(*manifest.frame_range):
        pre = preprocess_frame(manifest, i, preprocess_cfg)
        result = backend.run(pre)
        post = postprocess_frame(result, postprocess_cfg)
        write_frame(post, write_cfg)

See docs/corridorkey/integration/ for full usage examples.
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
    "setup_logging",
    "APP_NAME",
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    "GPUInfo",
    "ensure_model",
    "default_checkpoint_path",
    # Clip lifecycle
    "scan",
    "load",
    "resolve_alpha",
    "list_clip_frames",
    "Clip",
    "ScanResult",
    "SkippedPath",
    "ClipManifest",
    "ClipEntry",
    "ClipState",
    "InOutRange",
    "resolve_clip_state",
    # Layer 1 — managed runtime
    "Runner",
    "PipelineConfig",
    "PipelineEvents",
    # Layer 2 — frame loop
    "load_backend",
    "preprocess_frame",
    "postprocess_frame",
    "write_frame",
    "PreprocessedFrame",
    "FrameMeta",
    "InferenceResult",
    "PostprocessedFrame",
    "ModelBackend",
    # Internal stage configs (advanced — prefer config.to_*_config())
    "PreprocessConfig",
    "InferenceConfig",
    "PostprocessConfig",
    "WriteConfig",
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
